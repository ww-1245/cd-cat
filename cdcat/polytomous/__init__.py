"""多级评分认知诊断模型（顺序 G-DINA / Sequential G-DINA）。

为什么需要它
------------
到此为止所有模型都假定作答是 **0/1**（对/错）。但真实测验里大量题目是
多级记分的：解答题按步骤给分、量表题 0-3 分、作文分档。把它们硬压成
二分，信息量是白白丢掉的。

模型设定（Ma & de la Torre, 2016）
----------------------------------
把一道多级题拆成 **M 个顺序"步骤"**（M = 满分值）。
被试得 k 分 ⟺ **通过了前 k 个步骤**。每个步骤 m 是一个二分过程，
其"通过概率"由该步骤所需的属性集合 :math:`q_{jm}` 按 DINA 规则决定：

.. math::
    s_{jm}(\\alpha) = (1-\\text{slip}_{jm})^{\\eta_{jm}}
                      \\cdot \\text{guess}_{jm}^{\\,1-\\eta_{jm}},
    \\quad \\eta_{jm} = \\prod_{k \\in q_{jm}} \\alpha_k

于是得分为 k 的概率为

.. math::
    P(X_j = k \\mid \\alpha) =
      \\Big(\\prod_{l=1}^{k} s_{jl}\\Big)\\big(1 - s_{j,k+1}\\big),
    \\quad s_{j,M+1} \\equiv 0

关键实现技巧
------------
令 :math:`Y_{jm} = \\mathbb{1}[X_j \\ge m]`（"是否走到第 m 步"），
则 :math:`Y_{j\\cdot}` 是一组**普通二分数据**，且 :math:`P(Y_{jm}=1\\mid\\alpha) = s_{jm}`。
也就是说：**多级题展开成二分步骤题后，可以直接复用已有的 DINA EM 标定器**，
无需重写任何 EM 代码。这正是本模块 :func:`calibrate_polytomous` 的做法。

与主流程的衔接
--------------
:class:`SequentialGDINA` 继承 :class:`~cdcat.models.CDM`，因此
``AttributeEstimator``、``SelectionCriterion``、``CDCATSimulator``、
终止规则、曝光控制**全部可以直接复用** —— 只需换模型与选题准则。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models import CDModel
from ..qmatrix import QMatrix
from ..utils import enumerate_attribute_patterns

__all__ = [
    "PolytomousQMatrix",
    "SequentialGDINA",
    "expand_to_steps",
    "collapse_step_responses",
    "calibrate_polytomous",
    "build_polytomous_bank",
    "PolySelectionCriterion",
    "PolyKL",
    "PolyPWKL",
    "POLY_SELECTION_REGISTRY",
    "create_poly_selector",
]


# ====================================================================== #
# 1. 多级 Q 矩阵（步骤展开）
# ====================================================================== #
@dataclass(eq=False)
class PolytomousQMatrix:
    """多级记分题的 Q 矩阵（按"步骤"展开）。

    第 j 题被拆成 :math:`M_j` 个步骤，得 :math:`M_j + 1` 个分数档（0..M_j）。
    步骤按顺序排列：通过前 k 步即得 k 分。

    Parameters
    ----------
    steps : ndarray, shape (S, K)
        每个步骤所需的属性集合（0/1）。
    item_index : ndarray, shape (S,)
        每个步骤属于哪道题（0 基）。
    n_attrs : int
        属性个数 K。
    attr_names, item_names : list of str, optional

    Attributes
    ----------
    n_items : int
        题目数 J（不是步骤数）。
    n_steps : int
        步骤总数 S。
    n_categories : ndarray, shape (J,)
        每题的分数档数 = 步骤数 + 1。
    """

    steps: np.ndarray
    item_index: np.ndarray
    n_attrs: int
    attr_names: Optional[List[str]] = None
    item_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.steps = np.asarray(self.steps, dtype=np.int8)
        if self.steps.ndim != 2:
            raise ValueError("steps 必须是二维矩阵 (S, K)")
        self.item_index = np.asarray(self.item_index, dtype=np.int64).ravel()
        if self.steps.shape[0] != self.item_index.size:
            raise ValueError("steps 行数与 item_index 长度必须一致")
        self.n_attrs = int(self.steps.shape[1])
        if self.item_index.size and int(self.item_index.max()) + 1 > 200:
            raise ValueError("题目数异常，请检查 item_index")
        self.item_names = list(self.item_names) if self.item_names else None
        self.attr_names = list(self.attr_names) if self.attr_names else None

        # 题目必须按顺序、连续地拥有步骤
        uniq = np.unique(self.item_index)
        if not np.array_equal(uniq, np.arange(uniq.size)):
            raise ValueError("item_index 必须是从 0 开始连续编号")
        offsets = []
        for j in uniq:
            pos = np.where(self.item_index == j)[0]
            if not np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)):
                raise ValueError(f"第 {j} 题的步骤在 steps 中不连续")
            if pos.size < 1:
                raise ValueError(f"第 {j} 题至少要有 1 个步骤")
            offsets.append((int(pos[0]), int(pos[-1]) + 1))
        self._offsets: List[Tuple[int, int]] = offsets

        self._cat_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # 基本属性
    # ------------------------------------------------------------------ #
    @property
    def n_items(self) -> int:
        return len(self._offsets)

    @property
    def n_steps(self) -> int:
        return int(self.steps.shape[0])

    @property
    def n_patterns(self) -> int:
        return 1 << self.n_attrs

    @property
    def n_categories(self) -> np.ndarray:
        """每题分数档数，形状 ``(J,)``。"""
        return np.array([e - s + 1 for s, e in self._offsets], dtype=np.int64)

    @property
    def max_categories(self) -> int:
        return int(self.n_categories.max())

    def step_slice(self, item: int) -> Tuple[int, int]:
        """第 item 题在 ``steps`` 中占用的行区间 ``[start, end)``。"""
        return self._offsets[item]

    def step_matrix_of(self, item: int) -> np.ndarray:
        s, e = self._offsets[item]
        return self.steps[s:e]

    def item_attributes(self, item: int) -> List[int]:
        """第 item 题考查的属性（各步骤的并集）。"""
        return np.where(self.step_matrix_of(item).any(axis=0))[0].tolist()

    def attribute_patterns(self) -> np.ndarray:
        return enumerate_attribute_patterns(self.n_attrs)

    def union_q_matrix(self) -> QMatrix:
        """并集意义下的二分 Q 矩阵（每题一行），用于与既有工具衔接。"""
        m = np.zeros((self.n_items, self.n_attrs), dtype=np.int8)
        for j in range(self.n_items):
            m[j] = self.step_matrix_of(j).any(axis=0).astype(np.int8)
        return QMatrix(m, attr_names=self.attr_names, item_names=self.item_names)

    # ------------------------------------------------------------------ #
    # 构造快捷方式
    # ------------------------------------------------------------------ #
    @classmethod
    def from_steps(
        cls,
        steps: np.ndarray,
        item_index: Sequence[int],
        attr_names: Optional[Sequence[str]] = None,
        item_names: Optional[Sequence[str]] = None,
    ) -> "PolytomousQMatrix":
        """由显式步骤矩阵构造。"""
        return cls(np.asarray(steps, dtype=np.int8),
                   np.asarray(item_index, dtype=np.int64),
                   np.asarray(steps).shape[1], attr_names, item_names)

    @classmethod
    def sequential(
        cls,
        matrix: np.ndarray,
        n_categories: Optional[Sequence[int]] = None,
        attr_names: Optional[Sequence[str]] = None,
        item_names: Optional[Sequence[str]] = None,
    ) -> "PolytomousQMatrix":
        """由二分 Q 矩阵构造"属性即步骤"的顺序模型。

        第 j 题考查的属性（按列升序）依次作为步骤 1、2、…；
        默认分数档数 = 属性数 + 1。

        Parameters
        ----------
        matrix : ndarray, shape (J, K)
            二分 Q 矩阵。
        n_categories : sequence of int, optional
            每题分数档数。给定时，只取前 ``n_categories[j]-1`` 个属性作步骤
            （多余的属性被忽略），不足则报错。
        """
        m = np.asarray(matrix, dtype=np.int8)
        if m.ndim != 2:
            raise ValueError("matrix 必须是二维")
        j, k = m.shape
        if n_categories is None:
            n_cat = m.sum(axis=1).astype(int) + 1
        else:
            n_cat = np.asarray(n_categories, dtype=int).ravel()
            if n_cat.size != j:
                raise ValueError(f"n_categories 长度应为 {j}")
            if np.any(n_cat < 2):
                raise ValueError("每题至少要有 2 个分数档")

        rows: List[np.ndarray] = []
        idx: List[int] = []
        for it in range(j):
            need = int(n_cat[it]) - 1
            cols = np.where(m[it] == 1)[0]
            if cols.size < need:
                raise ValueError(
                    f"第 {it} 题需要 {need} 个步骤（{n_cat[it]} 个分数档），"
                    f"但只考查 {cols.size} 个属性；请补充属性或减少分数档"
                )
            if cols.size > need:
                cols = cols[:need]
            for c in cols:
                row = np.zeros(k, dtype=np.int8)
                row[c] = 1
                rows.append(row)
                idx.append(it)
        return cls(np.array(rows, dtype=np.int8), np.array(idx, dtype=np.int64),
                   k, attr_names, item_names)

    @classmethod
    def from_binary_qmatrix(
        cls,
        q: QMatrix,
        n_categories: Optional[Sequence[int]] = None,
    ) -> "PolytomousQMatrix":
        """由 :class:`QMatrix` 构造。"""
        return cls.sequential(
            q.matrix, n_categories,
            attr_names=q.attr_names, item_names=q.item_names,
        )

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        cat = self.n_categories
        lines = [
            "=" * 64,
            "多级记分 Q 矩阵（顺序展开）",
            "=" * 64,
            f"  题目数 J          : {self.n_items}",
            f"  属性数 K          : {self.n_attrs}",
            f"  步骤总数 S        : {self.n_steps}",
            f"  分数档分布        : {dict(zip(*np.unique(cat, return_counts=True)))}",
            f"  最大分数档        : {self.max_categories}",
        ]
        lines.append("-" * 64)
        for it in range(min(self.n_items, 8)):
            sm = self.step_matrix_of(it)
            desc = " | ".join(
                "+".join(str(x) for x in np.where(r == 1)[0]) or "∅" for r in sm
            )
            lines.append(f"  {it + 1:>3}. {cat[it]} 档 步骤: {desc}")
        if self.n_items > 8:
            lines.append(f"  ... 另有 {self.n_items - 8} 题")
        lines.append("=" * 64)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PolytomousQMatrix(J={self.n_items}, K={self.n_attrs}, "
                f"S={self.n_steps}, max_cat={self.max_categories})")


# ====================================================================== #
# 2. 顺序 G-DINA 模型
# ====================================================================== #
class SequentialGDINA(CDModel):
    """顺序 G-DINA 模型（多级记分）。

    参数按**步骤**给出：``slip`` / ``guess`` 长度均为步骤总数 S。

    Notes
    -----
    ``prob_matrix()`` 返回 :math:`P(X_j \\ge 1\\mid\\alpha)`（至少得 1 分），
    以便与既有二分工具兼容；多级信息通过 :meth:`category_probs` 提供。
    """

    name = "SEQUENTIAL-GDINA"
    ideal_mode = "conjunctive"
    param_names = ("slip", "guess")

    def __init__(self, pq: PolytomousQMatrix, slip=None, guess=None) -> None:
        # self.q 用并集 Q 矩阵，使基类的 n_items / n_patterns / attribute_patterns 可用
        super().__init__(pq.union_q_matrix())
        self.pq = pq
        self._cat_cache: Optional[np.ndarray] = None
        s = pq.n_steps
        self.slip = np.full(s, 0.15) if slip is None else np.asarray(slip, dtype=np.float64).ravel()
        self.guess = np.full(s, 0.15) if guess is None else np.asarray(guess, dtype=np.float64).ravel()
        self._check_params()

    def _check_params(self) -> None:
        s = self.pq.n_steps
        if len(self.slip) != s or len(self.guess) != s:
            raise ValueError(f"slip / guess 长度需为步骤数 {s}")
        if np.any((self.slip < 0) | (self.slip > 1)) or np.any((self.guess < 0) | (self.guess > 1)):
            raise ValueError("slip / guess 必须在 [0, 1] 区间")

    @property
    def n_steps(self) -> int:
        return self.pq.n_steps

    @property
    def n_categories(self) -> np.ndarray:
        return self.pq.n_categories

    # ------------------------------------------------------------------ #
    # 概率计算
    # ------------------------------------------------------------------ #
    def step_probs(self) -> np.ndarray:
        """各步骤的"通过概率"矩阵，形状 ``(2^K, S)``。

        :math:`s_{jm}(\\alpha) = (1-\\text{slip})^{\\eta}\\text{guess}^{1-\\eta}`，
        其中 :math:`\\eta` 是该步骤所需属性在 α 上的连接型理想反应。
        """
        patterns = self.pq.attribute_patterns().astype(np.float64)     # (2^K, K)
        steps = self.pq.steps.astype(np.float64)                       # (S, K)
        # η[α, m] = 1 当且仅当 α 覆盖步骤 m 的全部所需属性
        need = steps.sum(axis=1)                                       # (S,)
        covered = patterns @ steps.T                                   # (2^K, S)
        eta = (covered >= need[None, :] - 1e-9).astype(np.float64)
        eta[:, need == 0] = 1.0                                        # 空步骤：必通过
        p = (1.0 - self.slip)[None, :] ** eta * self.guess[None, :] ** (1.0 - eta)
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def _cat_cache_build(self) -> np.ndarray:
        """构造 (2^K, J, Cmax) 的分数分布张量（不足处补 0）。"""
        s_all = self.step_probs()
        j = self.pq.n_items
        cmax = self.pq.max_categories
        out = np.zeros((self.n_patterns, j, cmax), dtype=np.float64)
        for it in range(j):
            a, b = self.pq.step_slice(it)
            sp = s_all[:, a:b]                                  # (2^K, M)
            if sp.shape[1] == 0:
                out[:, it, 0] = 1.0
                continue
            cum = np.cumprod(sp, axis=1)                        # S_1..S_M
            n_cat = sp.shape[1] + 1
            probs = np.empty((self.n_patterns, n_cat), dtype=np.float64)
            probs[:, 0] = 1.0 - sp[:, 0]
            if n_cat > 2:
                probs[:, 1:n_cat - 1] = cum[:, :-1] * (1.0 - sp[:, 1:])
            probs[:, n_cat - 1] = cum[:, -1]
            out[:, it, :n_cat] = probs
        return out

    def category_probs(self) -> np.ndarray:
        """分数分布张量，形状 ``(2^K, J, Cmax)``；超出该题档数的位置为 0。"""
        if self._cat_cache is None:
            self._cat_cache = self._cat_cache_build()
        return self._cat_cache

    def prob_matrix(self) -> np.ndarray:
        """``P(X_j >= 1 | α)``，形状 ``(2^K, J)``（与二分工具兼容的视图）。"""
        cat = self.category_probs()
        return 1.0 - cat[:, :, 0]

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"slip": self.slip.copy(), "guess": self.guess.copy()}

    def set_params(self, slip=None, guess=None, **kwargs) -> "SequentialGDINA":
        if slip is not None:
            self.slip = np.asarray(slip, dtype=np.float64).ravel()
        if guess is not None:
            self.guess = np.asarray(guess, dtype=np.float64).ravel()
        self._check_params()
        self.clear_cache()
        return self

    def clear_cache(self) -> None:
        super().clear_cache()
        self._cat_cache = None

    # ------------------------------------------------------------------ #
    # 似然与模拟（覆盖基类的二分实现）
    # ------------------------------------------------------------------ #
    def log_likelihood(self, responses, item_indices=None) -> np.ndarray:
        """多级似然的对数，形状 ``(2^K,)``。

        ``responses`` 为该被试各题的**分数**（0..M_j）。
        """
        resp = np.asarray(responses, dtype=np.float64).ravel()
        if item_indices is None:
            item_indices = np.arange(resp.shape[0])
        item_indices = np.asarray(item_indices, dtype=np.int64).ravel()
        if resp.size != item_indices.size:
            raise ValueError("responses 与 item_indices 长度必须一致")
        if resp.size == 0:
            return np.zeros(self.n_patterns, dtype=np.float64)

        cat = self.category_probs()
        ll = np.zeros(self.n_patterns, dtype=np.float64)
        for t, it in enumerate(item_indices):
            n_cat = int(self.pq.n_categories[it])
            k = int(np.rint(resp[t]))
            if k < 0 or k >= n_cat:
                raise ValueError(
                    f"第 {it} 题的作答 {resp[t]} 超出分数范围 [0, {n_cat - 1}]"
                )
            ll += np.log(np.clip(cat[:, it, k], 1e-300, None))
        return ll

    def simulate(self, alpha, item_indices=None, rng=None) -> np.ndarray:
        """按多级分布生成作答（返回分数 0..M_j），dtype int8。"""
        if rng is None:
            rng = np.random.default_rng()
        alpha = np.asarray(alpha).ravel()
        pat = int(sum(int(a) << (self.pq.n_attrs - 1 - i) for i, a in enumerate(alpha)))
        if item_indices is None:
            item_indices = np.arange(self.pq.n_items)
        item_indices = np.asarray(item_indices, dtype=np.int64).ravel()

        cat = self.category_probs()[pat]              # (J, Cmax)
        out = np.zeros(len(item_indices), dtype=np.int8)
        for t, it in enumerate(item_indices):
            n_cat = int(self.pq.n_categories[it])
            p = cat[it, :n_cat]
            p = p / p.sum()
            out[t] = int(rng.choice(n_cat, p=p))
        return out

    def item_information(self, prior: Optional[np.ndarray] = None) -> np.ndarray:
        """各题的期望后验熵下降（多级版题目信息量），形状 ``(J,)``。"""
        s = self.n_patterns
        pr = np.full(s, 1.0 / s) if prior is None else np.asarray(prior, dtype=np.float64).ravel()
        pr = pr / pr.sum()
        cat = np.clip(self.category_probs(), 1e-12, None)
        info = np.zeros(self.pq.n_items)
        for it in range(self.pq.n_items):
            n_cat = int(self.pq.n_categories[it])
            h0 = float(-(pr * np.log(pr + 1e-300)).sum())
            h1 = 0.0
            for k in range(n_cat):
                pk = cat[:, it, k]
                w = pr * pk
                tot = w.sum()
                if tot <= 1e-300:
                    continue
                post = w / tot
                h1 += tot * float(-(post * np.log(post + 1e-300)).sum())
            info[it] = h0 - h1
        return info

    def __repr__(self) -> str:
        return (f"SequentialGDINA(J={self.pq.n_items}, K={self.pq.n_attrs}, "
                f"S={self.pq.n_steps})")


# ====================================================================== #
# 3. 多级 -> 二分步骤的展开与回收
# ====================================================================== #
def expand_to_steps(responses: np.ndarray, pq: PolytomousQMatrix) -> np.ndarray:
    """把多级作答矩阵展开成"是否走到第 m 步"的二分矩阵。

    Parameters
    ----------
    responses : ndarray, shape (n, J)
        多级分数（0..M_j），允许 ``np.nan`` 表示缺失。
    pq : PolytomousQMatrix

    Returns
    -------
    ndarray, shape (n, S), dtype float
        ``Y[i, m] = 1[X[i, item_of(m)] >= rank_of(m)]``；对应题目缺失时为 NaN。
    """
    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = resp.shape
    if j != pq.n_items:
        raise ValueError(f"作答矩阵列数 ({j}) 与题目数 ({pq.n_items}) 不一致")

    out = np.full((n, pq.n_steps), np.nan, dtype=np.float64)
    for it in range(j):
        a, b = pq.step_slice(it)
        col = resp[:, it]
        for m in range(a, b):
            rank = m - a + 1                       # 该题内的第几个步骤
            out[:, m] = np.where(np.isfinite(col), (col >= rank).astype(np.float64), np.nan)
    return out


def collapse_step_responses(step_responses: np.ndarray, pq: PolytomousQMatrix) -> np.ndarray:
    """:func:`expand_to_steps` 的逆运算：步骤二分作答 -> 多级分数。"""
    y = np.asarray(step_responses, dtype=np.float64)
    n = y.shape[0]
    out = np.zeros((n, pq.n_items), dtype=np.float64)
    for it in range(pq.n_items):
        a, b = pq.step_slice(it)
        block = y[:, a:b]
        out[:, it] = np.nansum(block, axis=1)
    return out


# ====================================================================== #
# 4. 标定：顺序结构的专用 MMLE-EM
# ====================================================================== #
def _step_group_masks(pq: PolytomousQMatrix) -> List[Tuple[int, int, np.ndarray]]:
    """预处理每个步骤的 ``(题目, 步内序号, η 掩码)``。"""
    patterns = pq.attribute_patterns()
    out: List[Tuple[int, int, np.ndarray]] = []
    for it in range(pq.n_items):
        a, b = pq.step_slice(it)
        for m in range(a, b):
            rank = m - a + 1
            req = np.where(pq.steps[m] == 1)[0]
            eta = (patterns[:, req].all(axis=1) if req.size
                   else np.ones(patterns.shape[0], dtype=bool))
            out.append((it, rank, eta))
    return out


def _steps_to_category_probs(
    pq: PolytomousQMatrix, slip: np.ndarray, guess: np.ndarray
) -> np.ndarray:
    """由步骤参数算出 ``(2^K, J, Cmax)`` 的分数分布（与模型内实现一致）。"""
    patterns = pq.attribute_patterns().astype(np.float64)
    steps = pq.steps.astype(np.float64)
    need = steps.sum(axis=1)
    covered = patterns @ steps.T
    eta = (covered >= need[None, :] - 1e-9).astype(np.float64)
    eta[:, need == 0] = 1.0
    s_all = np.clip((1.0 - slip)[None, :] ** eta * guess[None, :] ** (1.0 - eta), 1e-9, 1 - 1e-9)

    p = pq.n_patterns
    cmax = pq.max_categories
    out = np.zeros((p, pq.n_items, cmax), dtype=np.float64)
    for it in range(pq.n_items):
        a, b = pq.step_slice(it)
        sp = s_all[:, a:b]
        if sp.shape[1] == 0:
            out[:, it, 0] = 1.0
            continue
        cum = np.cumprod(sp, axis=1)
        n_cat = sp.shape[1] + 1
        probs = np.empty((p, n_cat), dtype=np.float64)
        probs[:, 0] = 1.0 - sp[:, 0]
        if n_cat > 2:
            probs[:, 1:n_cat - 1] = cum[:, :-1] * (1.0 - sp[:, 1:])
        probs[:, n_cat - 1] = cum[:, -1]
        out[:, it, :n_cat] = probs
    return out


def _polytomous_em(
    pq: PolytomousQMatrix,
    responses: np.ndarray,
    init_slip: Optional[np.ndarray] = None,
    init_guess: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    estimate_prior: bool = True,
    prior_smoothing: float = 1e-6,
    bounds: Tuple[float, float] = (0.001, 0.499),
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], int, bool, float]:
    """顺序 G-DINA 的 MMLE-EM。

    **为什么不能直接把多级题展开成二分题喂给普通 DINA EM**：
    令 :math:`Y_{jm}=\\mathbb{1}[X_j \\ge m]`，则

    .. math::
        P(Y_{jm}=1\\mid\\alpha) = \\prod_{l \\le m} s_{jl}(\\alpha)

    它含有前面所有步骤的乘积，**不服从**以 :math:`q_{jm}` 为条件的 DINA
    （那只是 m=1 的特例）。正确做法是按**条件结构**估计：第 m 步的参数
    只用"已经走到第 m 步"（:math:`X_j \\ge m-1`）的被试，判"是否继续走到 m"。

    于是 M 步有闭式解：对第 m 步的潜在组 g ∈ {0,1}，

    .. math::
        \\hat p_{m,g} = \\frac{\\sum_i \\sum_{\\alpha \\in g}
            h_i(\\alpha)\\,\\mathbb{1}[X_{ij} \\ge m]}
            {\\sum_i \\sum_{\\alpha \\in g} h_i(\\alpha)\\,
             \\mathbb{1}[X_{ij} \\ge m-1]}

    :math:`\\text{slip}_m = 1-\\hat p_{m,1}`，:math:`\\text{guess}_m = \\hat p_{m,0}`。
    """
    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = resp.shape
    if j != pq.n_items:
        raise ValueError(f"作答矩阵列数 ({j}) 与题目数 ({pq.n_items}) 不一致")

    p = pq.n_patterns
    step_info = _step_group_masks(pq)
    s_total = pq.n_steps

    slip = np.full(s_total, 0.15) if init_slip is None else np.asarray(init_slip, float).copy()
    guess = np.full(s_total, 0.15) if init_guess is None else np.asarray(init_guess, float).copy()
    pi = (np.full(p, 1.0 / p) if prior is None
          else np.asarray(prior, dtype=np.float64).ravel().copy())
    pi = pi / pi.sum()

    obs = np.isfinite(resp)
    codes = np.rint(np.nan_to_num(resp, nan=0.0)).astype(np.int64)
    for it in range(pq.n_items):
        codes[:, it] = np.clip(codes[:, it], 0, int(pq.n_categories[it]) - 1)

    lo_p, hi_p = 1.0 - bounds[1], 1.0 - bounds[0]     # 通过概率的界
    ll_hist: List[float] = []
    ll_prev = -np.inf
    converged = False
    iters = 0

    for iters in range(1, max_iter + 1):
        # ---------------- E 步 ----------------
        cat = _steps_to_category_probs(pq, slip, guess)      # (P, J, Cmax)
        loglik = np.tile(np.log(pi + 1e-300)[None, :], (n, 1))
        for jj in range(j):
            rows = np.where(obs[:, jj])[0]
            if rows.size == 0:
                continue
            c = int(pq.n_categories[jj])
            k = codes[rows, jj]
            # cat[:, jj, :c].T -> (c, P)，按观测到的档位取行
            loglik[rows] += np.log(np.clip(cat[:, jj, :c].T[k], 1e-300, None))

        mx = loglik.max(axis=1, keepdims=True)
        a = np.exp(loglik - mx)
        denom = a.sum(axis=1, keepdims=True)
        w = a / denom                                        # h_i(α)  (n, P)
        ll = float((mx.ravel() + np.log(denom.ravel())).sum())
        ll_hist.append(ll)

        # ---------------- M 步 ----------------
        for m, (jj, rank, eta) in enumerate(step_info):
            col = codes[:, jj]
            seen = obs[:, jj]
            reached = seen & (col >= rank - 1)
            if not reached.any():
                continue
            wm = w[reached]
            den1 = float(wm[:, eta].sum())
            den0 = float(wm[:, ~eta].sum())
            passed = seen & (col >= rank)
            wp = w[passed]
            if den1 > 1e-8:
                num1 = float(wp[:, eta].sum())
                slip[m] = 1.0 - float(np.clip(num1 / den1, lo_p, hi_p))
            if den0 > 1e-8:
                num0 = float(wp[:, ~eta].sum())
                guess[m] = float(np.clip(num0 / den0, bounds[0], bounds[1]))

        # 经验贝叶斯先验
        if estimate_prior:
            new_pi = w.mean(axis=0) + prior_smoothing
            pi = new_pi / new_pi.sum()

        if verbose and iters % 25 == 0:
            print(f"  iter {iters:4d}  logL={ll:.4f}  dLL={ll - ll_prev:+.2e}")

        if np.isfinite(ll_prev) and abs(ll - ll_prev) < tol:
            converged = True
            break
        ll_prev = ll

    return slip, guess, pi, ll_hist, iters, converged, float(ll_hist[-1] if ll_hist else 0.0)


def calibrate_polytomous(
    pq: PolytomousQMatrix,
    responses: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    estimate_prior: Optional[bool] = True,
    prior_smoothing: float = 1e-6,
    bounds: Tuple[float, float] = (0.001, 0.499),
    init_model: Optional[SequentialGDINA] = None,
    verbose: bool = False,
) -> Tuple[SequentialGDINA, object]:
    """标定顺序 G-DINA 模型（步骤级 MMLE-EM，条件结构）。

    Parameters
    ----------
    pq : PolytomousQMatrix
    responses : ndarray, shape (n, J)
        多级分数（0..M_j）。``np.nan`` 表示该题缺失，该题不参与该被试的似然。
    estimate_prior : bool
        是否用经验贝叶斯估计 π(α)（默认开启）。

    Returns
    -------
    (model, calibration_result)
        ``calibration_result`` 是 :class:`~cdcat.calibration.CalibrationResult`，
        其中 ``slip`` / ``guess`` 为**步骤级**参数（长度 = 步骤总数 S）。

    Notes
    -----
    注意本函数**不是**"把多级展开成二分再调用普通 DINA EM"——
    那样做会错误地把"尚未走到该步"的被试当成答错。详见 :func:`_polytomous_em`。
    """
    from ..calibration import CalibrationResult

    slip, guess, pi, ll_hist, iters, converged, ll = _polytomous_em(
        pq, responses,
        init_slip=(init_model.slip if init_model is not None else None),
        init_guess=(init_model.guess if init_model is not None else None),
        max_iter=max_iter, tol=tol,
        estimate_prior=bool(estimate_prior),
        prior_smoothing=prior_smoothing, bounds=bounds,
        verbose=verbose,
    )
    model = SequentialGDINA(pq, slip=slip, guess=guess)

    n_item_params = 2 * pq.n_steps
    n_prior_params = (pq.n_patterns - 1) if estimate_prior else 0
    n_params = n_item_params + n_prior_params
    nobs = int(np.isfinite(np.asarray(responses, dtype=float)).sum())
    aic = -2.0 * ll + 2.0 * n_params
    bic = -2.0 * ll + np.log(max(nobs, 1)) * n_params
    aicc = (aic + 2.0 * n_params * (n_params + 1) / max(nobs - n_params - 1, 1))

    res = CalibrationResult(
        model_name="Sequential G-DINA (step-level)",
        slip=slip.copy(), guess=guess.copy(),
        log_likelihood_history=ll_hist,
        n_iterations=iters, converged=converged, log_likelihood=float(ll),
        aic=float(aic), bic=float(bic), aicc=float(aicc),
        n_params=int(n_params), n_item_params=int(n_item_params),
        n_prior_params=int(n_prior_params),
        attr_prior=pi.copy(), prior_estimated=bool(estimate_prior),
    )
    res.n_steps = pq.n_steps
    res.n_categories = pq.n_categories
    return model, res


# ====================================================================== #
# 5. 合成的多级题库（演示与测试用）
# ====================================================================== #
def build_polytomous_bank(
    n_items: int = 30,
    n_attrs: int = 4,
    n_categories: Sequence[int] = (2, 3, 4),
    slip_range: Tuple[float, float] = (0.05, 0.20),
    guess_range: Tuple[float, float] = (0.05, 0.20),
    seed: Optional[int] = None,
    attr_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """生成一个合成的多级记分题库（顺序结构）。

    Returns
    -------
    dict
        ``{"pq": PolytomousQMatrix, "model": SequentialGDINA,
        "attr_names": list}``
    """
    rng = np.random.default_rng(seed)
    attr_names = list(attr_names) if attr_names else [f"A{i + 1}" for i in range(n_attrs)]
    cats = np.asarray(n_categories, dtype=int).ravel()

    steps: List[np.ndarray] = []
    idx: List[int] = []
    for it in range(n_items):
        n_cat = int(rng.choice(cats))
        need = n_cat - 1
        # 步骤 1..need 依次对应"逐步掌握"的属性；先随机排序属性，再顺序取
        order = rng.permutation(n_attrs)[:need]
        for c in order:
            row = np.zeros(n_attrs, dtype=np.int8)
            row[c] = 1
            steps.append(row)
            idx.append(it)

    pq = PolytomousQMatrix(
        np.array(steps, dtype=np.int8),
        np.array(idx, dtype=np.int64),
        n_attrs,
        attr_names,
        [f"P{i + 1}" for i in range(n_items)],
    )
    s = pq.n_steps
    slip = rng.uniform(*slip_range, size=s)
    guess = rng.uniform(*guess_range, size=s)
    model = SequentialGDINA(pq, slip=slip, guess=guess)
    # 保证 1-slip > guess（单调性），否则把 guess 压到 slip 之下
    bad = (1.0 - slip) <= guess
    if bad.any():
        guess[bad] = np.clip(1.0 - slip[bad] - 0.05, 0.01, 0.45)
        model = SequentialGDINA(pq, slip=slip, guess=guess)
    return {"pq": pq, "model": model, "attr_names": attr_names,
            "slip": slip, "guess": guess}


# ====================================================================== #
# 6. 多级记分的选题准则
# ====================================================================== #
class PolySelectionCriterion:
    """多级记分选题准则基类。

    与二分准则的区别：KL 散度在**全部分数档**上计算，
    因此能把"答对/答错"之外的"部分掌握"信息用起来。
    """

    name: str = "poly-base"

    def __init__(self, model: SequentialGDINA, eps: float = 1e-12) -> None:
        if not isinstance(model, SequentialGDINA):
            raise TypeError("多级选题准则需要 SequentialGDINA 模型")
        self.model = model
        self.eps = float(eps)
        cat = np.clip(model.category_probs(), eps, None)
        self._cat = cat
        self._log_cat = np.log(cat)
        # 每题的有效档数
        self._n_cat = model.pq.n_categories

    # ---- 基类兼容 ----
    @property
    def n_patterns(self) -> int:
        return self._cat.shape[0]

    @property
    def n_items(self) -> int:
        return self._cat.shape[1]

    def clear_cache(self) -> None:
        self.model.clear_cache()
        cat = np.clip(self.model.category_probs(), self.eps, None)
        self._cat = cat
        self._log_cat = np.log(cat)
        self._n_cat = self.model.pq.n_categories

    def _kl_rows(self, item: int) -> np.ndarray:
        """返回该题的 KL 矩阵，形状 ``(2^K, 2^K)``。

        ``KL[a, b] = Σ_c P(c|α_a) log[ P(c|α_a) / P(c|α_b) ]``
        """
        p = self._cat[:, item, :int(self._n_cat[item])]         # (S, C)
        lp = self._log_cat[:, item, :int(self._n_cat[item])]
        ent = (p * lp).sum(axis=1)                              # (S,)
        cross = p @ lp.T                                        # (S, S)
        return ent[:, None] - cross

    def select(self, posterior, available_items=None, exclude=None) -> int:
        cand = self._candidates(available_items, exclude)
        if len(cand) == 0:
            raise RuntimeError("没有可供选择的题目")
        values = self.compute(posterior, cand)
        return int(cand[int(np.argmax(values))])

    def _candidates(self, available_items, exclude) -> np.ndarray:
        cand = (np.arange(self.n_items) if available_items is None
                else np.asarray(available_items, dtype=np.int64).ravel())
        if exclude is not None and len(exclude) > 0:
            excl = set(int(x) for x in np.asarray(exclude).ravel())
            cand = np.array([c for c in cand if int(c) not in excl], dtype=np.int64)
        return cand

    def compute(self, posterior, available_items=None) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


class PolyKL(PolySelectionCriterion):
    """多级 KL 准则：后验加权的模式对 KL 之和（越大越有信息）。"""

    name = "POLY-KL"

    def compute(self, posterior, available_items=None) -> np.ndarray:
        post = np.asarray(posterior, dtype=np.float64).ravel()
        cand = (np.arange(self.n_items) if available_items is None
                else np.asarray(available_items, dtype=np.int64).ravel())
        out = np.zeros(len(cand))
        for t, it in enumerate(cand):
            kl = self._kl_rows(int(it))            # (S, S)
            out[t] = float(post @ (kl @ post))
        return out


class PolyPWKL(PolySelectionCriterion):
    """多级后验加权 KL（PWKL）：以当前 MAP 估计为参照的加权 KL。"""

    name = "POLY-PWKL"

    def compute(self, posterior, available_items=None) -> np.ndarray:
        post = np.asarray(posterior, dtype=np.float64).ravel()
        cand = (np.arange(self.n_items) if available_items is None
                else np.asarray(available_items, dtype=np.int64).ravel())
        ref = int(np.argmax(post))
        out = np.zeros(len(cand))
        for t, it in enumerate(cand):
            kl = self._kl_rows(int(it))
            out[t] = float(post @ kl[:, ref])
        return out


class PolyGDI(PolySelectionCriterion):
    """多级 GDI：先验方差中被该题解释掉的期望部分（越大越好）。"""

    name = "POLY-GDI"

    def compute(self, posterior, available_items=None) -> np.ndarray:
        post = np.asarray(posterior, dtype=np.float64).ravel()
        cand = (np.arange(self.n_items) if available_items is None
                else np.asarray(available_items, dtype=np.int64).ravel())
        # 当前后验下各属性的方差
        pats = self.model.pq.attribute_patterns().astype(np.float64)
        mean = post @ pats
        var_post = (post @ (pats ** 2)) - mean ** 2
        out = np.zeros(len(cand))
        for t, it in enumerate(cand):
            n_cat = int(self._n_cat[it])
            p = self._cat[:, it, :n_cat]                     # (S, C)
            expected_var = np.zeros(pats.shape[1])
            for c in range(n_cat):
                w = post * p[:, c]
                tot = w.sum()
                if tot <= 1e-300:
                    continue
                sub = w / tot
                m1 = sub @ pats
                m2 = sub @ (pats ** 2)
                expected_var += tot * (m2 - m1 ** 2)
            out[t] = float((var_post - expected_var).sum())
        return out


POLY_SELECTION_REGISTRY = {
    "POLY-KL": PolyKL,
    "POLY-PWKL": PolyPWKL,
    "POLY-GDI": PolyGDI,
    # 允许直接写 KL / PWKL / GDI
    "KL": PolyKL,
    "PWKL": PolyPWKL,
    "GDI": PolyGDI,
}


def create_poly_selector(name: str, model: SequentialGDINA) -> PolySelectionCriterion:
    """按名称创建多级选题准则。

    Examples
    --------
    >>> sel = create_poly_selector("PWKL", model)   # doctest: +SKIP
    """
    key = str(name).strip().upper()
    if key not in POLY_SELECTION_REGISTRY:
        raise ValueError(
            f"未知多级选题准则 '{name}'，可选: {sorted(POLY_SELECTION_REGISTRY)}"
        )
    return POLY_SELECTION_REGISTRY[key](model)
