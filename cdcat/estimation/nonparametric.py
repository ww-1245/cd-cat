"""非参数认知诊断分类：不估计任何题目参数的知识状态判别。

为什么需要
----------
参数化 CDM（DINA / DINO / G-DINA / RRUM）必须先标定 slip / guess，再据此算后验。
标定本身会带来误差：样本小、题目新、Q 矩阵有误时，参数估计不稳，分类精度随之下降。
距离判别类方法**只用 Q 矩阵决定的理想反应模式**与作答向量比对，完全不碰题目参数：

- 标定不可行时（样本太少、题目刚上线）仍能立刻出结果；
- 作为参数模型的"干净基线"——若参数模型的 PMR 追不平非参数方法，
  说明标定没有带来实际收益，问题多半在 Q 矩阵或模型设定上。

方法
----
``hamming``
    等权汉明距离。对模式 α，:math:`d(\\alpha)=\\sum_j |X_j-\\eta_j(\\alpha)|`，
    取距离最小的 α。简单、无需任何统计量，但对区分度弱的题目一视同仁。

``gdd``
    广义距离判别（Leighton, Gierl & Hunka, 2004）。给每题加权
    :math:`w_j = 1/[p_j(1-p_j)]`，p_j 为样本通过率：越靠近 0.5 的题方差越大、
    权重越高，极端难/易题自动降权。注意 p_j 是**样本统计量**，不是模型参数，
    因此仍属非参数范畴。权重会归一化到均值 1，使距离的单位是"等效题数"。

属性层级
--------
给定 :class:`~cdcat.qmatrix.AttributeHierarchy` 时，非法模式的距离置为 +∞，
等价于 AHM（属性层级方法）里"只在合法模式集中搜索"的做法。
这是非参数方法相对参数模型的一个天然优势：约束直接加在候选集上，不需要改似然。

距离 → 伪后验
-------------
报告与绘图需要一个 0~1 的连续刻度，这里用
:math:`\\tilde\\pi(\\alpha) \\propto \\exp(-d(\\alpha)/T)`。
权重已归一化，故 T=1 表示"距离每多 1 个等效题，相对权重衰减 e 倍"。
**它不是真实后验**：没有似然、没有先验，只是距离的柔性排序。
所有对外文案都应写成"伪后验/置信度"，不要与 MAP 后验混为一谈。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..qmatrix import QMatrix

__all__ = [
    "NONPARAMETRIC_METHODS",
    "NonparametricResult",
    "item_weights",
    "distance_matrix",
    "classify_nonparametric",
]

#: 支持的非参数判别方法
NONPARAMETRIC_METHODS: Dict[str, str] = {
    "hamming": "等权汉明距离",
    "gdd": "广义距离判别（方差加权）",
}


# ====================================================================== #
# 权重与距离
# ====================================================================== #
def item_weights(
    responses: np.ndarray,
    method: str = "gdd",
) -> np.ndarray:
    """计算题目权重，形状 ``(J,)``。

    Parameters
    ----------
    responses : ndarray, shape (n_examinees, n_items)
        作答矩阵，允许 ``np.nan`` 表示未作答。
    method : {'hamming', 'gdd'}
        ``hamming`` 返回全 1；``gdd`` 返回 :math:`1/[p(1-p)]` 并归一化到均值 1。

    Notes
    -----
    通过率做了拉普拉斯平滑（(正确数+0.5)/(作答数+1)）并截断到 [0.02, 0.98]，
    否则满分/零分题会让权重发散，把其它题的信息完全淹没。
    """
    key = str(method).strip().lower()
    if key not in NONPARAMETRIC_METHODS:
        raise ValueError(
            f"未知非参数方法 '{method}'，可选: {sorted(NONPARAMETRIC_METHODS)}"
        )
    x = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")

    if key == "hamming":
        return np.ones(x.shape[1], dtype=np.float64)

    mask = np.isfinite(x)
    n_obs = mask.sum(axis=0).astype(np.float64)
    correct = np.where(mask, x, 0.0).sum(axis=0)
    p = (correct + 0.5) / (n_obs + 1.0)
    p = np.clip(p, 0.02, 0.98)
    w = 1.0 / (p * (1.0 - p))
    return w / float(np.mean(w))


def distance_matrix(
    q: QMatrix,
    responses: np.ndarray,
    method: str = "gdd",
    weights: Optional[np.ndarray] = None,
    hierarchy=None,
    ideal_mode: str = "conjunctive",
) -> np.ndarray:
    """作答向量到每个理想反应模式的加权距离，形状 ``(n_examinees, 2^K)``。

    用两次矩阵乘法算完，不展开 ``(n, 2^K, J)`` 的三维数组——
    K=6、J=120、n=5000 时三维数组要 3.8 GB，而这里只有 ``(n, 2^K)``。

    Parameters
    ----------
    q : QMatrix
    responses : ndarray, shape (n_examinees, n_items)
        允许 ``np.nan``；该题不计入距离（等价于"未作答不提供信息"）。
    method : {'hamming', 'gdd'}
        仅在 ``weights`` 为 None 时用于生成权重。
    weights : ndarray, optional
        自定义题目权重，长度 J。
    hierarchy : AttributeHierarchy, optional
        给定时非法模式的距离置 +∞。
    ideal_mode : {'conjunctive', 'disjunctive'}
        理想反应的定义方式。DINA 型用 conjunctive，DINO 型用 disjunctive。
    """
    q = q if isinstance(q, QMatrix) else QMatrix(q)
    x = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    if x.shape[1] != q.n_items:
        raise ValueError(
            f"作答矩阵列数 {x.shape[1]} 与题目数 {q.n_items} 不符"
        )

    mask = np.isfinite(x)
    xf = np.where(mask, x, 0.0)             # 缺失按 0 填，靠 mask 项抵消
    m = mask.astype(np.float64)

    if weights is None:
        w = item_weights(x, method)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != q.n_items:
            raise ValueError(f"权重长度需为 {q.n_items}")
        if np.any(w < 0):
            raise ValueError("权重不能为负")

    e = q.ideal_response(ideal_mode).astype(np.float64)     # (S, J)
    ew = e * w                                              # (S, J)

    # |X - E| = X(1-E) + (1-X)E，对 J 求和后等于下面两个内积之和
    d = xf @ (w - ew).T + (m - xf) @ ew.T                   # (n, S)

    if hierarchy is not None and not hierarchy.is_empty:
        legal = np.asarray(hierarchy.legal_mask(), dtype=bool)
        if legal.shape[0] != d.shape[1]:
            raise ValueError(
                f"层级结构的模式数 ({legal.shape[0]}) 与 Q 矩阵 ({d.shape[1]}) 不一致"
            )
        d = np.where(legal[None, :], d, np.inf)

    return d


# ====================================================================== #
# 结果对象
# ====================================================================== #
@dataclass
class NonparametricResult:
    """非参数分类结果。

    Attributes
    ----------
    method : str
        方法名（``hamming`` / ``gdd``）。
    distances : ndarray, shape (n_examinees, 2^K)
        加权距离矩阵（非法模式为 inf）。
    patterns : ndarray, shape (n_examinees, K)
        判别出的属性模式（距离最小者）。
    pattern_indices : ndarray, shape (n_examinees,)
        模式在 ``q.attribute_patterns()`` 中的行索引。
    posterior : ndarray, shape (n_examinees, 2^K)
        **伪后验** ``exp(-d/T)`` 归一化结果，仅供绘图与置信度使用。
    marginal : ndarray, shape (n_examinees, K)
        伪后验下各属性的边际掌握概率。
    confidence : ndarray, shape (n_examinees,)
        伪后验在所选模式上的取值，可理解为"判别把握"。
    weights : ndarray, shape (n_items,)
        实际使用的题目权重。
    attr_names : list of str
    temperature : float
        伪后验的温度 T。
    hierarchy : str
        层级约束的说明（空串表示无约束）。
    pmr, amr : float, optional
        仅在提供真实模式时填充。
    """

    method: str
    distances: np.ndarray
    patterns: np.ndarray
    pattern_indices: np.ndarray
    posterior: np.ndarray
    marginal: np.ndarray
    confidence: np.ndarray
    weights: np.ndarray
    attr_names: List[str] = field(default_factory=list)
    temperature: float = 1.0
    hierarchy: str = ""
    pmr: Optional[float] = None
    amr: Optional[float] = None

    # ------------------------------------------------------------------ #
    @property
    def n_examinees(self) -> int:
        return self.patterns.shape[0]

    @property
    def n_attrs(self) -> int:
        return self.patterns.shape[1]

    def mastery_proportion(self) -> np.ndarray:
        """各属性被判为掌握的比例。"""
        return self.patterns.mean(axis=0)

    def mean_confidence(self) -> float:
        return float(np.mean(self.confidence))

    def distance_gap(self) -> np.ndarray:
        """每个被试"最优 vs 次优"的距离差，可作为判别稳定性指标。

        差值接近 0 说明两个模式几乎并列，该被试的分类结果不可信。
        """
        d = np.where(np.isfinite(self.distances), self.distances, np.inf)
        if d.shape[1] < 2:
            return np.full(d.shape[0], np.inf)
        part = np.partition(d, 1, axis=1)
        return part[:, 1] - part[:, 0]

    def accuracy(self, truth: np.ndarray) -> Dict[str, object]:
        """与真实模式比对，返回 PMR / AMR / 分属性判准率。"""
        truth = np.asarray(truth)
        if truth.shape != self.patterns.shape:
            raise ValueError(
                f"真实模式形状 {truth.shape} 与判别结果 {self.patterns.shape} 不符"
            )
        correct = self.patterns == truth
        per_attr = correct.mean(axis=0)
        return {
            "pmr": float(correct.all(axis=1).mean()),
            "amr": float(correct.mean()),
            "per_attr": [float(v) for v in per_attr],
        }

    def to_pattern_diagnosis(self):
        """转换成 :class:`~cdcat.diagnostics.PatternDiagnosis`。

        这样非参数结果可以直接复用后验热图、掌握率柱图、群体报告等现有组件。
        注意其中的 ``posterior`` 是伪后验，画图时标题应注明。
        """
        from ..diagnostics import PatternDiagnosis

        return PatternDiagnosis(
            posterior=self.posterior,
            map_patterns=self.patterns.astype(np.int8),
            eap_marginal=self.marginal,
            eap_patterns=(self.marginal >= 0.5).astype(np.int8),
            map_confidence=self.confidence,
            attr_reliability=2.0 * np.abs(self.marginal - 0.5),
            pattern_indices=self.pattern_indices,
            attr_names=list(self.attr_names),
        )

    def summary(self) -> str:
        label = NONPARAMETRIC_METHODS.get(self.method, self.method)
        lines = [
            f"非参数分类（{self.method} — {label}）",
            f"  被试数 {self.n_examinees}  属性数 {self.n_attrs}  "
            f"温度 T={self.temperature:g}",
            f"  权重范围 {self.weights.min():.3f} ~ {self.weights.max():.3f}"
            f"（均值 {self.weights.mean():.3f}）",
            f"  平均判别把握 {self.mean_confidence():.4f}",
            f"  平均距离差 {np.nanmean(self.distance_gap()):.4f}",
        ]
        if self.hierarchy:
            lines.append(f"  层级约束：{self.hierarchy}")
        if self.pmr is not None:
            lines.append(f"  PMR={self.pmr:.4f}  AMR={self.amr:.4f}")
        mastered = " ".join(
            f"{n}={v:.2f}" for n, v in zip(self.attr_names, self.mastery_proportion())
        )
        if mastered:
            lines.append(f"  掌握比例：{mastered}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"NonparametricResult(method={self.method!r}, "
            f"n_examinees={self.n_examinees}, n_attrs={self.n_attrs})"
        )


# ====================================================================== #
# 主入口
# ====================================================================== #
def classify_nonparametric(
    q: QMatrix,
    responses: np.ndarray,
    method: str = "gdd",
    hierarchy=None,
    temperature: float = 1.0,
    weights: Optional[np.ndarray] = None,
    ideal_mode: str = "conjunctive",
    attr_names: Optional[Sequence[str]] = None,
    prior: Optional[np.ndarray] = None,
) -> NonparametricResult:
    """非参数分类主入口。

    Parameters
    ----------
    q : QMatrix
    responses : ndarray, shape (n_examinees, n_items)
    method : {'hamming', 'gdd'}
    hierarchy : AttributeHierarchy, optional
        层级约束；非法模式不参与判别。
    temperature : float
        伪后验温度，默认 1.0（距离单位=等效题数）。
    weights : ndarray, optional
        自定义题目权重；给定时 ``method`` 仅用于命名。
    ideal_mode : {'conjunctive', 'disjunctive'}
    attr_names : sequence of str, optional
    prior : ndarray, optional
        长度 2^K 的先验，**仅用于距离并列时的打破**（加一个极小的
        ``-1e-9·log π(α)``），不改变距离本身，因此不影响伪后验。

    Returns
    -------
    NonparametricResult
    """
    q = q if isinstance(q, QMatrix) else QMatrix(q)
    key = str(method).strip().lower()
    if key not in NONPARAMETRIC_METHODS:
        raise ValueError(
            f"未知非参数方法 '{method}'，可选: {sorted(NONPARAMETRIC_METHODS)}"
        )
    if temperature <= 0:
        raise ValueError("temperature 必须为正")

    x = np.asarray(responses, dtype=np.float64)
    d = distance_matrix(
        q, x, method=key, weights=weights, hierarchy=hierarchy, ideal_mode=ideal_mode
    )

    # 并列时的打破：距离完全相同很常见（全 0 作答、题目不足时），
    # 用先验做微小偏移，量级 1e-9 远小于任何真实距离差，不改变判别结论。
    tie = np.zeros(d.shape[1], dtype=np.float64)
    if prior is not None:
        p = np.asarray(prior, dtype=np.float64).ravel()
        if p.shape[0] != d.shape[1]:
            raise ValueError(f"先验长度需为 {d.shape[1]}")
        p = np.clip(p, 1e-300, None)
        tie = -1e-9 * np.log(p / p.sum())
    idx = np.argmin(d + tie[None, :], axis=1)

    patterns = q.attribute_patterns()
    chosen = patterns[idx].astype(np.int8)

    # 伪后验：exp(-d/T)，非法模式（inf）自然得 0。
    # 注意是减去**每行最大值**（= 负的最小距离），这样最小距离项恰好为 1。
    # 若误写成减去最小值，全部值会被下面一行 clip 成 0，后验退化为均匀分布。
    z = -(d / float(temperature))
    z = z - np.max(z, axis=1, keepdims=True)
    np.clip(z, -700.0, 0.0, out=z)          # 防止 exp 下溢报警
    post = np.exp(z)
    denom = post.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    post = post / denom

    ones = patterns.astype(np.float64)
    marginal = post @ ones
    confidence = post[np.arange(post.shape[0]), idx]

    if weights is None:
        w_used = item_weights(x, key)
    else:
        w_used = np.asarray(weights, dtype=np.float64).ravel()

    names = list(attr_names) if attr_names is not None else list(q.attr_names or [])
    if not names:
        names = [f"A{i + 1}" for i in range(q.n_attrs)]

    h_desc = ""
    if hierarchy is not None and not hierarchy.is_empty:
        h_desc = f"{hierarchy.structure}（{hierarchy.n_legal_patterns}/{q.n_patterns} 个合法模式）"

    return NonparametricResult(
        method=key,
        distances=d,
        patterns=chosen,
        pattern_indices=idx.astype(np.int64),
        posterior=post,
        marginal=marginal,
        confidence=confidence,
        weights=w_used,
        attr_names=names,
        temperature=float(temperature),
        hierarchy=h_desc,
    )
