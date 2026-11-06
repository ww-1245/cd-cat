"""选题准则 (Item Selection Criteria)。

CD-CAT 的核心：根据当前属性后验分布，从剩余题库中挑选"信息量最大"的题目。

统一接口
--------
- ``compute(posterior, available_items, **ctx) -> ndarray``
  返回每个候选题目的准则值 (越大越优)。
- ``select(...) -> int``
  返回被选中题目的索引。

准则一览
--------
======== ==================================================================
KL       Kullback-Leibler 信息量（后验加权，Xu et al. 2003 基础形式）
PWKL     后验加权 KL (Cheng 2009)，对每个可能模式计算与当前估计的 KL
MPWKL    修正后验加权 KL (Kaplan et al. 2015)，对后验做扰动，抗局部最优
GDI      广义区分度指标 (de la Torre 2011)，直接基于题目概率差异
SHE      Shannon 熵缩减量，选择能最大程度降低后验熵的题目
CDI      认知诊断信息量，基于属性边际概率变化
MIV      最大理想反应方差，**只用 Q 矩阵、不需要题目参数**
NPSHE    理想反应下的期望熵缩减，SHE 的无参数版本
ERHD     期望汉明距离缩减，直接优化"属性误判个数"
AMI      属性边际熵缩减，绕开模式不可辨识、只盯属性
======== ==================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..models import CDModel
from .exposure import (
    ExposureControl,
    SympsonHetter,
    RandomesqueExposure,
    MaxExposureLimit,
    StratifiedExposure,
    NoExposureControl,
    create_exposure_control,
)

__all__ = [
    "SelectionCriterion",
    "KL",
    "PWKL",
    "MPWKL",
    "GDI",
    "SHE",
    "CDI",
    "MIV",
    "IdealResponseCriterion",
    "NPSHE",
    "ERHD",
    "AMI",
    "create_selector",
    "SELECTION_REGISTRY",
    "ExposureControl",
    "SympsonHetter",
    "RandomesqueExposure",
    "MaxExposureLimit",
    "StratifiedExposure",
    "NoExposureControl",
    "create_exposure_control",
    "AssemblyResult",
    "assemble_test_bank",
]


class SelectionCriterion(ABC):
    """选题准则基类。"""

    name: str = "base"

    def __init__(self, model: CDModel, eps: float = 1e-10) -> None:
        self.model = model
        self.eps = eps
        # 预取概率矩阵 (2^K, J)，所有准则共用
        self._P = np.clip(model.prob_matrix(), eps, 1.0 - eps)

    @property
    def n_patterns(self) -> int:
        return self._P.shape[0]

    @property
    def n_items(self) -> int:
        return self._P.shape[1]

    def clear_cache(self) -> None:
        """模型参数更新后需刷新缓存。"""
        self._P = np.clip(self.model.prob_matrix(), self.eps, 1.0 - self.eps)

    @abstractmethod
    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        """计算所有候选题目的准则值。

        Parameters
        ----------
        posterior : ndarray, shape (2^K,)
            当前属性模式后验分布。
        available_items : ndarray, optional
            可选题目的索引；默认全部未作答题。

        Returns
        -------
        ndarray
            与 ``available_items`` 等长的准则值数组（越大越好）。
        """

    def select(
        self,
        posterior: np.ndarray,
        available_items: Optional[np.ndarray] = None,
        exclude: Optional[np.ndarray] = None,
    ) -> int:
        """选出准则值最大的题目索引。

        Parameters
        ----------
        posterior : ndarray
            后验分布。
        available_items : ndarray, optional
            候选题目索引集合。
        exclude : ndarray, optional
            需排除的题目索引（如已作答）。

        Returns
        -------
        int
            题目索引。
        """
        cand = self._resolve_candidates(available_items, exclude)
        if len(cand) == 0:
            raise RuntimeError("没有可供选择的题目")
        values = self.compute(posterior, cand)
        return int(cand[int(np.argmax(values))])

    def _resolve_candidates(
        self,
        available_items: Optional[np.ndarray],
        exclude: Optional[np.ndarray],
    ) -> np.ndarray:
        if available_items is None:
            cand = np.arange(self.n_items)
        else:
            cand = np.asarray(available_items, dtype=np.int64).ravel()
        if exclude is not None and len(exclude) > 0:
            excl = set(int(x) for x in np.asarray(exclude).ravel())
            cand = np.array([c for c in cand if int(c) not in excl], dtype=np.int64)
        return cand

    def rank(
        self,
        posterior: np.ndarray,
        available_items: Optional[np.ndarray] = None,
        top_n: int = 10,
    ) -> list:
        """返回准则值最高的前 N 道题 ``[(item_idx, value), ...]``。"""
        cand = self._resolve_candidates(available_items, None)
        values = self.compute(posterior, cand)
        order = np.argsort(-values)[:top_n]
        return [(int(cand[i]), float(values[i])) for i in order]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_items={self.n_items}, n_patterns={self.n_patterns})"


# ====================================================================== #
# KL —— 后验加权 KL 信息量
# ====================================================================== #
class KL(SelectionCriterion):
    """KL 信息量准则（早期 CD-CAT 常用形式）。

    对候选题目 j，在"答对"和"答错"两种结果下分别计算后验 KL 散度，
    并按当前后验预测的作答概率加权：

        KL_j = Σ_α π(α) [ D_KL( P(α|X_j=1) || π(α) ) · P(X_j=1|α)
                        + D_KL( P(α|X_j=0) || π(α) ) · P(X_j=0|α) ]

    直观上：该题能把后验分布"推开"多少。
    """

    name = "KL"

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        P = self._P[:, cand]                          # (S, n_cand)
        post = posterior[:, None]                     # (S, 1)
        q_obs = post * P                              # P(α) * P(X=1|α)
        p_item = q_obs.sum(axis=0)                    # P(X_j=1)
        p_item = np.clip(p_item, self.eps, 1 - self.eps)

        # 两个后验：答复 1 和答复 0
        post1 = q_obs / p_item[None, :]
        post0 = (post * (1 - P)) / (1 - p_item)[None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            kl1 = np.where(post1 > 0, post1 * np.log(post1 / (posterior[:, None] + self.eps)), 0.0)
            kl0 = np.where(post0 > 0, post0 * np.log(post0 / (posterior[:, None] + self.eps)), 0.0)
        # 按模式加权求和
        w1 = (posterior[:, None] * P)
        w0 = (posterior[:, None] * (1 - P))
        # Σ_α π(α) P(X=1|α) D_KL(post1||π)  等
        term1 = (post * kl1).sum(axis=0)
        term0 = (post * kl0).sum(axis=0)
        return term1 + term0


# ====================================================================== #
# PWKL —— 后验加权 KL（当前主流）
# ====================================================================== #
class PWKL(SelectionCriterion):
    """后验加权 KL 信息量 (Cheng, 2009)。

        PWKL_j = Σ_α π(α) Σ_x P(X_j=x|α) log[ P(X_j=x|α) / P(X_j=x|α̂) ]

    其中 ``α̂`` 为当前估计的属性模式。相比 KL，PWKL 用"当前估计模式的
    作答分布"作为参照分布，计算更稳定，是 CD-CAT 中最常用的准则。
    """

    name = "PWKL"

    def __init__(self, model: CDModel, estimate: Optional[np.ndarray] = None, eps: float = 1e-10) -> None:
        super().__init__(model, eps)
        self._estimate_idx: Optional[int] = None
        if estimate is not None:
            self.set_estimate(estimate)

    def set_estimate(self, pattern_or_idx) -> "PWKL":
        """设置当前估计模式（可以是模式向量或模式编号）。"""
        if np.isscalar(pattern_or_idx):
            self._estimate_idx = int(pattern_or_idx)
        else:
            arr = np.asarray(pattern_or_idx).ravel()
            k = self.model.q.n_attrs
            self._estimate_idx = int(sum(int(a) << (k - 1 - i) for i, a in enumerate(arr)))
        return self

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        P = self._P[:, cand]                              # (S, C)
        post = posterior[:, None]                         # (S, 1)

        if self._estimate_idx is None:
            self._estimate_idx = int(np.argmax(posterior))
        p_hat = self._P[self._estimate_idx, cand][None, :]  # (1, C)

        with np.errstate(divide="ignore", invalid="ignore"):
            kl1 = np.where(P > 0, P * np.log(P / p_hat), 0.0)
            kl0 = np.where((1 - P) > 0, (1 - P) * np.log((1 - P) / (1 - p_hat)), 0.0)
        per_pattern = kl1 + kl0                            # (S, C)
        return (post * per_pattern).sum(axis=0)


# ====================================================================== #
# MPWKL —— 修正后验加权 KL
# ====================================================================== #
class MPWKL(SelectionCriterion):
    """修正后验加权 KL (Kaplan, de la Torre & Barrada, 2015)。

    为避免 PWKL 过早锁定到单一估计模式而陷入局部最优，MPWKL 对后验
    进行多次扰动，模拟"估计可能出错"的情形，再综合评分：

        MPWKL_j = (1/M) Σ_m Σ_α π_m(α) KL( P(X_j|α) || P(X_j|α̂_m) )

    其中 ``π_m`` 为第 m 个扰动后的后验（随机翻转部分属性的模式权重），
    ``α̂_m`` 为该扰动后验的众数。
    """

    name = "MPWKL"

    def __init__(
        self,
        model: CDModel,
        n_perturbations: int = 5,
        perturb_strength: float = 0.2,
        seed: int = 0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(model, eps)
        self.n_perturbations = n_perturbations
        self.perturb_strength = perturb_strength
        # 默认固定种子而非 None：default_rng(None) 会从操作系统熵源取种，
        # 同一份数据两次运行会给出不同的选题序列（实测 PMR 可差 0.5 个百分点），
        # 与 compare_criteria(seed=...) 承诺的可复现性冲突。需要随机化时显式传其他种子。
        self.rng = np.random.default_rng(seed)

    def _perturb(self, posterior: np.ndarray) -> tuple:
        """对后验做一次扰动，返回 (扰动后验, 众数索引)。"""
        k = self.model.q.n_attrs
        patterns = self.model.q.attribute_patterns()
        # 随机选择若干属性做"权重下沉"，模拟这些属性估计错误的可能
        n_flip = max(1, int(round(self.perturb_strength * k)))
        flip_attrs = self.rng.choice(k, size=n_flip, replace=False)

        pi = posterior.copy()
        for attr in flip_attrs:
            mismatch = patterns[:, attr] != patterns[int(np.argmax(posterior)), attr]
            pi[mismatch] *= 0.5          # 压低与当前众数不一致的模式
        total = pi.sum()
        pi = pi / total if total > 0 else posterior.copy()
        return pi, int(np.argmax(pi))

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        P = self._P[:, cand]
        acc = np.zeros(len(cand), dtype=np.float64)

        for _ in range(self.n_perturbations):
            pi, est_idx = self._perturb(posterior)
            p_hat = self._P[est_idx, cand][None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                kl1 = np.where(P > 0, P * np.log(P / p_hat), 0.0)
                kl0 = np.where((1 - P) > 0, (1 - P) * np.log((1 - P) / (1 - p_hat)), 0.0)
            acc += (pi[:, None] * (kl1 + kl0)).sum(axis=0)
        return acc / self.n_perturbations


# ====================================================================== #
# GDI —— 广义区分度指标
# ====================================================================== #
class GDI(SelectionCriterion):
    """广义区分度指标 (de la Torre, 2011; Kaplan et al. 2015 用于 CAT)。

        GDI_j = Σ_{α} Σ_{α'} π(α) π(α') [ P(X_j=1|α) - P(X_j=1|α') ]²

    完全基于题目在不同模式下的答对概率差异，与"当前估计"无关，
    因此计算较稳、不易受早期估计偏差影响，适合作为前几题的选题准则。
    """

    name = "GDI"

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        P = self._P[:, cand]                      # (S, C)
        # Σ_α π(α) P(α) 与 Σ_α π(α)² 的期望形式
        # 展开：E_α E_α'[(P(α)-P(α'))²] = 2(E[P²] - (E[P])²)
        ep = posterior @ P                        # (C,)
        ep2 = posterior @ (P ** 2)                # (C,)
        return 2.0 * (ep2 - ep ** 2)


# ====================================================================== #
# SHE —— Shannon 熵缩减
# ====================================================================== #
class SHE(SelectionCriterion):
    """Shannon 熵准则。

    选择能最大程度降低后验熵（期望信息增益最大）的题目：

        SHE_j = H(π) − Σ_x P(X_j=x) H( π(α|X_j=x) )

    其中 ``H`` 为 Shannon 熵。与 KL 精神一致但直接度量不确定性下降。
    """

    name = "SHE"

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        P = self._P[:, cand]
        post = posterior[:, None]

        def entropy(dist: np.ndarray, axis: int) -> np.ndarray:
            return -np.sum(np.where(dist > 0, dist * np.log(dist), 0.0), axis=axis)

        h_prior = entropy(posterior, axis=0)      # 标量
        q_obs = post * P
        p_item = np.clip(q_obs.sum(axis=0), self.eps, 1 - self.eps)
        post1 = q_obs / p_item[None, :]
        post0 = (post * (1 - P)) / (1 - p_item)[None, :]
        h1 = entropy(post1, axis=0)
        h0 = entropy(post0, axis=0)
        return h_prior - (p_item * h1 + (1 - p_item) * h0)


# ====================================================================== #
# CDI —— 认知诊断信息量（基于属性边际）
# ====================================================================== #
class CDI(SelectionCriterion):
    """属性边际不确定性下降量。

    对每个属性计算其边际掌握概率的方差 ``p(1-p)``（伯努利熵的近似），
    选择能使所有属性边际方差之和下降最多的题目：

        CDI_j = Σ_k [ Var_k(π) − E_x Var_k(π(α|X_j=x)) ]

    直观：兼顾所有属性，避免某些属性长期不确定。
    """

    name = "CDI"

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        P = self._P[:, cand]
        patterns = self.model.q.attribute_patterns().astype(np.float64)  # (S, K)
        post = posterior[:, None]

        def attr_var(dist: np.ndarray) -> np.ndarray:
            """返回 (K,) 各属性的边际方差（对模式分布求）。"""
            marg = dist.T @ patterns                     # (K,) 若 dist 是 (S,)
            return marg * (1 - marg)

        var_prior = attr_var(posterior).sum()

        q_obs = post * P
        p_item = np.clip(q_obs.sum(axis=0), self.eps, 1 - self.eps)
        post1 = q_obs / p_item[None, :]
        post0 = (post * (1 - P)) / (1 - p_item)[None, :]

        var1 = (post1.T @ patterns) * (1 - post1.T @ patterns)   # (C, K)
        var0 = (post0.T @ patterns) * (1 - post0.T @ patterns)
        exp_var = (p_item[:, None] * var1 + (1 - p_item)[:, None] * var0).sum(axis=1)
        return var_prior - exp_var


# ====================================================================== #
# MIV —— 最大理想反应方差（非参数选题）
# ====================================================================== #
class MIV(SelectionCriterion):
    """最大理想反应方差：唯一不需要题目参数的选题准则。

    对候选题 j 令 :math:`\\eta_j(\\alpha)` 为 Q 矩阵决定的理想作答，
    :math:`p_j = \\mathbb{E}[\\eta_j \\mid \\pi]` 为该题在后验下的期望得分，取

        MIV_j = p_j (1 - p_j)

    最大的题目。它等价于最大化理想反应在后验下的方差，也就是挑那道
    "当前最拿不准该不该答对"的题——信息量最大。因为 :math:`\\eta_j` 只由
    Q 矩阵决定，**不需要 slip/guess**，标定不可用时也能选题。

    与 KL/PWKL 的区别：KL 用概率差 :math:`P(1)-P(0)` 度量区分度，依赖题目参数；
    MIV 只看理想反应的二分性。当参数可信时 PWKL 通常更优，MIV 是它的
    无参数替代，也可作为"参数是否真的带来增益"的对照基准。

    实现上不继承 :class:`IdealResponseCriterion`：MIV 只要理想反应的期望，
    不需要劈开两支后验，用不上那套分支机制。
    """

    name = "MIV"

    def __init__(
        self,
        model: CDModel,
        ideal_mode: Optional[str] = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(model, eps)
        # 与 NPSHE / ERHD / AMI 保持一致：默认跟随模型的理想反应定义
        self.ideal_mode = ideal_mode or getattr(model, "ideal_mode", "conjunctive")
        # (S, J)：只由 Q 矩阵决定，与题目参数无关
        self._E = model.q.ideal_response(self.ideal_mode).astype(np.float64)

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        post = np.asarray(posterior, dtype=np.float64).ravel()
        total = post.sum()
        if total <= 0:
            post = np.full_like(post, 1.0 / post.size)
        else:
            post = post / total
        p = post @ self._E[:, cand]            # (n_cand,) 期望理想得分
        return p * (1.0 - p)


# ====================================================================== #
# 非参数准则的共同底座
# ====================================================================== #
def _entropy(p: np.ndarray) -> np.ndarray:
    """按列计算 Shannon 熵；0·log0 记为 0。"""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=0)


def _marginal_entropy(patterns: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """按列计算属性边际熵之和 ``Σ_k H(α_k)``。

    ``dist`` 形状 (S, C)，返回 (C,)。属性边际熵忽略属性间相关，
    因此能绕开"模式不可辨识"造成的 PMR 上限——这是它与模式级熵的区别。
    """
    marg = np.clip(patterns.T @ dist, 0.0, 1.0)          # (K, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(marg > 0, marg * np.log(marg), 0.0) + np.where(
            marg < 1, (1 - marg) * np.log(1 - marg), 0.0
        )
    return -terms.sum(axis=0)


class IdealResponseCriterion(SelectionCriterion):
    """只读理想反应的非参数准则底座。

    参数化准则（KL / PWKL / GDI / SHE / CDI）都要读 ``model.prob_matrix()``，
    也就意味着必须先有 slip / guess。这一族改成只读
    ``q.ideal_response()``：把"该答对 / 该答错"当成确定性观察，再问
    "看到这个观察之后后验会怎么变"。

    由此得到的两个性质，正是它们的价值所在：

    1. **不需要标定**——标定不可行（新题、小样本、参数未收敛）时仍能选题；
    2. **可作为对照组**——参数化准则相对它们多出来的精度，就是标定换来的收益。

    代价是丢掉了"答对概率有多确定"这一层信息：理想反应把所有题目都
    当成 slip=guess=0，因此区分不出"这题大概率会答对"和"这题一定答对"。
    """

    def __init__(
        self,
        model: CDModel,
        ideal_mode: Optional[str] = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(model, eps)
        # 默认跟随模型的理想反应定义：DINO 是析取，其余是合取
        self.ideal_mode = ideal_mode or getattr(model, "ideal_mode", "conjunctive")
        self._E = model.q.ideal_response(self.ideal_mode).astype(np.float64)   # (S, J)
        self._patterns = model.q.attribute_patterns().astype(np.float64)       # (S, K)

    def _branches(
        self, posterior: np.ndarray, cand: np.ndarray
    ) -> tuple:
        """把后验按每个候选题的两种理想作答劈开。

        Returns
        -------
        post : ndarray (S,)
            归一化后的当前后验。
        w1, w0 : ndarray (C,)
            该题答对 / 答错这一支的总概率（= 该分支覆盖的后验质量）。
        post1, post0 : ndarray (S, C)
            条件于该分支的后验；不可达的分支整列为 0。
        """
        post = np.asarray(posterior, dtype=np.float64).ravel()
        total = post.sum()
        post = post / total if total > 0 else np.full_like(post, 1.0 / post.size)

        hit = self._E[:, cand] > 0.5                   # (S, C)
        un1 = post[:, None] * hit
        un0 = post[:, None] * ~hit
        w1 = un1.sum(axis=0)
        w0 = un0.sum(axis=0)
        post1 = np.divide(un1, w1, out=np.zeros_like(un1), where=w1 > 0)
        post0 = np.divide(un0, w0, out=np.zeros_like(un0), where=w0 > 0)
        return post, w1, w0, post1, post0


# ====================================================================== #
# NPSHE —— 理想反应下的期望熵缩减
# ====================================================================== #
class NPSHE(IdealResponseCriterion):
    """理想反应下的 Shannon 熵缩减（非参数 SHE）。

        NPSHE_j = H(π) − [ w1·H(π | X_j=1) + w0·H(π | X_j=0) ]

    与 ``SHE`` 的唯一区别是两支后验怎么算：SHE 用 ``P(X=1|α)`` 加权，
    NPSHE 直接用理想反应切成两半，因此**不需要 slip / guess**。

    附带的差别值得注意：NPSHE 恒把"答对"和"答错"当成同等确定的信息，
    所以它偏爱"能把模式集对半劈开"的题；SHE 则还会考虑"这题答对本身
    有多大概率"，因而会避开那些几乎必然答对的题。
    """

    name = "NPSHE"

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        post, w1, w0, post1, post0 = self._branches(posterior, cand)
        h_now = float(_entropy(post))
        return h_now - (w1 * _entropy(post1) + w0 * _entropy(post0))


# ====================================================================== #
# ERHD —— 期望汉明距离缩减
# ====================================================================== #
class ERHD(IdealResponseCriterion):
    """期望汉明距离缩减 (Expected Reduction in Hamming Distance)。

        ERHD_j = EHD(π) − [ w1·EHD(π | X_j=1) + w0·EHD(π | X_j=0) ]
        EHD(π) = Σ_α π(α) · d_H(α, α̂),  α̂ = argmax π

    直接把"还差几个属性判错"当作目标函数：EHD 就是当前后验下的期望
    属性误判个数，ERHD 是做完这道题后它的期望下降量。与 AMR 的口径
    完全一致，所以它选出的题最"对得起"最终指标。

    与 NPSHE 的区别在尺度：熵对"把 1 个属性搞错"和"把 3 个属性搞错"
    一视同仁（只要分布形状相同），汉明距离则按错的个数线性计费。
    """

    name = "ERHD"

    def __init__(
        self,
        model: CDModel,
        ideal_mode: Optional[str] = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(model, ideal_mode, eps)
        # (S, S) 两两汉明距离；K=5 时只有 32×32，成本可忽略
        diff = self._patterns[:, None, :] != self._patterns[None, :, :]
        self._hamming = diff.sum(axis=-1).astype(np.float64)

    def _ehd(self, dist: np.ndarray) -> np.ndarray:
        """按列计算期望汉明距离（到各自众数）。``dist`` 形状 (S, C)。"""
        modes = np.argmax(dist, axis=0)                    # (C,)
        return (dist * self._hamming[:, modes]).sum(axis=0)

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        post, w1, w0, post1, post0 = self._branches(posterior, cand)
        cur = float(post @ self._hamming[:, int(np.argmax(post))])
        return cur - (w1 * self._ehd(post1) + w0 * self._ehd(post0))


# ====================================================================== #
# AMI —— 属性边际熵缩减
# ====================================================================== #
class AMI(IdealResponseCriterion):
    """属性边际熵缩减 (Attribute Marginal entropy reduction)。

        AMI_j = Σ_k H(α_k) − [ w1·Σ_k H(α_k | X_j=1) + w0·Σ_k H(α_k | X_j=0) ]

    只看每个属性的掌握/未掌握有多确定，**不看属性之间的相关**。
    这一点在诊断测验里很关键：当 Q 矩阵存在等价类时（两个模式对所有题
    理想反应完全相同），模式级准则会一直"追"那些区分不开的题，而 AMI
    从一开始就只盯属性边际，因而不会把题量浪费在不可辨识的方向上。

    代价是它可能忽略"属性组合"的信息——两个模式属性边际相同但联合不同，
    AMI 视为等价。实践上常与模式级准则（NPSHE / ERHD）搭配使用。
    """

    name = "AMI"

    def compute(self, posterior: np.ndarray, available_items: Optional[np.ndarray] = None) -> np.ndarray:
        cand = self._resolve_candidates(available_items, None)
        post, w1, w0, post1, post0 = self._branches(posterior, cand)
        cur = float(_marginal_entropy(self._patterns, post[:, None])[0])
        return cur - (
            w1 * _marginal_entropy(self._patterns, post1)
            + w0 * _marginal_entropy(self._patterns, post0)
        )


# ====================================================================== #
# 注册表与工厂
# ====================================================================== #
SELECTION_REGISTRY = {
    "KL": KL,
    "PWKL": PWKL,
    "MPWKL": MPWKL,
    "GDI": GDI,
    "SHE": SHE,
    "CDI": CDI,
    "MIV": MIV,
    "NPSHE": NPSHE,
    "ERHD": ERHD,
    "AMI": AMI,
}


def create_selector(name: str, model: CDModel, **kwargs) -> SelectionCriterion:
    """按名称创建选题准则。

    Examples
    --------
    >>> sel = create_selector("PWKL", model)   # doctest: +SKIP
    >>> sel = create_selector("MPWKL", model, n_perturbations=10, seed=1)  # doctest: +SKIP
    """
    key = name.strip().upper()
    if key not in SELECTION_REGISTRY:
        raise ValueError(f"未知选题准则 '{name}'，可选: {sorted(SELECTION_REGISTRY)}")
    return SELECTION_REGISTRY[key](model, **kwargs)


# 题库装配（放在最后，避免与上面的准则导入顺序纠缠）
from .assembly import AssemblyResult, assemble_test_bank  # noqa: E402
