"""属性掌握模式估计 (能力估计)。

对 CD-CAT 而言，每作答一题后都要基于累积作答重新估计 α 的后验分布。
本模块提供三类**参数化**估计器：

- **EAP** 期望后验：返回后验概率加权的期望模式（各属性边际概率），稳健但偏低。
- **MAP** 最大后验：返回后验概率最大的模式，常用且准确。
- **MLE** 最大似然：仅用似然最大化，小样本时不稳定。

另提供**非参数**判别（:mod:`cdcat.estimation.nonparametric`）：
不估计任何题目参数，直接用 Q 矩阵的理想反应模式与作答向量的加权距离分类，
用于标定不可行时的兜底，以及"参数到底有没有带来增益"的基线对照。

同时提供属性边际掌握概率 ``marginal_probabilities``，用于终止判定与输出报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..models import CDModel
from ..qmatrix import QMatrix
from ..utils import pattern_to_str
from .nonparametric import (
    NONPARAMETRIC_METHODS,
    NonparametricResult,
    classify_nonparametric,
    distance_matrix,
    item_weights,
)

__all__ = [
    "AttributeEstimator",
    "EstimatorResult",
    "EstimationHistory",
    "NONPARAMETRIC_METHODS",
    "NonparametricResult",
    "classify_nonparametric",
    "distance_matrix",
    "item_weights",
]


@dataclass
class EstimatorResult:
    """单次估计结果。"""

    pattern: np.ndarray            # 估计的属性模式 (K,)
    pattern_idx: int               # 模式编号
    marginal: np.ndarray           # 各属性边际掌握概率 (K,)
    posterior: np.ndarray          # 完整后验分布 (2^K,)
    max_posterior: float           # 最大后验概率
    method: str                    # 'EAP' / 'MAP' / 'MLE'

    def __repr__(self) -> str:
        return (
            f"EstimatorResult(pattern={pattern_to_str(self.pattern)}, "
            f"max_posterior={self.max_posterior:.3f}, method={self.method})"
        )

    def marginal_str(self) -> str:
        """属性边际概率的可读字符串。"""
        return " ".join(f"{p:.3f}" for p in self.marginal)


@dataclass
class EstimationHistory:
    """估计轨迹记录（用于分析收敛过程）。"""

    step: List[int] = field(default_factory=list)
    item_idx: List[int] = field(default_factory=list)
    pattern_idx: List[int] = field(default_factory=list)
    max_posterior: List[float] = field(default_factory=list)
    marginal: List[np.ndarray] = field(default_factory=list)

    def append(self, step: int, item_idx: int, result: EstimatorResult) -> None:
        self.step.append(step)
        self.item_idx.append(item_idx)
        self.pattern_idx.append(result.pattern_idx)
        self.max_posterior.append(result.max_posterior)
        self.marginal.append(result.marginal.copy())

    def marginal_matrix(self) -> np.ndarray:
        """返回 (n_steps, K) 的边际概率矩阵。"""
        if not self.marginal:
            return np.zeros((0, 0))
        return np.vstack(self.marginal)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "item_idx": self.item_idx,
            "pattern_idx": self.pattern_idx,
            "max_posterior": self.max_posterior,
            "marginal": [m.tolist() for m in self.marginal],
        }


class AttributeEstimator:
    """属性掌握模式估计器。

    Parameters
    ----------
    model : CDModel
        响应模型，提供似然计算。
    prior : array-like, optional
        属性模式先验分布，长度 2^K；默认均匀先验。
    method : {'MAP', 'EAP', 'MLE'}
        默认估计方法。
    higher_order : bool, optional
        是否使用高阶先验（属性间独立且各属性掌握概率可指定）。
        若为 True，``attr_prior`` 给出各属性的边际掌握概率。
    hierarchy : AttributeHierarchy, optional
        属性层级结构。给定时后验被**约束在合法模式集合内**
        （非法模式的先验置 0），从而不会输出教学上不可能的知识状态。

    Examples
    --------
    >>> est = AttributeEstimator(model, method='MAP')
    >>> result = est.estimate(responses=[1,0,1], item_indices=[0,2,5])
    >>> result.pattern  # doctest: +SKIP
    array([1, 0, 1], dtype=int8)
    """

    def __init__(
        self,
        model: CDModel,
        prior: Optional[np.ndarray] = None,
        method: str = "MAP",
        higher_order: bool = False,
        attr_prior: Optional[np.ndarray] = None,
        hierarchy=None,
    ) -> None:
        self.model = model
        self.method = method.upper()
        if self.method not in {"MAP", "EAP", "MLE"}:
            raise ValueError("method 必须是 'MAP' / 'EAP' / 'MLE'")

        n_patterns = model.n_patterns
        self.higher_order = higher_order
        self.hierarchy = hierarchy
        if higher_order and attr_prior is not None:
            prob = np.clip(np.asarray(attr_prior, dtype=np.float64).ravel(), 1e-6, 1 - 1e-6)
            patterns = model.q.attribute_patterns().astype(np.float64)
            logp = patterns @ np.log(prob) + (1 - patterns) @ np.log(1 - prob)
            p = np.exp(logp - logp.max())
            self.prior = p / p.sum()
        elif prior is not None:
            p = np.asarray(prior, dtype=np.float64).ravel()
            if len(p) != n_patterns:
                raise ValueError(f"先验长度需为 {n_patterns}")
            self.prior = p / p.sum()
        else:
            self.prior = np.full(n_patterns, 1.0 / n_patterns)

        # 层级约束：把非法模式的先验清零并重新归一化
        self.legal_mask: Optional[np.ndarray] = None
        if hierarchy is not None and not hierarchy.is_empty:
            mask = np.asarray(hierarchy.legal_mask(), dtype=bool)
            if mask.shape[0] != n_patterns:
                raise ValueError(
                    f"层级结构的模式数 ({mask.shape[0]}) 与模型 ({n_patterns}) 不一致"
                )
            self.legal_mask = mask
            self.prior = np.where(mask, self.prior, 0.0)
            total = self.prior.sum()
            if total <= 0:
                raise ValueError("层级约束下先验全部为 0，请检查层级定义")
            self.prior = self.prior / total

        self.history = EstimationHistory()

    # ------------------------------------------------------------------ #
    # 后验计算
    # ------------------------------------------------------------------ #
    def posterior(
        self,
        responses: np.ndarray,
        item_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """返回归一化的属性模式后验分布，形状 ``(2^K,)``。"""
        if len(responses) == 0:
            return self.prior.copy()
        lik = self.model.likelihood(responses, item_indices)
        post = lik * self.prior
        total = post.sum()
        if total <= 0 or not np.isfinite(total):
            # 数值退化时退回先验
            return self.prior.copy()
        return post / total

    def marginal_probabilities(self, posterior: np.ndarray) -> np.ndarray:
        """由后验分布计算每个属性的边际掌握概率，形状 ``(K,)``。"""
        patterns = self.model.q.attribute_patterns().astype(np.float64)
        return posterior @ patterns

    # ------------------------------------------------------------------ #
    # 主估计入口
    # ------------------------------------------------------------------ #
    def estimate(
        self,
        responses: np.ndarray,
        item_indices: Optional[np.ndarray] = None,
        method: Optional[str] = None,
    ) -> EstimatorResult:
        """估计属性掌握模式。

        Parameters
        ----------
        responses : array-like, shape (n_answered,)
            累积的 0/1 作答。
        item_indices : array-like, optional
            每题对应的题目索引。
        method : str, optional
            覆盖默认方法。

        Returns
        -------
        EstimatorResult
        """
        m = (method or self.method).upper()
        responses = np.asarray(responses, dtype=np.float64).ravel()
        post = self.posterior(responses, item_indices)
        marginal = self.marginal_probabilities(post)

        if len(responses) == 0:
            # 无作答：返回各属性边际 > 0.5 的模式
            pattern = (marginal >= 0.5).astype(np.int8)
            idx = self._pattern_index(pattern)
            return EstimatorResult(pattern, idx, marginal, post, float(post.max()), m)

        if m == "MAP":
            idx = int(np.argmax(post))
            pattern = self.model.q.attribute_patterns()[idx].astype(np.int8)
        elif m == "EAP":
            pattern = (marginal >= 0.5).astype(np.int8)
            idx = self._pattern_index(pattern)
        else:  # MLE
            lik = self.model.likelihood(responses, item_indices)
            if self.legal_mask is not None:
                # 纯似然不含先验，需显式把非法模式排除
                lik = np.where(self.legal_mask, lik, -np.inf)
            idx = int(np.argmax(lik))
            pattern = self.model.q.attribute_patterns()[idx].astype(np.int8)

        return EstimatorResult(pattern, idx, marginal, post, float(post.max()), m)

    def _pattern_index(self, pattern: np.ndarray) -> int:
        """把属性模式转成模式编号。"""
        pattern = np.asarray(pattern).ravel()
        return int(sum(int(a) << (self.model.q.n_attrs - 1 - i) for i, a in enumerate(pattern)))

    # ------------------------------------------------------------------ #
    # 便捷方法
    # ------------------------------------------------------------------ #
    def reset_history(self) -> None:
        self.history = EstimationHistory()

    def estimate_sequential(
        self,
        responses: np.ndarray,
        item_indices: np.ndarray,
        record: bool = True,
    ) -> List[EstimatorResult]:
        """逐步估计：每作答一题后记录一次估计结果。

        用于可视化"估计如何随题目累进而收敛"。
        """
        results: List[EstimatorResult] = []
        for t in range(len(responses)):
            sub_r = responses[: t + 1]
            sub_i = item_indices[: t + 1]
            res = self.estimate(sub_r, sub_i)
            results.append(res)
            if record:
                self.history.append(t + 1, int(item_indices[t]), res)
        return results

    def attribute_accuracy(self, estimated: np.ndarray, truth: np.ndarray) -> float:
        """属性级判准率 (AMR)：估计模式与真实模式逐属性一致比例。"""
        estimated = np.asarray(estimated).ravel()
        truth = np.asarray(truth).ravel()
        return float(np.mean(estimated == truth))

    def pattern_correct(self, estimated: np.ndarray, truth: np.ndarray) -> bool:
        """模式级判准 (PMR)：整条模式是否完全正确。"""
        return bool(np.array_equal(np.asarray(estimated).ravel(), np.asarray(truth).ravel()))
