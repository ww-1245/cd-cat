"""认知诊断模型的拟合指标、被试诊断与模型比较。

本模块提供三类能力：

1. **模型拟合指标**（:class:`FitIndices`）
   - 观测 ``-2LL``、参数个数 ``n_params``、``AIC`` / ``BIC`` / ``AICc``
   - 饱和模型 ``-2LL``（各反应向量自由估计）与相对拟合指数
   - 属性掌握比例合理性检查

2. **被试诊断**（:class:`PatternDiagnosis`）
   - MAP / EAP 两种知识状态估计
   - 属性边际掌握概率、模式后验分布、属性级分类信度
   - 低信度属性告警

3. **模型比较**（:func:`compare_models`、:func:`likelihood_ratio_test`）
   - 多模型 AIC/BIC 择优表
   - 嵌套模型的似然比检验

4. **绝对拟合与局部依赖**（:mod:`cdcat.diagnostics.absolute_fit`）
   - 题目级 Pearson ``χ²`` / 似然比 ``G²`` 检验与效应量
   - Yen Q3 残差相关，检出局部依赖（Q 矩阵漏标属性的信号）
   - :func:`analyse_absolute_fit` 一次跑完

约定
----
所有诊断的**模式索引**均遵循 :meth:`QMatrix.attribute_patterns` 的行序，
即行 ``i`` 对应 ``i`` 的二进制展开（高位在左）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..models import CDModel, GDINA, RRUM
from ..qmatrix import QMatrix
from .absolute_fit import (
    AbsoluteFitResult,
    ItemFitResult,
    LocalDependenceResult,
    analyse_absolute_fit,
    item_fit,
    local_dependence,
)

__all__ = [
    "FitIndices",
    "compute_fit_indices",
    "saturated_log_likelihood",
    "PatternDiagnosis",
    "diagnose_patterns",
    "compare_models",
    "likelihood_ratio_test",
    "parameter_recovery",
    "ItemFitResult",
    "item_fit",
    "LocalDependenceResult",
    "local_dependence",
    "AbsoluteFitResult",
    "analyse_absolute_fit",
]


def __getattr__(name: str):
    """按需加载较重的子模块，避免包初始化时拉入全部诊断代码。"""
    if name in ("DIFItem", "DIFReport", "dif_analysis"):
        from . import dif as _dif

        return getattr(_dif, name)
    if name in ("PPMCResult", "posterior_predictive_check"):
        from . import ppmc as _ppmc

        return getattr(_ppmc, name)
    if name in ("HierarchyCandidate", "HierarchyComparisonReport",
                "compare_hierarchies", "DEFAULT_STRUCTURES"):
        from . import hierarchy_test as _ht

        return getattr(_ht, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ += [
    "DIFItem",
    "DIFReport",
    "dif_analysis",
    "PPMCResult",
    "posterior_predictive_check",
    "HierarchyCandidate",
    "HierarchyComparisonReport",
    "compare_hierarchies",
    "DEFAULT_STRUCTURES",
]


# ====================================================================== #
# 工具
# ====================================================================== #
def _count_free_params(model: CDModel, fix_guess: bool = False) -> int:
    """统计模型的自由参数个数。

    - DINA / DINO：每题 2 个（slip, guess）
    - RRUM：每题 1 + r_j 个（pi_star 与各属性惩罚）
    - G-DINA：每题 ``2^{r_j}`` 个（饱和潜在组概率，logit 基下即 lambda 个数）
    """
    j = model.n_items
    if isinstance(model, GDINA):
        total = 0
        for i in range(j):
            r = int(model.q.matrix[i].sum())
            total += 1 << r          # baseline + 各交互项 = 2^r
        return total
    if isinstance(model, RRUM):
        total = 0
        for i in range(j):
            total += 1 + max(int(model.q.matrix[i].sum()), 1)
        return total
    # DINA / DINO
    return 2 * j if not fix_guess else j


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-10, 1 - 1e-10)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ====================================================================== #
# 拟合指标
# ====================================================================== #
@dataclass
class FitIndices:
    """模型拟合指标集合。

    Attributes
    ----------
    log_likelihood : float
        观测数据的边际对数似然。
    n_params : int
        自由参数个数。
    n_observations : int
        观测数（被试数 × 题目数），用于 BIC 的样本量。
    n_examinees, n_items : int
    aic, bic, aicc : float
        信息准则，越小越好。``AICc`` 为小样本修正版。
    deviance : float
        偏差 ``-2·logL``。
    saturated_ll : float
        饱和模型的对数似然（各反应向量自由估计）。
    saturated_deviance : float
        饱和模型偏差。
    cfi : float
        比较拟合指数 (0~1，越接近 1 越好)。基于饱和与独立模型。
    tli : float
        Tucker-Lewis 指数。
    rmsea : float
        近似误差均方根 (越接近 0 越好)，可按需解读。
    """

    log_likelihood: float
    n_params: int
    n_observations: int
    n_examinees: int
    n_items: int
    aic: float
    bic: float
    aicc: float
    deviance: float
    saturated_ll: float
    saturated_deviance: float
    cfi: float
    tli: float
    rmsea: float
    model_name: str = ""
    pmr: Optional[float] = None
    amr: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "model": self.model_name,
            "log_likelihood": self.log_likelihood,
            "n_params": self.n_params,
            "n_observations": self.n_observations,
            "aic": self.aic,
            "bic": self.bic,
            "aicc": self.aicc,
            "deviance": self.deviance,
            "saturated_ll": self.saturated_ll,
            "saturated_deviance": self.saturated_deviance,
            "cfi": self.cfi,
            "tli": self.tli,
            "rmsea": self.rmsea,
        }
        if self.pmr is not None:
            d["pmr"] = self.pmr
        if self.amr is not None:
            d["amr"] = self.amr
        return d

    def summary(self) -> str:
        lines = [
            f"拟合指标 · {self.model_name or '模型'}",
            "-" * 52,
            f"  边际对数似然  : {self.log_likelihood:.4f}",
            f"  偏差 (-2LL)   : {self.deviance:.4f}",
            f"  自由参数个数  : {self.n_params}",
            f"  被试数 / 题数 : {self.n_examinees} / {self.n_items}",
            f"  AIC           : {self.aic:.4f}",
            f"  BIC           : {self.bic:.4f}",
            f"  AICc          : {self.aicc:.4f}",
            f"  饱和模型 -2LL : {self.saturated_deviance:.4f}",
            f"  CFI           : {self.cfi:.4f}",
            f"  TLI           : {self.tli:.4f}",
        ]
        return "\n".join(lines)


def saturated_log_likelihood(
    responses: np.ndarray, min_cell_count: float = 1.0
) -> float:
    """饱和模型的对数似然。

    饱和模型对**每个不同的反应向量**自由估计其概率，其对数似然为：

        logL_sat = Σ_v n_v · log(n_v / N)

    其中 ``n_v`` 为反应向量 ``v`` 的出现次数。当被试数远大于可能的回答
    向量数时，饱和模型会过拟合；此处对出现次数低于 ``min_cell_count``
    的单元格从计数中剔除其"孤立贡献"以避免 ``log(0)`` 与极端值。

    缺失值按 0 处理：饱和模型需要一个完整的反应向量才能定义"某个模式出现了
    几次"，缺失位无法参与计数。这会让饱和 -2LL 略偏低，因此 CFI/TLI 在缺失
    较多时只能作粗略参考。
    """
    responses = np.asarray(responses, dtype=np.float64)
    responses = np.nan_to_num(responses, nan=0.0).astype(np.int8)
    n, j = responses.shape
    # 把每行编码成整数以统计频次
    packed = np.zeros(n, dtype=np.int64)
    for c in range(j):
        packed = (packed << 1) | responses[:, c].astype(np.int64)
    _, counts = np.unique(packed, return_counts=True)
    counts = counts[counts >= min_cell_count]
    total = counts.sum()
    return float(np.sum(counts * np.log(counts / total)))


def _independence_log_likelihood(responses: np.ndarray) -> float:
    """独立（零模型）对数似然：假设题目间相互独立，忽略属性结构。

    每个题目只有一个答对率参数 ``p_j``，用于作为 CFI/TLI 的基线模型。

    缺失位不参与计算：``p_j`` 用观测均值，似然只累加已作答的单元。否则
    ``mean`` 遇到 NaN 会把整条基线似然变成 NaN，CFI/TLI 跟着一起报废。
    """
    responses = np.asarray(responses, dtype=np.float64)
    n, j = responses.shape
    obs = np.isfinite(responses)
    filled = np.where(obs, responses, 0.0)
    counts = np.maximum(obs.sum(axis=0), 1)
    pj = np.clip(filled.sum(axis=0) / counts, 1e-10, 1 - 1e-10)
    ll = 0.0
    for c in range(j):
        xc = filled[obs[:, c], c]
        if xc.size == 0:
            continue
        p = float(pj[c])
        ll += float((xc * np.log(p) + (1 - xc) * np.log(1 - p)).sum())
    return ll


def compute_fit_indices(
    responses: np.ndarray,
    log_likelihood: float,
    model: CDModel,
    model_name: str = "",
    n_extra_params: int = 0,
) -> FitIndices:
    """计算完整的模型拟合指标。

    Parameters
    ----------
    responses : ndarray, shape (n_examinees, n_items)
        0/1 作答矩阵。
    log_likelihood : float
        该模型在数据上的边际对数似然。
    model : CDModel
        已标定好的模型实例（用于统计参数个数）。
    model_name : str
        模型名，用于报告。
    n_extra_params : int
        题目参数之外的额外自由参数个数。例如在 EM 中一并估计了属性
        先验 π(α) 时传入 ``2^K - 1``，使这里的 AIC/BIC 与
        :class:`~cdcat.calibration.CalibrationResult` 保持一致。

    Returns
    -------
    FitIndices
    """
    responses = np.asarray(responses, dtype=np.float64)
    n, j = responses.shape

    n_params = _count_free_params(model) + int(max(n_extra_params, 0))
    n_obs = n * j
    deviance = -2.0 * log_likelihood
    aic = deviance + 2 * n_params
    bic = deviance + n_params * np.log(n)        # BIC 用被试数作样本量
    # AICc：小样本修正
    denom = max(n - n_params - 1, 1)
    aicc = aic + (2 * n_params * (n_params + 1)) / denom

    sat_ll = saturated_log_likelihood(responses)
    sat_dev = -2.0 * sat_ll

    # CFI / TLI 需要基线（独立）模型
    ind_ll = _independence_log_likelihood(responses)
    ind_dev = -2.0 * ind_ll
    ind_params = j
    ind_aic = ind_dev + 2 * ind_params

    # 自由度
    df_model = max(n_obs - n_params, 1)
    df_null = max(n_obs - ind_params, 1)
    df_sat = 1  # 饱和模型 df 视为 1 以避免除零

    cfi = 1.0 - max(deviance - df_model, 0.0) / max(ind_dev - df_null, 1e-10)
    cfi = float(np.clip(cfi, 0.0, 1.0))

    # TLI 基于 AIC 的偏差-自由度比
    num = (ind_dev / df_null) - (deviance / df_model)
    den = (ind_dev / df_null) - 1.0
    tli = float(np.clip(num / den, 0.0, 1.0)) if abs(den) > 1e-10 else 0.0

    # RMSEA：基于饱和模型的偏差与自由度
    rmsea_num = max(deviance - (n_obs - n_params), 0.0)
    rmsea = float(np.sqrt(rmsea_num / max((n_obs - n_params) * (n - 1), 1)))

    return FitIndices(
        log_likelihood=float(log_likelihood),
        n_params=int(n_params),
        n_observations=int(n_obs),
        n_examinees=int(n),
        n_items=int(j),
        aic=float(aic),
        bic=float(bic),
        aicc=float(aicc),
        deviance=float(deviance),
        saturated_ll=float(sat_ll),
        saturated_deviance=float(sat_dev),
        cfi=cfi,
        tli=tli,
        rmsea=rmsea,
        model_name=model_name or getattr(model, "name", ""),
    )


# ====================================================================== #
# 被试诊断
# ====================================================================== #
@dataclass
class PatternDiagnosis:
    """被试知识状态诊断结果。

    Attributes
    ----------
    posterior : ndarray, shape (n_examinees, 2^K)
        每个被试属于每个属性模式的**后验概率**。
    map_patterns : ndarray, shape (n_examinees, K)
        MAP（最大后验）知识状态。
    eap_marginal : ndarray, shape (n_examinees, K)
        各属性的 EAP 边际掌握概率。
    eap_patterns : ndarray, shape (n_examinees, K)
        EAP 知识状态（边际概率 ≥ 0.5 为掌握）。
    map_confidence : ndarray, shape (n_examinees,)
        MAP 模式的后验概率（分类信度，越接近 1 越可靠）。
    attr_reliability : ndarray, shape (n_examinees, K)
        属性级分类信度：边际概率偏离 0.5 的程度（0~1）。
    pattern_indices : ndarray, shape (n_examinees,)
        MAP 模式在 ``attribute_patterns()`` 中的行索引。
    attr_names : List[str]
    """

    posterior: np.ndarray
    map_patterns: np.ndarray
    eap_marginal: np.ndarray
    eap_patterns: np.ndarray
    map_confidence: np.ndarray
    attr_reliability: np.ndarray
    pattern_indices: np.ndarray
    attr_names: List[str] = field(default_factory=list)

    @property
    def n_examinees(self) -> int:
        return self.posterior.shape[0]

    @property
    def n_attrs(self) -> int:
        return self.map_patterns.shape[1]

    def mastered_counts(self) -> np.ndarray:
        """各属性在样本中被判为掌握的人数。"""
        return self.map_patterns.sum(axis=0)

    def mastery_proportion(self) -> np.ndarray:
        """各属性的掌握比例。"""
        return self.map_patterns.mean(axis=0)

    def low_reliability_attrs(self, threshold: float = 0.6):
        """返回每个被试中信度低于阈值的属性索引列表。"""
        return [np.where(self.attr_reliability[i] < threshold)[0].tolist()
                for i in range(self.n_examinees)]

    def mean_confidence(self) -> float:
        return float(self.map_confidence.mean())

    def pattern_distribution(self) -> Dict[int, int]:
        """诊断出的模式频次分布。"""
        vals, counts = np.unique(self.pattern_indices, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def summary(self) -> str:
        lines = [
            "被试诊断汇总",
            "-" * 52,
            f"  被试数            : {self.n_examinees}",
            f"  属性数            : {self.n_attrs}",
            f"  平均分类信度      : {self.mean_confidence():.4f}",
            "",
            "  各属性掌握比例:",
        ]
        prop = self.mastery_proportion()
        for i in range(self.n_attrs):
            name = self.attr_names[i] if i < len(self.attr_names) else f"A{i+1}"
            bar = "#" * int(round(prop[i] * 20))
            lines.append(f"    {name:<10s} {prop[i]:.3f}  [{bar:<20s}]")
        n_patterns_used = len(self.pattern_distribution())
        lines.append("")
        lines.append(f"  出现过的知识状态数: {n_patterns_used}")
        return "\n".join(lines)

    def to_dataframe(self):
        """导出为 DataFrame（需 pandas）。"""
        import pandas as pd

        data = {"examinee_id": np.arange(self.n_examinees)}
        if self.attr_names and len(self.attr_names) == self.n_attrs:
            names = self.attr_names
        else:
            names = [f"A{i+1}" for i in range(self.n_attrs)]
        for i, nm in enumerate(names):
            data[f"{nm}_掌握"] = self.map_patterns[:, i]
            data[f"{nm}_概率"] = np.round(self.eap_marginal[:, i], 4)
            data[f"{nm}_信度"] = np.round(self.attr_reliability[:, i], 4)
        data["MAP后验概率"] = np.round(self.map_confidence, 4)
        data["模式索引"] = self.pattern_indices
        return pd.DataFrame(data)


def diagnose_patterns(
    model: CDModel,
    responses: np.ndarray,
    prior: Optional[np.ndarray] = None,
    method: str = "MAP",
    attr_names: Optional[Sequence[str]] = None,
) -> PatternDiagnosis:
    """对一批作答矩阵进行知识状态诊断。

    Parameters
    ----------
    model : CDModel
        已标定好的响应模型。
    responses : ndarray, shape (n_examinees, n_items)
        0/1 作答矩阵（同一批题目顺序）。
    prior : array-like, optional
        属性模式先验（长度 2^K），默认均匀。
    method : {'MAP', 'EAP'}
        主诊断方法；两种结果都会输出。
    attr_names : sequence of str, optional

    Returns
    -------
    PatternDiagnosis
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    if responses.shape[1] != model.n_items:
        raise ValueError(
            f"作答矩阵列数 {responses.shape[1]} 与题目数 {model.n_items} 不符"
        )

    n = responses.shape[0]
    s = model.n_patterns
    if prior is None:
        log_prior = np.full(s, -np.log(s))
    else:
        p = np.asarray(prior, dtype=np.float64).ravel()
        log_prior = np.log(p / p.sum() + 1e-300)

    prob = model._get_prob()                       # (S, J)
    log_p1 = np.log(prob)
    log_p0 = np.log(1.0 - prob)

    # 缺失作答（np.nan）按"该题不参与似然"处理：逐人只在已作答题上累加。
    # 若直接参与运算，nan 会污染整个似然向量，导致全员被判成模式 0 —— 静默错误。
    mask = np.isfinite(responses)
    posterior = np.empty((n, s), dtype=np.float64)
    for i in range(n):
        m = mask[i]
        if not m.any():
            # 一题未答：退回先验
            z = np.exp(log_prior - log_prior.max())
            posterior[i] = z / z.sum()
            continue
        xi = responses[i][m]
        ll = (xi[None, :] * log_p1[:, m] + (1.0 - xi)[None, :] * log_p0[:, m]).sum(axis=1)
        ll = ll + log_prior
        ll -= ll.max()
        z = np.exp(ll)
        posterior[i] = z / z.sum()

    patterns = model.q.attribute_patterns()
    pattern_indices = posterior.argmax(axis=1)
    map_patterns = patterns[pattern_indices].astype(np.int8)
    map_confidence = posterior[np.arange(n), pattern_indices]

    # EAP 边际：P(alpha_k = 1 | X) = Σ_{α: α_k=1} P(α | X)
    ones_mask = patterns.astype(np.float64)         # (S, K)
    eap_marginal = posterior @ ones_mask            # (n, K)
    eap_patterns = (eap_marginal >= 0.5).astype(np.int8)

    # 属性级信度：边际概率偏离 0.5 的程度
    attr_reliability = 2.0 * np.abs(eap_marginal - 0.5)

    names = list(attr_names) if attr_names is not None else list(model.q.attr_names or [])
    return PatternDiagnosis(
        posterior=posterior,
        map_patterns=map_patterns,
        eap_marginal=eap_marginal,
        eap_patterns=eap_patterns,
        map_confidence=map_confidence,
        attr_reliability=attr_reliability,
        pattern_indices=pattern_indices,
        attr_names=names,
    )


# ====================================================================== #
# 模型比较
# ====================================================================== #
@dataclass
class ModelComparison:
    """多模型拟合比较结果。"""

    rows: List[Dict[str, object]]
    best_aic: str = ""
    best_bic: str = ""
    best_aicc: str = ""

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def summary(self) -> str:
        if not self.rows:
            return "（无模型可比）"
        header = f"{'模型':<10s}{'参数数':>7s}{'-2LL':>13s}{'AIC':>13s}{'BIC':>13s}{'AICc':>13s}"
        lines = ["多模型拟合比较", "-" * len(header), header, "-" * len(header)]
        for r in self.rows:
            lines.append(
                f"{str(r['model']):<10s}{int(r['n_params']):>7d}"
                f"{float(r['deviance']):>13.2f}{float(r['aic']):>13.2f}"
                f"{float(r['bic']):>13.2f}{float(r['aicc']):>13.2f}"
            )
        lines.append("-" * len(header))
        lines.append(f"  AIC  最优 : {self.best_aic}")
        lines.append(f"  BIC  最优 : {self.best_bic}")
        lines.append(f"  AICc 最优 : {self.best_aicc}")
        return "\n".join(lines)


def compare_models(fits: Dict[str, "FitIndices"]) -> ModelComparison:
    """比较多组拟合指标，给出 AIC/BIC/AICc 最优模型。

    Parameters
    ----------
    fits : dict[str, FitIndices]
        模型名 -> 拟合指标。
    """
    rows = [f.to_dict() for f in fits.values()]
    cmp = ModelComparison(rows=rows)
    if rows:
        cmp.best_aic = min(rows, key=lambda r: r["aic"])["model"]
        cmp.best_bic = min(rows, key=lambda r: r["bic"])["model"]
        cmp.best_aicc = min(rows, key=lambda r: r["aicc"])["model"]
    return cmp


def likelihood_ratio_test(
    ll_restricted: float,
    n_params_restricted: int,
    ll_full: float,
    n_params_full: int,
) -> Dict[str, float]:
    """嵌套模型的似然比检验 (LRT)。

    统计量 ``LR = 2·(logL_full - logL_restricted)``，在零假设（受限模型
    为真）下渐近服从自由度为参数个数差的卡方分布。

    Returns
    -------
    dict
        含 ``lr_statistic``、``df``、``p_value``、``significant``。
    """
    from math import lgamma, log, exp

    lr = 2.0 * (ll_full - ll_restricted)
    df = int(n_params_full - n_params_restricted)
    if df <= 0:
        return {"lr_statistic": 0.0, "df": 0, "p_value": 1.0, "significant": 0.0}

    # 卡方分布上尾概率 P(X > lr)。
    # chi2(k) 的生存函数 = Q(k/2, lr/2)，Q 为正则化上不完全伽马函数。
    def chi2_sf(x: float, k: int) -> float:
        if x <= 0:
            return 1.0
        if k <= 0:
            return 1.0
        a = k / 2.0
        z = x / 2.0                      # 关键：伽马函数的自变量是统计量的一半
        if z < a + 1.0:
            # 下不完全伽马 P(a, z) 的级数展开，则 Q = 1 - P
            term = 1.0 / a
            total = term
            for n in range(1, 500):
                term *= z / (a + n)
                total += term
                if abs(term) < 1e-16 * abs(total):
                    break
            p_lower = total * exp(-z + a * log(z) - lgamma(a))
            return 1.0 - p_lower
        # Q(a, z) 的连分式（Lentz 算法）
        tiny = 1e-300
        b = z + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 500):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        return exp(-z + a * log(z) - lgamma(a)) * h

    p = float(np.clip(chi2_sf(lr, df), 0.0, 1.0))
    return {
        "lr_statistic": float(lr),
        "df": df,
        "p_value": p,
        "significant": 1.0 if p < 0.05 else 0.0,
    }


def _gamma_fn(x: float) -> float:
    """Lanczos 近似的 Gamma 函数（避免依赖 scipy）。"""
    from math import exp, log, pi

    g = 7
    coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if x < 0.5:
        return pi / (np.sin(pi * x) * _gamma_fn(1 - x))
    x -= 1
    a = coef[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += coef[i] / (x + i)
    return np.sqrt(2 * pi) * t ** (x + 0.5) * exp(-t) * a


# ====================================================================== #
# 参数恢复评价
# ====================================================================== #
def parameter_recovery(
    estimated: np.ndarray, true: np.ndarray
) -> Dict[str, float]:
    """参数恢复评价：相关系数、平均绝对误差、均方根误差、偏差。

    Parameters
    ----------
    estimated, true : array-like
        长度一致的参数向量。
    """
    e = np.asarray(estimated, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()
    if e.shape != t.shape:
        raise ValueError("estimated 与 true 长度必须一致")
    if len(e) < 2 or np.std(e) < 1e-12 or np.std(t) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(e, t)[0, 1])
    diff = e - t
    return {
        "correlation": corr,
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "bias": float(diff.mean()),
    }
