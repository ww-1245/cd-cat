"""一体化 CDM 标定分析流程。

把「标定 → 拟合指标 → 被试诊断 → 可视化数据准备」串成一条流水线，
对外只暴露一个 :func:`run_cdm_analysis` 入口。

典型用法::

    from cdcat import QMatrix
    from cdcat.pipeline import run_cdm_analysis

    q = QMatrix.from_csv("qmatrix.csv")
    analysis = run_cdm_analysis(q, responses, models=["DINA", "G-DINA"])
    print(analysis.summary())
    analysis.to_excel("cdm_report.xlsx")   # 需 openpyxl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .calibration import CalibrationResult, EMCalibrator
from .diagnostics import (
    AbsoluteFitResult,
    FitIndices,
    ModelComparison,
    PatternDiagnosis,
    analyse_absolute_fit,
    compare_models,
    compute_fit_indices,
    diagnose_patterns,
    likelihood_ratio_test,
    parameter_recovery,
)
from .estimation.nonparametric import (
    NONPARAMETRIC_METHODS,
    NonparametricResult,
    classify_nonparametric,
)
from .qmatrix import QMatrix

__all__ = ["CDMAnalysis", "run_cdm_analysis"]

#: 已知的嵌套关系（restricted ⊂ full）；只有这些组合的 LRT 才服从 χ²
_NESTED_PAIRS = {
    frozenset(("DINA", "G-DINA")),
    frozenset(("DINO", "G-DINA")),
}


@dataclass
class CDMAnalysis:
    """一次完整 CDM 分析的汇总结果。

    Attributes
    ----------
    q_matrix : QMatrix
    models : dict[str, CalibrationResult]
        各模型的标定结果。
    fits : dict[str, FitIndices]
        各模型的拟合指标。
    comparison : ModelComparison
        模型比较（AIC/BIC/AICc 择优）。
    diagnosis : dict[str, PatternDiagnosis]
        各模型的被试诊断结果。
    best_model : str
        按 BIC 选出的最优模型。
    lrt : dict, optional
        最优模型 vs 最简模型的似然比检验。
    recovery : dict, optional
        若提供了真实参数与真实模式，附带参数恢复评价。
    """

    q_matrix: QMatrix
    models: Dict[str, CalibrationResult] = field(default_factory=dict)
    fits: Dict[str, FitIndices] = field(default_factory=dict)
    comparison: Optional[ModelComparison] = None
    diagnosis: Dict[str, PatternDiagnosis] = field(default_factory=dict)
    best_model: str = ""
    lrt: Optional[dict] = None
    recovery: Optional[dict] = None
    absolute_fit: Optional["AbsoluteFitResult"] = None
    #: 非参数分类结果（方法名 → 结果）。这些方法不估计题目参数，
    #: 因此不进入 ``fits``（没有似然，算不出 AIC/BIC），单独成表。
    nonparametric: Dict[str, NonparametricResult] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def model(self, name: str):
        """取某模型的响应模型实例。"""
        return self.models[name].to_model(self.q_matrix)

    def summary(self) -> str:
        lines = ["=" * 62, "CDM 标定分析报告", "=" * 62]
        lines.append(
            f"题库: {self.q_matrix.n_items} 题 x {self.q_matrix.n_attrs} 属性 "
            f"({self.q_matrix.n_patterns} 种模式)"
        )
        lines.append("")
        if self.comparison is not None:
            lines.append(self.comparison.summary())
        lines.append("")
        lines.append(f"最优模型 (BIC): {self.best_model}")
        if self.best_model in self.fits:
            lines.append("")
            lines.append(self.fits[self.best_model].summary())
        if self.best_model in self.diagnosis:
            lines.append("")
            lines.append(self.diagnosis[self.best_model].summary())
        if self.nonparametric:
            lines.append("")
            lines.append("非参数分类基线（不使用题目参数，0 个自由参数）:")
            best_pmr = None
            if self.best_model in self.fits:
                # FitIndices 没有 pmr 字段，只在给了真实模式时才动态挂上
                best_pmr = getattr(self.fits[self.best_model], "pmr", None)
            for name, res in self.nonparametric.items():
                if res.pmr is not None:
                    row = (f"  {name:<8s} PMR={res.pmr:.4f}  AMR={res.amr:.4f}  "
                           f"平均把握={res.mean_confidence():.4f}")
                    if best_pmr is not None:
                        delta = best_pmr - res.pmr
                        if abs(delta) < 5e-5:
                            row += f"    与 {self.best_model} 持平"
                        elif delta > 0:
                            row += f"    {self.best_model} 领先 {delta:+.4f}"
                        else:
                            row += f"    非参数领先 {-delta:.4f}"
                else:
                    row = (f"  {name:<8s} 平均把握={res.mean_confidence():.4f}  "
                           f"平均距离差={np.nanmean(res.distance_gap()):.4f}")
                lines.append(row)
        if self.absolute_fit is not None:
            lines.append("")
            lines.append(self.absolute_fit.item_fit.summary())
            lines.append("")
            lines.append(self.absolute_fit.local_dependence.summary())
        if self.lrt is not None:
            lines.append("")
            lines.append(
                f"似然比检验: LR={self.lrt['lr_statistic']:.2f}, "
                f"df={self.lrt['df']}, p={self.lrt['p_value']:.4g} -> "
                f"{'模型间差异显著' if self.lrt['significant'] else '模型间无显著差异'}"
            )
            if self.lrt.get("note"):
                lines.append(f"  注意: {self.lrt['note']}")
        if self.recovery:
            lines.append("")
            lines.append("参数恢复评价:")
            for k, v in self.recovery.items():
                lines.append(
                    f"  {k:<12s} r={v['correlation']:.4f}  "
                    f"MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}"
                )
        lines.append("=" * 62)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def fit_table(self):
        """模型拟合指标对比表（含题目参数 / 先验参数的分解）。"""
        import pandas as pd

        rows = []
        for name, fit in self.fits.items():
            row = fit.to_dict()
            res = self.models.get(name)
            if res is not None:
                row["n_item_params"] = res.n_item_params
                row["n_prior_params"] = res.n_prior_params
                if res.attr_prior is not None:
                    row["prior_nonzero"] = int((res.attr_prior > 1e-5).sum())
            rows.append(row)
        return pd.DataFrame(rows)

    def param_table(self):
        """各模型的题目参数表（统一列：题号、Q向量、slip、guess、区分度）。"""
        import pandas as pd

        q = self.q_matrix
        rows = []
        for name, res in self.models.items():
            if res.slip is None:
                continue
            for i in range(q.n_items):
                rows.append({
                    "模型": name,
                    "题号": q.item_names[i],
                    "Q向量": "+".join(q.attr_names[k] for k in q.required_attrs(i)) or "—",
                    "slip": round(float(res.slip[i]), 4),
                    "guess": round(float(res.guess[i]), 4),
                    "区分度": round(float(1.0 - res.slip[i] - res.guess[i]), 4),
                })
        return pd.DataFrame(rows)

    def diagnosis_table(self, model_name: Optional[str] = None):
        """最优（或指定）模型的被试诊断表。"""
        name = model_name or self.best_model
        return self.diagnosis[name].to_dataframe()

    def nonparametric_table(self):
        """非参数基线的对照表。

        非参数方法没有似然，无法计算 AIC/BIC，因此单独成表；
        ``参数数`` 恒为 0，正是它作为"零参数基线"的意义所在。
        """
        import pandas as pd

        best_pmr = None
        if self.best_model in self.fits:
            best_pmr = getattr(self.fits[self.best_model], "pmr", None)
        rows = []
        for name, res in self.nonparametric.items():
            rows.append({
                "方法": name,
                "说明": NONPARAMETRIC_METHODS.get(name, ""),
                "参数数": 0,
                "PMR": None if res.pmr is None else round(res.pmr, 4),
                "AMR": None if res.amr is None else round(res.amr, 4),
                "平均判别把握": round(res.mean_confidence(), 4),
                "PMR 差（非参数−参数）": (
                    None if (best_pmr is None or res.pmr is None)
                    else round(res.pmr - best_pmr, 4)
                ),
            })
        return pd.DataFrame(rows)

    def to_excel(self, path: str) -> None:
        """把完整分析导出为多 sheet 的 Excel 文件。"""
        with _excel_writer(path) as writer:
            self.fit_table().to_excel(writer, sheet_name="模型拟合指标", index=False)
            self.param_table().to_excel(writer, sheet_name="题目参数", index=False)
            self.diagnosis_table().to_excel(writer, sheet_name="被试诊断", index=False)
            if self.nonparametric:
                self.nonparametric_table().to_excel(
                    writer, sheet_name="非参数基线", index=False,
                )
            if self.absolute_fit is not None:
                self.absolute_fit.item_fit.to_dataframe().to_excel(
                    writer, sheet_name="题目绝对拟合", index=False,
                )
                ld = self.absolute_fit.local_dependence
                ld.flagged_pairs().to_excel(
                    writer, sheet_name="局部依赖Q3", index=False,
                )
            if self.best_model in self.models:
                prior = self.models[self.best_model].attr_prior
                if prior is not None:
                    import pandas as pd

                    patterns = self.q_matrix.attribute_patterns()
                    rows = []
                    for i, p in enumerate(patterns):
                        rows.append({
                            "模式索引": i,
                            "模式": "".join(str(int(v)) for v in p),
                            "π(α)": round(float(prior[i]), 6),
                        })
                    pd.DataFrame(rows).to_excel(
                        writer, sheet_name="属性先验", index=False,
                    )


def _excel_writer(path: str):
    try:
        import pandas as pd

        return pd.ExcelWriter(path, engine="openpyxl")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "导出 Excel 需要 openpyxl，请先 `pip install openpyxl`"
        ) from exc


# ====================================================================== #
# 主入口
# ====================================================================== #
def run_cdm_analysis(
    q_matrix: QMatrix,
    responses: np.ndarray,
    models: Sequence[str] = ("DINA",),
    prior: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    attr_names: Optional[Sequence[str]] = None,
    true_slip: Optional[np.ndarray] = None,
    true_guess: Optional[np.ndarray] = None,
    true_patterns: Optional[np.ndarray] = None,
    estimate_prior: Optional[bool] = None,
    absolute_fit: bool = True,
    n_bins: int = 5,
    q3_threshold: float = 0.2,
    nonparametric: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> CDMAnalysis:
    """运行完整的 CDM 标定分析流水线。

    Parameters
    ----------
    q_matrix : QMatrix
    responses : ndarray, shape (n_examinees, n_items)
        0/1 作答矩阵。
    models : sequence of str
        要标定的模型列表，如 ``("DINA", "DINO", "G-DINA")``。
    prior : array-like, optional
        属性模式先验。
    max_iter, tol : float
        传给 :class:`EMCalibrator`。
    attr_names : sequence of str, optional
        属性名，用于报告。
    true_slip, true_guess : array-like, optional
        真实参数，提供后会附带参数恢复评价。
    true_patterns : ndarray, optional
        真实属性模式（n_examinees, K），提供后会附带 PMR/AMR。
    estimate_prior : bool, optional
        是否在 EM 中经验贝叶斯估计属性先验 π(α)。默认 ``None``
        （未提供 ``prior`` 时自动开启）。
    absolute_fit : bool
        是否对最优模型做题目级绝对拟合与 Q3 局部依赖分析（默认开启）。
    n_bins : int
        绝对拟合中每个潜在组内按剩余得分分的箱数。
    q3_threshold : float
        Q3 局部依赖的告警阈值，Yen 建议 0.2。
    nonparametric : sequence of str, optional
        要附带运行的非参数分类方法，如 ``("gdd", "hamming")``。这些方法只用
        Q 矩阵的理想反应模式，不估计题目参数；结果放在
        :attr:`CDMAnalysis.nonparametric`，不进 ``fits``（无似然，算不出 AIC/BIC）。
    verbose : bool

    Returns
    -------
    CDMAnalysis
    """
    q_matrix = q_matrix if isinstance(q_matrix, QMatrix) else QMatrix(q_matrix)
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    if responses.shape[1] != q_matrix.n_items:
        raise ValueError(
            f"作答矩阵列数 {responses.shape[1]} 与题目数 {q_matrix.n_items} 不符"
        )

    analysis = CDMAnalysis(q_matrix=q_matrix)

    for name in models:
        if verbose:
            print(f"[pipeline] 标定模型 {name} ...")
        cal = EMCalibrator(
            q_matrix, model_type=name, max_iter=max_iter, tol=tol,
            prior=prior, estimate_prior=estimate_prior,
        )
        result = cal.fit(responses, verbose=verbose)
        analysis.models[name] = result

        # 拟合指标（把 π(α) 的自由参数计入，与 CalibrationResult 口径一致）
        fitted = result.to_model(q_matrix)
        analysis.fits[name] = compute_fit_indices(
            responses, result.log_likelihood, fitted, model_name=name,
            n_extra_params=result.n_prior_params,
        )

        # 被试诊断：沿用标定得到的属性先验 π(α)
        diag_prior = prior if prior is not None else result.attr_prior
        analysis.diagnosis[name] = diagnose_patterns(
            fitted, responses, prior=diag_prior, attr_names=attr_names
        )

    # 模型比较
    analysis.comparison = compare_models(analysis.fits)
    analysis.best_model = analysis.comparison.best_bic or next(iter(analysis.fits))

    # 非参数基线（与标定完全独立，可在标定失败时单独使用）
    if nonparametric:
        for name in nonparametric:
            key = str(name).strip().lower()
            if key not in NONPARAMETRIC_METHODS:
                raise ValueError(
                    f"未知非参数方法 '{name}'，可选: {sorted(NONPARAMETRIC_METHODS)}"
                )
            if verbose:
                print(f"[pipeline] 非参数分类 {key} ...")
            analysis.nonparametric[key] = classify_nonparametric(
                q_matrix, responses, method=key, attr_names=attr_names
            )

    # 绝对拟合与局部依赖（对最优模型）
    if absolute_fit and analysis.best_model in analysis.models:
        best_res = analysis.models[analysis.best_model]
        analysis.absolute_fit = analyse_absolute_fit(
            best_res.to_model(q_matrix),
            responses,
            prior=prior if prior is not None else best_res.attr_prior,
            n_bins=n_bins,
            q3_threshold=q3_threshold,
        )

    # 似然比检验：最简（参数最少）模型 vs 最优模型
    if len(analysis.models) >= 2:
        ordered = sorted(
            analysis.models,
            key=lambda k: (analysis.models[k].n_params, -analysis.models[k].log_likelihood),
        )
        restricted = ordered[0]
        if restricted != analysis.best_model:
            full = analysis.best_model
            note = None
        else:
            # 最优模型本身就是最简模型：与"参数更多的次优模型"比较，
            # 仍给出信息量差异（此时 df 可能为 0，属正常情形）
            full = ordered[-1]
            note = "最简模型即为最优模型，此处为与参数最多模型的比较"
        r = analysis.models[restricted]
        f = analysis.models[full]
        analysis.lrt = likelihood_ratio_test(
            r.log_likelihood, r.n_params, f.log_likelihood, f.n_params
        )
        analysis.lrt["restricted"] = restricted
        analysis.lrt["full"] = full
        # χ² 近似只在嵌套模型下成立。DINA / DINO 嵌套于 G-DINA，其余
        # （RRUM、ACDM、LLM 与它们之间）都不构成嵌套关系，此时 LR 仍可
        # 作为"似然差有多大"的描述量，但 p 值不可信，必须提示改看 AIC/BIC。
        if frozenset((restricted, full)) not in _NESTED_PAIRS:
            analysis.lrt["note"] = (
                f"{restricted} 与 {full} 不是嵌套关系，χ² 近似不成立；"
                "模型选择请以 AIC / BIC 为准"
            )
        if note:
            analysis.lrt["note"] = note

    # 参数恢复评价
    if true_slip is not None and true_guess is not None:
        rec: Dict[str, dict] = {}
        for name, res in analysis.models.items():
            if res.slip is None:
                continue
            rec[f"{name}-slip"] = parameter_recovery(res.slip, true_slip)
            rec[f"{name}-guess"] = parameter_recovery(res.guess, true_guess)
        analysis.recovery = rec

    # 若给了真实模式，补充分类准确率
    if true_patterns is not None:
        true_patterns = np.asarray(true_patterns)
        for name, diag in analysis.diagnosis.items():
            pmr = float(np.mean([
                np.array_equal(diag.map_patterns[i], true_patterns[i])
                for i in range(len(true_patterns))
            ]))
            amr = float(np.mean([
                np.mean(diag.map_patterns[i] == true_patterns[i])
                for i in range(len(true_patterns))
            ]))
            analysis.fits[name].pmr = pmr
            analysis.fits[name].amr = amr

        for res in analysis.nonparametric.values():
            acc = res.accuracy(true_patterns)
            res.pmr = acc["pmr"]
            res.amr = acc["amr"]

    return analysis
