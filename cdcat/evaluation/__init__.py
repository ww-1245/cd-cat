"""性能评估模块。

从 ``SimulationResult`` 计算论文级的评估指标与对比表：

- 判准率：PMR / AMR（整体与分属性）
- 效率：平均题量、题量分布
- 题库质量：曝光率、题库利用率、曝光均匀性（可用 χ² 检验）
- 收敛速度：达到给定精度所需题量
- 多准则对比表
- 题库信息量分析（``analyse_items``）
- 被试诊断报告（``build_examinee_report``）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..simulation import SimulationResult
from .item_analysis import ItemAnalysisResult, analyse_items
from .report import ExamineeReport, build_examinee_report
from .group import GroupReport, build_group_report

__all__ = [
    "Evaluator",
    "EvaluationReport",
    "ItemAnalysisResult",
    "analyse_items",
    "ExamineeReport",
    "build_examinee_report",
    "GroupReport",
    "build_group_report",
]


@dataclass
class EvaluationReport:
    """单个仿真的评估报告。"""

    label: str
    pmr: float
    amr: float
    avg_items: float
    items_sd: float
    test_utilization: float
    exposure_sd: float
    max_exposure_rate: float
    min_exposure_rate: float
    attribute_accuracy: np.ndarray
    pmr_ci: tuple
    amr_ci: tuple

    def to_row(self) -> dict:
        return {
            "方案": self.label,
            "PMR": round(self.pmr, 4),
            "AMR": round(self.amr, 4),
            "平均题量": round(self.avg_items, 2),
            "题量标准差": round(self.items_sd, 2),
            "题库利用率": round(self.test_utilization, 4),
            "曝光率标准差": round(self.exposure_sd, 4),
            "最大曝光率": round(self.max_exposure_rate, 4),
            "PMR 95%CI": f"[{self.pmr_ci[0]:.3f}, {self.pmr_ci[1]:.3f}]",
            "AMR 95%CI": f"[{self.amr_ci[0]:.3f}, {self.amr_ci[1]:.3f}]",
        }

    def summary(self) -> str:
        lines = [
            f"评估方案: {self.label}",
            f"  模式判准率 PMR : {self.pmr:.4f}  95%CI [{self.pmr_ci[0]:.4f}, {self.pmr_ci[1]:.4f}]",
            f"  属性判准率 AMR : {self.amr:.4f}  95%CI [{self.amr_ci[0]:.4f}, {self.amr_ci[1]:.4f}]",
            f"  平均题量       : {self.avg_items:.2f} (SD={self.items_sd:.2f})",
            f"  题库利用率     : {self.test_utilization:.4f}",
            f"  曝光率         : SD={self.exposure_sd:.4f}, "
            f"范围 [{self.min_exposure_rate:.4f}, {self.max_exposure_rate:.4f}]",
            f"  各属性判准率   : " + " ".join(f"{v:.3f}" for v in self.attribute_accuracy),
        ]
        return "\n".join(lines)


class Evaluator:
    """从仿真结果计算评估指标。

    Parameters
    ----------
    confidence : float
        置信区间水平，默认 0.95（正态近似）。

    Examples
    --------
    >>> ev = Evaluator()
    >>> report = ev.evaluate(result, label="PWKL")
    >>> print(report.summary())  # doctest: +SKIP
    """

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence
        # 正态分位数
        from scipy import stats

        self.z = stats.norm.ppf(1 - (1 - confidence) / 2)

    # ------------------------------------------------------------------ #
    def _ci(self, values: np.ndarray) -> tuple:
        """均值的正态近似置信区间。"""
        if len(values) == 0:
            return (0.0, 0.0)
        m = values.mean()
        if len(values) == 1:
            return (float(m), float(m))
        se = values.std(ddof=1) / np.sqrt(len(values))
        return (float(m - self.z * se), float(m + self.z * se))

    def evaluate(self, result: SimulationResult, label: Optional[str] = None) -> EvaluationReport:
        """计算单个仿真结果的完整评估报告。"""
        pc = np.array([r.pattern_correct for r in result.records], dtype=float)
        aa = np.array([r.attribute_accuracy for r in result.records], dtype=float)
        ni = np.array([r.n_answered for r in result.records], dtype=float)
        exposure = result.item_exposure_rate()

        return EvaluationReport(
            label=label or str(result.config.get("selector", "run")),
            pmr=float(pc.mean()) if len(pc) else 0.0,
            amr=float(aa.mean()) if len(aa) else 0.0,
            avg_items=float(ni.mean()) if len(ni) else 0.0,
            items_sd=float(ni.std(ddof=1)) if len(ni) > 1 else 0.0,
            test_utilization=result.test_utilization(),
            exposure_sd=float(exposure.std()),
            max_exposure_rate=float(exposure.max()) if len(exposure) else 0.0,
            min_exposure_rate=float(exposure.min()) if len(exposure) else 0.0,
            attribute_accuracy=result.attribute_accuracy_by_attr(),
            pmr_ci=self._ci(pc),
            amr_ci=self._ci(aa),
        )

    # ------------------------------------------------------------------ #
    def compare(self, results: Dict[str, SimulationResult]) -> "object":
        """多方案对比，返回 pandas DataFrame。"""
        import pandas as pd

        rows = [self.evaluate(res, label=name).to_row() for name, res in results.items()]
        return pd.DataFrame(rows)

    def compare_text(self, results: Dict[str, SimulationResult]) -> str:
        """多方案对比的文本表格。"""
        df = self.compare(results)
        return df.to_string(index=False)

    # ------------------------------------------------------------------ #
    # 进阶分析
    # ------------------------------------------------------------------ #
    def items_to_criterion(
        self,
        result: SimulationResult,
        target_pmr: float = 0.9,
    ) -> Optional[float]:
        """估计达到目标 PMR 所需的平均题量（基于固定题量仿真的记录）。

        用记录中的边际概率历史，逐题计算累积 PMR，找到首次达到目标的题量。
        """
        needed: List[int] = []
        for r in result.records:
            if r.marginal_history.size == 0:
                continue
            mh = r.marginal_history
            hit = None
            for t in range(1, mh.shape[0]):
                est = (mh[t] >= 0.5).astype(np.int8)
                if np.array_equal(est, r.true_pattern):
                    hit = t
                    break
            if hit is not None:
                needed.append(hit)
        if not needed:
            return None
        return float(np.mean(needed))

    def exposure_chi_square(self, result: SimulationResult) -> dict:
        """对题目曝光分布做均匀性 χ² 检验。

        返回 ``{'statistic':..., 'p_value':..., 'df':...}``。
        """
        from scipy import stats

        usage = result.item_usage()
        total = usage.sum()
        j = len(usage)
        if total == 0 or j == 0:
            return {"statistic": 0.0, "p_value": 1.0, "df": 0}
        expected = total / j
        chi2 = float(np.sum((usage - expected) ** 2 / expected))
        df = j - 1
        p = float(stats.chi2.sf(chi2, df))
        return {"statistic": chi2, "p_value": p, "df": df}

    def attribute_difficulty(self, result: SimulationResult) -> np.ndarray:
        """由真实模式估计各属性的掌握比例（用于分析难度分布）。"""
        truths = np.vstack([r.true_pattern for r in result.records])
        return truths.mean(axis=0)

    def efficiency_index(self, result: SimulationResult) -> float:
        """效率指数：单位题量下的属性判准率增益。

        ``AMR / avg_items``，越大表示用题越少而判准越高。
        """
        a = result.avg_items()
        return float(result.amr() / a) if a > 0 else 0.0

    def full_report(self, results, title: str = "CD-CAT 评估报告") -> str:
        """生成完整的多方案文本报告。"""
        if isinstance(results, SimulationResult):
            results = {"result": results}
        lines = ["=" * 70, title, "=" * 70, ""]
        for name, res in results.items():
            rep = self.evaluate(res, label=name)
            lines.append(rep.summary())
            chi = self.exposure_chi_square(res)
            lines.append(
                f"  曝光均匀性 χ² 检验: χ²={chi['statistic']:.2f}, df={chi['df']}, "
                f"p={chi['p_value']:.4f} {'(均匀)' if chi['p_value'] > 0.05 else '(不均匀)'}"
            )
            lines.append(f"  效率指数          : {self.efficiency_index(res):.4f}")
            lines.append("")
        if len(results) > 1:
            lines.append("-" * 70)
            lines.append("多方案对比:")
            lines.append(self.compare_text(results))
        lines.append("=" * 70)
        return "\n".join(lines)
