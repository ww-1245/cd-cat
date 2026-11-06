"""被试诊断报告生成。

把单次施测结果转成**面向使用者**的可读报告，包含：

- 掌握/未掌握属性清单与置信度
- 与群体常模的对比（可选）
- 逐题作答明细与每题提供的诊断贡献
- 诊断建议（哪些属性需要加强）

输出格式支持纯文本、Markdown 与字典（便于导出 JSON/CSV）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..simulation import ExamineeRecord
from ..utils import pattern_to_str

__all__ = ["ExamineeReport", "build_examinee_report"]


@dataclass
class ExamineeReport:
    """单个被试的诊断报告。"""

    examinee_id: int
    true_pattern: Optional[np.ndarray]
    est_pattern: np.ndarray
    marginal: np.ndarray
    attr_names: List[str]
    n_answered: int
    administered: List[int]
    responses: List[int]
    item_names: List[str]
    item_attrs: List[List[str]]
    max_posterior: float
    stop_reason: str
    mastery_threshold: float = 0.5

    # ---------------- 派生 ----------------
    @property
    def mastered(self) -> List[int]:
        return [i for i in range(len(self.est_pattern)) if self.est_pattern[i] == 1]

    @property
    def not_mastered(self) -> List[int]:
        return [i for i in range(len(self.est_pattern)) if self.est_pattern[i] == 0]

    @property
    def correct_count(self) -> int:
        return int(sum(self.responses))

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.n_answered if self.n_answered else 0.0

    def attr_confidence(self, attr_idx: int) -> float:
        """该属性的诊断置信度：边际概率偏离 0.5 的程度（0~1）。"""
        return float(abs(self.marginal[attr_idx] - 0.5) * 2)

    def low_confidence_attrs(self, threshold: float = 0.6) -> List[int]:
        """诊断置信度偏低的属性（可能需要加题或人工复核）。"""
        return [i for i in range(len(self.est_pattern)) if self.attr_confidence(i) < threshold]

    @property
    def pattern_correct(self) -> Optional[bool]:
        if self.true_pattern is None:
            return None
        return bool(np.array_equal(self.est_pattern, self.true_pattern))

    # ---------------- 输出 ----------------
    def to_dict(self) -> dict:
        return {
            "examinee_id": self.examinee_id,
            "estimated_pattern": pattern_to_str(self.est_pattern),
            "true_pattern": None if self.true_pattern is None else pattern_to_str(self.true_pattern),
            "pattern_correct": self.pattern_correct,
            "n_answered": self.n_answered,
            "correct": self.correct_count,
            "accuracy": round(self.accuracy, 4),
            "max_posterior": round(self.max_posterior, 4),
            "stop_reason": self.stop_reason,
            "attributes": [
                {
                    "name": self.attr_names[i],
                    "mastered": bool(self.est_pattern[i]),
                    "probability": round(float(self.marginal[i]), 4),
                    "confidence": round(self.attr_confidence(i), 4),
                }
                for i in range(len(self.est_pattern))
            ],
            "responses": [
                {
                    "order": t + 1,
                    "item": self.item_names[it],
                    "attrs": self.item_attrs[t],
                    "correct": bool(r),
                }
                for t, (it, r) in enumerate(zip(self.administered, self.responses))
            ],
        }

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            f"认知诊断报告  ·  被试 #{self.examinee_id}",
            "=" * 60,
            f"用题量      : {self.n_answered} 题",
            f"答对        : {self.correct_count} / {self.n_answered} ({self.accuracy:.1%})",
            f"诊断模式    : {pattern_to_str(self.est_pattern)}",
            f"诊断置信度  : {self.max_posterior:.4f}",
            f"终止原因    : {self.stop_reason}",
        ]
        if self.true_pattern is not None:
            lines.append(f"真实模式    : {pattern_to_str(self.true_pattern)}")
            lines.append(f"是否全对    : {'是' if self.pattern_correct else '否'}")

        lines.append("-" * 60)
        lines.append("属性掌握情况")
        lines.append("-" * 60)
        for i in range(len(self.est_pattern)):
            mark = "✓ 已掌握" if self.est_pattern[i] else "✗ 未掌握"
            bar = self._bar(self.marginal[i])
            lines.append(
                f"  {self.attr_names[i]:<14s} {mark}   "
                f"P={self.marginal[i]:.3f}  置信={self.attr_confidence(i):.2f}  {bar}"
            )

        low = self.low_confidence_attrs()
        if low:
            names = [self.attr_names[i] for i in low]
            lines.append("-" * 60)
            lines.append(f"[注意] 以下属性诊断置信度偏低，建议加测或人工复核: {names}")

        lines.append("-" * 60)
        lines.append("作答明细")
        lines.append("-" * 60)
        for t, (it, r) in enumerate(zip(self.administered, self.responses)):
            mark = "✓" if r else "✗"
            attrs = " + ".join(self.item_attrs[t])
            lines.append(f"  {t+1:>3}. {self.item_names[it]:<10s} [{attrs}]  {mark}")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _bar(p: float, width: int = 20) -> str:
        filled = int(round(p * width))
        return "[" + "█" * filled + "·" * (width - filled) + "]"


def build_examinee_report(
    record: ExamineeRecord,
    attr_names: List[str],
    item_names: List[str],
    q_matrix,
    include_truth: bool = True,
) -> ExamineeReport:
    """从施测记录构造可读报告。

    Parameters
    ----------
    record : ExamineeRecord
    attr_names : list of str
    item_names : list of str
    q_matrix : QMatrix
    include_truth : bool
        是否包含真实模式（仿真评估用；真实施测应设为 False）。
    """
    # 用最终边际概率重建
    marginal = record.marginal_history[-1] if record.marginal_history.size else (
        record.est_pattern.astype(float)
    )
    item_attrs = []
    for it in record.administered:
        req = q_matrix.required_attrs(it)
        item_attrs.append([attr_names[k] for k in req])

    return ExamineeReport(
        examinee_id=record.examinee_id,
        true_pattern=record.true_pattern.copy() if (include_truth and record.true_pattern is not None) else None,
        est_pattern=np.asarray(record.est_pattern).copy(),
        marginal=np.asarray(marginal, dtype=float).copy(),
        attr_names=list(attr_names),
        n_answered=record.n_answered,
        administered=list(record.administered),
        responses=list(record.responses),
        item_names=list(item_names),
        item_attrs=item_attrs,
        max_posterior=float(record.max_posterior),
        stop_reason=record.stop_reason,
    )
