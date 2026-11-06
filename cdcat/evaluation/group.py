"""群体（班级 / 学校）层面的诊断报告。

个人报告回答"这个学生哪里不会"，群体报告回答**教学层面**的问题：

- 哪个知识点全班掌握率最低（薄弱知识点排序）
- 哪些属性已经教到位、哪些还需要补课
- 全班落在哪些典型掌握模式上（模式频次）
- 班级之间存在多大差距（分组对比）
- 某个学生相对全班的位置（个体 vs 群体）

数据来源有两条：

1. **仿真结果**：直接传 ``SimulationResult.records``（研究 / 校验用）
2. **真实作答**：传 ``responses`` 矩阵 + 已标定 ``model``，内部逐人做
   MAP / EAP 诊断（真实教学用）

两条路径产出的 :class:`GroupReport` 结构完全一致，后续统计与绘图共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["GroupReport", "build_group_report"]


@dataclass
class GroupReport:
    """群体诊断报告。

    Attributes
    ----------
    label : str
        报告标题（如班级名）。
    n_examinees : int
        有效被试数。
    attr_names : list of str
        属性名，长度 K。
    attr_mastery_rate : ndarray, shape (K,)
        各属性掌握率（按最终 0/1 判定统计）。
    attr_mean_prob : ndarray, shape (K,)
        各属性平均掌握概率（用边际概率，比 0/1 更细腻）。
    attr_sd_prob : ndarray, shape (K,)
        掌握概率的组内标准差 —— 越大说明班级内部差异越大（分层明显）。
    mastery_matrix : ndarray, shape (N, K), dtype int8
        每人的属性掌握判定。
    probabilities : ndarray, shape (N, K)
        每人的属性边际掌握概率。
    pattern_idx : ndarray, shape (N,), dtype int
        每人的估计模式编号。
    pattern_freq : ndarray, shape (2^K,)
        各模式在群体中的占比。
    n_answered : ndarray, shape (N,)
        每人的用题量（真实数据下为实际作答题数）。
    examinee_ids : list of str
        被试标识。
    group_labels : list of str or None
        被试所属分组（如班级 / 学校），用于分组对比。
    true_patterns : ndarray or None, shape (N, K)
        仿真数据下的真实模式，用于计算群体级判准率。
    """

    label: str
    n_examinees: int
    attr_names: List[str]
    attr_mastery_rate: np.ndarray
    attr_mean_prob: np.ndarray
    attr_sd_prob: np.ndarray
    mastery_matrix: np.ndarray
    probabilities: np.ndarray
    pattern_idx: np.ndarray
    pattern_freq: np.ndarray
    n_answered: np.ndarray
    examinee_ids: List[str] = field(default_factory=list)
    group_labels: Optional[List[str]] = None
    true_patterns: Optional[np.ndarray] = None
    method: str = "MAP"

    # ------------------------------------------------------------------ #
    # 派生统计
    # ------------------------------------------------------------------ #
    @property
    def n_attrs(self) -> int:
        return len(self.attr_names)

    @property
    def mean_mastered(self) -> float:
        """人均掌握属性数。"""
        return float(self.mastery_matrix.sum(axis=1).mean())

    @property
    def n_full_mastery(self) -> int:
        """完全掌握全部属性的人数。"""
        return int((self.mastery_matrix.sum(axis=1) == self.n_attrs).sum())

    @property
    def n_zero_mastery(self) -> int:
        """一个属性都没掌握的人数。"""
        return int((self.mastery_matrix.sum(axis=1) == 0).sum())

    @property
    def full_mastery_rate(self) -> float:
        return self.n_full_mastery / self.n_examinees if self.n_examinees else 0.0

    def weakest_attrs(self, n: int = 3) -> List[int]:
        """掌握率最低的 n 个属性索引（升序，最弱在前）。"""
        n = min(n, self.n_attrs)
        return np.argsort(self.attr_mastery_rate, kind="stable")[:n].tolist()

    def strongest_attrs(self, n: int = 3) -> List[int]:
        """掌握率最高的 n 个属性索引（降序）。"""
        n = min(n, self.n_attrs)
        return np.argsort(-self.attr_mastery_rate, kind="stable")[:n].tolist()

    def most_common_patterns(self, n: int = 5) -> List[int]:
        """出现频次最高的 n 个模式编号。"""
        n = min(n, len(self.pattern_freq))
        return np.argsort(-self.pattern_freq, kind="stable")[:n].tolist()

    def group_attr_mastery(self) -> Dict[str, np.ndarray]:
        """按 ``group_labels`` 分组计算各属性掌握率。

        Returns
        -------
        dict
            ``{组名: ndarray(K,)}``；无分组信息时返回空字典。
        """
        if not self.group_labels:
            return {}
        labels = np.asarray(self.group_labels)
        out: Dict[str, np.ndarray] = {}
        for g in dict.fromkeys(self.group_labels):
            mask = labels == g
            if mask.sum() == 0:
                continue
            out[str(g)] = self.mastery_matrix[mask].mean(axis=0)
        return out

    def compare_examinee(self, index: int) -> Dict[str, object]:
        """单个被试与群体常模的对比。"""
        if not 0 <= index < self.n_examinees:
            raise IndexError(f"被试索引越界: {index}")
        own = self.mastery_matrix[index].astype(bool)
        rate = self.attr_mastery_rate
        delta = self.probabilities[index] - self.attr_mean_prob
        return {
            "id": self.examinee_ids[index] if index < len(self.examinee_ids) else index,
            "mastered": own.tolist(),
            "n_mastered": int(own.sum()),
            "group_mean_mastered": self.mean_mastered,
            "attribute_prob": self.probabilities[index].tolist(),
            "percentile": float((self.mastery_matrix.sum(axis=1) <= own.sum()).mean()),
            "above_average": [bool(self.probabilities[index][k] > rate[k])
                              for k in range(self.n_attrs)],
            "delta_vs_group": delta.tolist(),
        }

    # ------------------------------------------------------------------ #
    # 表格化输出
    # ------------------------------------------------------------------ #
    def attr_dataframe(self):
        """属性层面汇总表（按掌握率升序，便于直接找薄弱点）。"""
        import pandas as pd

        order = np.argsort(self.attr_mastery_rate, kind="stable")
        return pd.DataFrame({
            "属性": [self.attr_names[i] for i in order],
            "掌握率": np.round(self.attr_mastery_rate[order], 4),
            "平均掌握概率": np.round(self.attr_mean_prob[order], 4),
            "组内标准差": np.round(self.attr_sd_prob[order], 4),
            "掌握人数": [int(self.mastery_matrix[:, i].sum()) for i in order],
        })

    def pattern_dataframe(self, n: int = 10):
        """模式频次表（前 n 个高频模式）。"""
        import pandas as pd

        rows = []
        for idx in self.most_common_patterns(n):
            if self.pattern_freq[idx] <= 0:
                continue
            bits = format(int(idx), f"0{self.n_attrs}b")
            rows.append({
                "模式": bits,
                "掌握属性": "、".join(
                    self.attr_names[k] for k, b in enumerate(bits) if b == "1"
                ) or "（全未掌握）",
                "人数": int(round(self.pattern_freq[idx] * self.n_examinees)),
                "占比": round(float(self.pattern_freq[idx]), 4),
            })
        return pd.DataFrame(rows)

    def group_dataframe(self):
        """分组对比表。"""
        import pandas as pd

        by_group = self.group_attr_mastery()
        if not by_group:
            return pd.DataFrame()
        labels = np.asarray(self.group_labels)
        rows = []
        for g, rates in by_group.items():
            mask = labels == g
            rows.append({
                "组别": g,
                "人数": int(mask.sum()),
                "人均掌握属性数": round(float(self.mastery_matrix[mask].sum(axis=1).mean()), 3),
                "最弱属性": self.attr_names[int(np.argmin(rates))],
                **{f"{self.attr_names[k]}掌握率": round(float(rates[k]), 4)
                   for k in range(self.n_attrs)},
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 文本 / 字典
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_examinees": self.n_examinees,
            "method": self.method,
            "n_attrs": self.n_attrs,
            "mean_mastered": round(self.mean_mastered, 4),
            "full_mastery_rate": round(self.full_mastery_rate, 4),
            "n_full_mastery": self.n_full_mastery,
            "n_zero_mastery": self.n_zero_mastery,
            "mean_items": round(float(self.n_answered.mean()), 3) if self.n_examinees else 0.0,
            "attributes": [
                {
                    "name": self.attr_names[k],
                    "mastery_rate": round(float(self.attr_mastery_rate[k]), 4),
                    "mean_prob": round(float(self.attr_mean_prob[k]), 4),
                    "sd_prob": round(float(self.attr_sd_prob[k]), 4),
                }
                for k in range(self.n_attrs)
            ],
            "weakest_attributes": [self.attr_names[i] for i in self.weakest_attrs(3)],
            "pattern_frequency": {
                format(int(i), f"0{self.n_attrs}b"): round(float(self.pattern_freq[i]), 4)
                for i in self.most_common_patterns(10) if self.pattern_freq[i] > 0
            },
            "group_comparison": [
                {"group": g, "rates": [round(float(v), 4) for v in rates]}
                for g, rates in self.group_attr_mastery().items()
            ],
        }

    def to_text(self) -> str:
        w = 62
        lines = ["=" * w, f"群体诊断报告  ·  {self.label}", "=" * w]
        lines.append(f"有效被试数    : {self.n_examinees}")
        lines.append(f"估计方法      : {self.method}")
        lines.append(f"人均掌握属性  : {self.mean_mastered:.2f} / {self.n_attrs}")
        lines.append(
            f"完全掌握      : {self.n_full_mastery} 人 ({self.full_mastery_rate:.1%})"
        )
        if self.n_zero_mastery:
            lines.append(f"全未掌握      : {self.n_zero_mastery} 人")
        lines.append("-" * w)
        lines.append("各属性掌握率（由低到高）")
        lines.append("-" * w)
        for i in np.argsort(self.attr_mastery_rate, kind="stable"):
            bar = "█" * int(round(self.attr_mastery_rate[i] * 20))
            lines.append(
                f"  {self.attr_names[i]:<14s} {self.attr_mastery_rate[i]:6.1%}  "
                f"P={self.attr_mean_prob[i]:.3f}  SD={self.attr_sd_prob[i]:.3f}  |{bar}"
            )

        weak = self.weakest_attrs(3)
        lines.append("-" * w)
        lines.append(
            "薄弱知识点（建议优先补强）: "
            + "、".join(self.attr_names[i] for i in weak)
        )

        top = self.most_common_patterns(5)
        if self.pattern_freq[top[0]] > 0:
            lines.append("-" * w)
            lines.append("典型掌握模式（前 5）")
            lines.append("-" * w)
            for idx in top:
                if self.pattern_freq[idx] <= 0:
                    continue
                bits = format(int(idx), f"0{self.n_attrs}b")
                names = "、".join(
                    self.attr_names[k] for k, b in enumerate(bits) if b == "1"
                ) or "（全未掌握）"
                lines.append(
                    f"  {bits}  {self.pattern_freq[idx]:6.1%}  "
                    f"{int(round(self.pattern_freq[idx] * self.n_examinees)):>4} 人   {names}"
                )

        by_group = self.group_attr_mastery()
        if by_group:
            lines.append("-" * w)
            lines.append("分组对比")
            lines.append("-" * w)
            header = "  " + "组别".ljust(10) + "".join(
                self.attr_names[k][:6].ljust(9) for k in range(self.n_attrs)
            )
            lines.append(header)
            for g, rates in by_group.items():
                lines.append(
                    "  " + str(g)[:9].ljust(10)
                    + "".join(f"{rates[k]:<9.3f}" for k in range(self.n_attrs))
                )
        lines.append("=" * w)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GroupReport(label={self.label!r}, n={self.n_examinees}, "
            f"K={self.n_attrs}, mean_mastered={self.mean_mastered:.2f})"
        )


# ====================================================================== #
# 构造
# ====================================================================== #
def _from_records(
    records: Sequence,
    attr_names: Sequence[str],
    q_matrix,
    item_names: Optional[Sequence[str]],
    method: str,
) -> tuple:
    """从仿真记录抽取诊断结果。"""
    n = len(records)
    k = len(attr_names)
    mastery = np.zeros((n, k), dtype=np.int8)
    probs = np.zeros((n, k), dtype=np.float64)
    pat_idx = np.zeros(n, dtype=np.int64)
    n_answered = np.zeros(n, dtype=np.float64)
    ids: List[str] = []
    truths = np.zeros((n, k), dtype=np.int8)
    has_truth = True
    for i, rec in enumerate(records):
        est = np.asarray(rec.est_pattern)
        mastery[i] = est.astype(np.int8)
        if rec.marginal_history is not None and len(rec.marginal_history):
            probs[i] = np.asarray(rec.marginal_history[-1], dtype=np.float64)
        else:
            probs[i] = est.astype(np.float64)
        pat_idx[i] = int(rec.est_pattern_idx)
        n_answered[i] = int(rec.n_answered)
        ids.append(str(getattr(rec, "examinee_id", i)))
        if rec.true_pattern is None:
            has_truth = False
        else:
            truths[i] = np.asarray(rec.true_pattern).astype(np.int8)
    return mastery, probs, pat_idx, n_answered, ids, (truths if has_truth else None)


def _from_responses(
    responses: np.ndarray,
    model,
    q_matrix,
    prior: Optional[np.ndarray],
    method: str,
    missing: str,
    hierarchy=None,
) -> tuple:
    """对真实作答矩阵逐人诊断。"""
    from ..estimation import AttributeEstimator

    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = resp.shape
    if j != q_matrix.n_items:
        raise ValueError(f"作答矩阵列数 ({j}) 与 Q 矩阵题目数 ({q_matrix.n_items}) 不一致")
    if missing == "ignore" and not np.isfinite(resp).all():
        bad = np.where(~np.isfinite(resp).all(axis=1))[0]
        raise ValueError(f"作答矩阵含缺失值（被试索引 {bad[:10].tolist()}…）")

    est = AttributeEstimator(model, prior=prior, method=method, hierarchy=hierarchy)
    k = q_matrix.n_attrs
    mastery = np.zeros((n, k), dtype=np.int8)
    probs = np.zeros((n, k), dtype=np.float64)
    pat_idx = np.zeros(n, dtype=np.int64)
    n_answered = np.zeros(n, dtype=np.float64)

    for i in range(n):
        row = resp[i]
        if missing == "wrong":
            obs = np.where(np.isfinite(row), row, 0.0)
            idx = np.arange(j)
        else:  # skip / ignore
            idx = np.where(np.isfinite(row))[0]
            obs = row[idx]
        res = est.estimate(obs, idx)
        mastery[i] = np.asarray(res.pattern).astype(np.int8)
        probs[i] = np.asarray(res.marginal, dtype=np.float64)
        pat_idx[i] = int(res.pattern_idx)
        n_answered[i] = len(idx)
    return mastery, probs, pat_idx, n_answered, None


def build_group_report(
    records: Optional[Sequence] = None,
    *,
    attr_names: Sequence[str],
    q_matrix,
    item_names: Optional[Sequence[str]] = None,
    model=None,
    responses: Optional[np.ndarray] = None,
    examinee_ids: Optional[Sequence[str]] = None,
    group_labels: Optional[Sequence[str]] = None,
    prior: Optional[np.ndarray] = None,
    method: str = "MAP",
    label: str = "群体诊断",
    missing: str = "skip",
    include_truth: bool = False,
    hierarchy=None,
) -> GroupReport:
    """构造群体报告。

    两条数据路径二选一：

    - ``records``：直接给仿真产出的施测记录列表
    - ``model`` + ``responses``：给已标定模型与真实作答矩阵

    Parameters
    ----------
    missing : {'skip', 'wrong', 'ignore'}
        缺失作答处理方式，仅对 ``responses`` 路径生效。
        ``skip`` = 该题不计入该被试的似然；``wrong`` = 视为答错；
        ``ignore`` = 遇到缺失直接报错。
    hierarchy : AttributeHierarchy, optional
        属性层级约束；给出后 ``responses`` 路径的逐人估计只在合法模式下进行
        （非法模式的先验被清零），可显著抬高判准率上限。
    """
    if records is None and (model is None or responses is None):
        raise ValueError("必须提供 records，或同时提供 model 与 responses")

    attr_names = [str(a) for a in attr_names]

    if records is not None:
        mastery, probs, pat_idx, n_answered, ids, truths = _from_records(
            records, attr_names, q_matrix, item_names, method
        )
        if not include_truth:
            truths = None
        if not ids:
            ids = [f"E{i + 1}" for i in range(len(records))]
    else:
        mastery, probs, pat_idx, n_answered, _ = _from_responses(
            responses, model, q_matrix, prior, method, missing, hierarchy
        )
        ids = list(examinee_ids) if examinee_ids is not None else [
            f"E{i + 1}" for i in range(len(mastery))
        ]
        truths = None

    n, k = mastery.shape
    n_patterns = 1 << k
    rates = mastery.mean(axis=0) if n else np.zeros(k)
    mean_prob = probs.mean(axis=0) if n else np.zeros(k)
    sd_prob = probs.std(axis=0) if n else np.zeros(k)

    counts = np.zeros(n_patterns, dtype=np.float64)
    for idx in pat_idx:
        if 0 <= int(idx) < n_patterns:
            counts[int(idx)] += 1.0
    freq = counts / counts.sum() if counts.sum() > 0 else counts

    return GroupReport(
        label=label,
        n_examinees=n,
        attr_names=attr_names,
        attr_mastery_rate=rates,
        attr_mean_prob=mean_prob,
        attr_sd_prob=sd_prob,
        mastery_matrix=mastery,
        probabilities=probs,
        pattern_idx=pat_idx,
        pattern_freq=freq,
        n_answered=n_answered,
        examinee_ids=[str(x) for x in ids],
        group_labels=list(group_labels) if group_labels is not None else None,
        true_patterns=truths,
        method=method,
    )
