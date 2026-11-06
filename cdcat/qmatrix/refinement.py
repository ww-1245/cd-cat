"""Q 矩阵校验与自动修正建议（Qval / PVAF）。

为什么需要它
------------
Q 矩阵是 CDM 的**先验假设**，一旦标错，后续所有诊断都会系统性偏掉。
绝对拟合能告诉你"哪些题拟合不好"，但不会告诉你**该怎么改**。
本模块补上这一步：对每道题反推"最可能真实的属性集合"。

方法（de la Torre & Chiu, 2016；Qval 部分见 Chiu, 2013）
--------------------------------------------------------
记被试 i 的估计属性模式为 :math:`\\hat{\\alpha}_i`，第 j 题的作答为
:math:`X_{ij}`。对任一候选属性集合 :math:`s`，定义理想作答

.. math::
    \\eta_{is} = \\prod_{k \\in s} \\hat{\\alpha}_{ik}
    \\quad(\\text{连接型理想反应})

把被试按 :math:`\\eta_{is}` 分成"全掌握组"与"未全掌握组"，两组的
实际答对率记为 :math:`\\hat P_1, \\hat P_0`。定义**方差解释比例**

.. math::
    \\mathrm{PVAF}(s) = 1 -
    \\frac{\\sum_{\\eta \\in \\{0,1\\}} n_\\eta (\\hat P_\\eta - \\eta)^2}
         {\\sum_i (X_{ij} - \\bar X_j)^2}

直觉：若 :math:`s` 就是真实的考查属性集，则全掌握组应当基本答对
（:math:`\\hat P_1 \\to 1`）、另一组基本答错（:math:`\\hat P_0 \\to 0`），
残差平方和趋近 0，PVAF 趋近 1。

**选解规则**：在 PVAF ≥ 阈值的所有候选中，取属性个数最少者（奥卡姆剃刀）；
若没有候选达标，则取 PVAF 最大者。这样既能揪出漏标，也不会无脑加属性。

计算效率
--------
候选集合共 :math:`2^K - 1` 个，但理想向量只取决于模式编号：若把模式
编号与属性集合都用位掩码表示，则 :math:`\\eta = [\\,(idx \\,\\&\\, mask) = mask\\,]`，
全部候选可用一次向量化扫描算完，无需逐人循环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["QSuggestion", "QValidationReport", "validate_q_matrix"]


# ---------------------------------------------------------------------- #
# 结果容器
# ---------------------------------------------------------------------- #
@dataclass
class QSuggestion:
    """单道题的 Q 行修正建议。"""

    item_idx: int
    item_name: str
    current_attrs: List[int]
    suggested_attrs: List[int]
    pvaf_current: float
    pvaf_suggested: float
    attr_names: List[str] = field(default_factory=list)

    @property
    def added(self) -> List[int]:
        return sorted(set(self.suggested_attrs) - set(self.current_attrs))

    @property
    def removed(self) -> List[int]:
        return sorted(set(self.current_attrs) - set(self.suggested_attrs))

    @property
    def improvement(self) -> float:
        return float(self.pvaf_suggested - self.pvaf_current)

    def _fmt(self, idxs: Sequence[int]) -> str:
        if not idxs:
            return "（空）"
        if self.attr_names and max(idxs) < len(self.attr_names):
            return "+".join(self.attr_names[i] for i in idxs)
        return "+".join(f"A{i + 1}" for i in idxs)

    def current_attrs_str(self) -> str:
        return self._fmt(self.current_attrs)

    def suggested_attrs_str(self) -> str:
        return self._fmt(self.suggested_attrs)

    def describe(self) -> str:
        parts = []
        if self.added:
            parts.append(f"新增 {self._fmt(self.added)}")
        if self.removed:
            parts.append(f"删除 {self._fmt(self.removed)}")
        return "；".join(parts) or "无变化"

    def to_dict(self) -> dict:
        return {
            "item": self.item_name,
            "item_idx": self.item_idx,
            "current": self.current_attrs_str(),
            "suggested": self.suggested_attrs_str(),
            "pvaf_current": round(self.pvaf_current, 4),
            "pvaf_suggested": round(self.pvaf_suggested, 4),
            "improvement": round(self.improvement, 4),
            "added": [self.attr_names[i] for i in self.added] or [f"A{i+1}" for i in self.added],
            "removed": [self.attr_names[i] for i in self.removed] or [f"A{i+1}" for i in self.removed],
        }


@dataclass
class QValidationReport:
    """Q 矩阵校验总结果。"""

    threshold: float
    attr_names: List[str]
    item_names: List[str]
    item_pvaf: np.ndarray                 # (J,) 当前 Q 行的 PVAF
    item_best_pvaf: np.ndarray            # (J,) 最优候选的 PVAF
    item_best_attrs: List[List[int]]      # (J,) 最优候选的属性索引
    item_current_attrs: List[List[int]]   # (J,) 当前 Q 行的属性索引
    suggestions: List[QSuggestion] = field(default_factory=list)
    n_candidates_scanned: int = 0
    max_attrs_per_item: int = 0

    # ---- 计数 ----
    @property
    def n_items(self) -> int:
        return len(self.item_pvaf)

    @property
    def n_suspect(self) -> int:
        return len(self.suggestions)

    @property
    def n_additions(self) -> int:
        return sum(1 for s in self.suggestions if s.added)

    @property
    def n_removals(self) -> int:
        return sum(1 for s in self.suggestions if s.removed)

    @property
    def mean_pvaf(self) -> float:
        return float(self.item_pvaf.mean()) if self.n_items else 0.0

    # ---- 输出 ----
    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "n_items": self.n_items,
            "n_suspect": self.n_suspect,
            "n_additions": self.n_additions,
            "n_removals": self.n_removals,
            "mean_pvaf_current": round(self.mean_pvaf, 4),
            "mean_pvaf_best": round(float(self.item_best_pvaf.mean()), 4) if self.n_items else 0.0,
            "attr_names": list(self.attr_names),
            "suggestions": [s.to_dict() for s in self.suggestions],
            "item_pvaf": [round(float(v), 4) for v in self.item_pvaf],
        }

    def to_dataframe(self):
        """全部题目的 PVAF 明细表（按改善量降序）。"""
        import pandas as pd

        rows = []
        for i in range(self.n_items):
            cur = set(self.item_current_attrs[i])
            best = set(self.item_best_attrs[i])
            rows.append({
                "题号": self.item_names[i],
                "原 Q 行": _fmt_attrs(self.item_current_attrs[i], self.attr_names),
                "建议 Q 行": _fmt_attrs(self.item_best_attrs[i], self.attr_names),
                "原 PVAF": round(float(self.item_pvaf[i]), 4),
                "最优 PVAF": round(float(self.item_best_pvaf[i]), 4),
                "改善量": round(float(self.item_best_pvaf[i] - self.item_pvaf[i]), 4),
                "变化": ("+" if best - cur else "") + ("-" if cur - best else "") or "",
            })
        df = pd.DataFrame(rows)
        return df.sort_values("改善量", ascending=False).reset_index(drop=True)

    def summary(self, top_n: int = 10) -> str:
        lines = [
            "=" * 64,
            "Q 矩阵校验（Qval / PVAF）",
            "=" * 64,
            f"  题目数            : {self.n_items}",
            f"  判定阈值          : PVAF ≥ {self.threshold}",
            f"  扫描候选数/题     : {self.n_candidates_scanned}"
            f"（每行最多 {self.max_attrs_per_item} 个属性）",
            f"  平均 PVAF（当前） : {self.mean_pvaf:.4f}",
            f"  疑似需修正题目    : {self.n_suspect} "
            f"(建议新增 {self.n_additions} / 建议删除 {self.n_removals})",
        ]
        if self.suggestions:
            lines.append("")
            lines.append("  修正建议（按改善量降序）:")
            for s in self.suggestions[:top_n]:
                lines.append(
                    f"    {s.item_name:<10s} {s.current_attrs_str():<22s} -> "
                    f"{s.suggested_attrs_str():<22s} "
                    f"PVAF {s.pvaf_current:.3f} -> {s.pvaf_suggested:.3f}"
                )
        else:
            lines.append("  未发现需要修正的 Q 行。")
        lines.append("=" * 64)
        return "\n".join(lines)

    def apply(self, q_matrix, min_improvement: float = 0.0):
        """按建议生成**新的** Q 矩阵（不修改原矩阵）。

        Parameters
        ----------
        q_matrix : QMatrix
        min_improvement : float
            只采纳 PVAF 改善量不小于该值的建议，默认 0（全部采纳）。
        """
        from . import QMatrix

        matrix = np.array(q_matrix.matrix, dtype=np.int8, copy=True)
        for s in self.suggestions:
            if s.improvement < min_improvement:
                continue
            matrix[s.item_idx] = 0
            for k in s.suggested_attrs:
                matrix[s.item_idx, k] = 1
        return QMatrix(matrix, attr_names=q_matrix.attr_names,
                       item_names=q_matrix.item_names)


def _fmt_attrs(idxs: Sequence[int], names: Sequence[str]) -> str:
    if not idxs:
        return "（空）"
    if names and max(idxs) < len(names):
        return "+".join(names[i] for i in idxs)
    return "+".join(f"A{i + 1}" for i in idxs)


# ---------------------------------------------------------------------- #
# 主函数
# ---------------------------------------------------------------------- #
def _pattern_masks(n_attrs: int, max_attrs: Optional[int]) -> List[int]:
    """枚举候选属性集合的位掩码。

    属性 k 对应位 ``1 << (n_attrs - 1 - k)``，与 :class:`QMatrix` 的模式编号
    约定保持一致（这样理想向量可以直接由模式编号位运算得到）。
    """
    bits = [1 << (n_attrs - 1 - k) for k in range(n_attrs)]
    limit = n_attrs if max_attrs is None else max(1, min(int(max_attrs), n_attrs))
    masks: List[int] = []
    for size in range(1, limit + 1):
        for combo in combinations(range(n_attrs), size):
            m = 0
            for k in combo:
                m |= bits[k]
            masks.append(m)
    return masks


def _pvaf_vectorized(x: np.ndarray, idx: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """一次算出所有候选掩码的 PVAF。

    ``eta = (idx & mask) == mask`` 是全向量化运算，因此 K 稍大时也很快。
    """
    n = x.shape[0]
    x_mean = x.mean()
    ss_tot = float(((x - x_mean) ** 2).sum())
    if ss_tot <= 1e-12:                       # 全对或全错：无区分信息
        return np.zeros(len(masks), dtype=np.float64)

    out = np.zeros(len(masks), dtype=np.float64)
    for pos, mask in enumerate(masks):
        eta = ((idx & mask) == mask).astype(np.float64)
        n1 = float(eta.sum())
        n0 = float(n - n1)
        if n1 <= 0 or n0 <= 0:                # 退化分组，无信息
            out[pos] = 0.0
            continue
        p1 = float((eta * x).sum()) / n1
        p0 = float(((1.0 - eta) * x).sum()) / n0
        ss_res = n1 * (p1 - 1.0) ** 2 + n0 * (p0 - 0.0) ** 2
        out[pos] = 1.0 - ss_res / ss_tot
    return out


def validate_q_matrix(
    q_matrix,
    responses: np.ndarray,
    model,
    pvaf_threshold: float = 0.95,
    max_attrs_per_item: Optional[int] = None,
    alpha_hat: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    missing: str = "skip",
) -> QValidationReport:
    """对整份 Q 矩阵做逐题校验，输出修正建议。

    Parameters
    ----------
    q_matrix : QMatrix
    responses : ndarray, shape (n_examinees, n_items)
        0/1 作答矩阵（允许含 NaN，见 ``missing``）。
    model : CDModel
        已标定的响应模型，用于推断被试属性模式。
    pvaf_threshold : float
        PVAF 判定阈值，默认 0.95。达标的候选中优先选**属性最少**的。
    max_attrs_per_item : int, optional
        限制候选集合的最大属性数，默认 ``min(K, 4)``（控制 :math:`2^K` 规模）。
    alpha_hat : ndarray, optional
        直接提供被试属性模式（N, K）；不给则用 ``model`` 逐个估计。
    missing : {'skip', 'wrong'}
        缺失作答处理方式。

    Returns
    -------
    QValidationReport

    Examples
    --------
    >>> from cdcat import QMatrix, calibrate, validate_q_matrix
    >>> q = QMatrix.from_random(30, 4, seed=0)              # doctest: +SKIP
    >>> resp = ...                                          # doctest: +SKIP
    >>> model, _ = calibrate(q, resp, "DINA")               # doctest: +SKIP
    >>> rep = validate_q_matrix(q, resp, model)             # doctest: +SKIP
    >>> rep.n_suspect                                        # doctest: +SKIP
    """
    from ..estimation import AttributeEstimator

    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = resp.shape
    if j != q_matrix.n_items:
        raise ValueError(f"作答矩阵列数 ({j}) 与 Q 矩阵题目数 ({q_matrix.n_items}) 不一致")
    k = q_matrix.n_attrs
    if k > 12:
        raise ValueError(
            f"属性数 K={k} 过大，候选集合枚举不可行；请手动传入 alpha_hat "
            "并降低 max_attrs_per_item，或先用属性层级/专家知识筛减属性"
        )

    # ---- 1) 被试属性模式 ----
    if alpha_hat is None:
        est = AttributeEstimator(model, prior=prior, method="MAP")
        alpha_hat = np.zeros((n, k), dtype=np.int8)
        for i in range(n):
            if missing == "wrong":
                obs = np.where(np.isfinite(resp[i]), resp[i], 0.0)
                idx = np.arange(j)
            else:
                idx = np.where(np.isfinite(resp[i]))[0]
                obs = resp[i][idx]
            alpha_hat[i] = np.asarray(est.estimate(obs, idx).pattern).astype(np.int8)
    else:
        alpha_hat = np.asarray(alpha_hat, dtype=np.int8)
        if alpha_hat.shape != (n, k):
            raise ValueError(f"alpha_hat 形状应为 {(n, k)}，实际 {alpha_hat.shape}")

    # 模式编号（与 QMatrix 位序一致）：属性 k 对应 1 << (K-1-k)
    idx = np.zeros(n, dtype=np.int64)
    for kk in range(k):
        idx |= (alpha_hat[:, kk].astype(np.int64) << (k - 1 - kk))

    masks = _pattern_masks(k, max_attrs_per_item)
    mas = np.asarray(masks, dtype=np.int64)

    attr_names = list(q_matrix.attr_names)
    item_names = list(q_matrix.item_names or [f"I{i + 1}" for i in range(j)])

    item_pvaf = np.zeros(j, dtype=np.float64)
    item_best = np.zeros(j, dtype=np.float64)
    best_attrs: List[List[int]] = []
    cur_attrs: List[List[int]] = []
    suggestions: List[QSuggestion] = []

    for it in range(j):
        x = resp[:, it]
        valid = np.isfinite(x)
        if valid.sum() < max(10, k * 5) or np.unique(x[valid]).size < 2:
            # 有效样本太少或作答无变异：跳过，避免给出不可靠建议
            cur = [kk for kk in range(k) if q_matrix.matrix[it, kk] == 1]
            cur_attrs.append(cur)
            best_attrs.append(list(cur))
            item_pvaf[it] = float("nan")
            item_best[it] = float("nan")
            continue

        xv = x[valid]
        iv = idx[valid]
        pvafs = _pvaf_vectorized(xv, iv, mas)

        # ---- 选解：达标者中取属性最少，否则取 PVAF 最大 ----
        ok = np.where(pvafs >= pvaf_threshold)[0]
        if ok.size:
            sizes = np.array([bin(int(mas[p])).count("1") for p in ok])
            pick = int(ok[np.argmin(sizes)])
        else:
            pick = int(np.argmax(pvafs))

        mask = int(mas[pick])
        sug = [kk for kk in range(k) if mask & (1 << (k - 1 - kk))]
        cur = [kk for kk in range(k) if q_matrix.matrix[it, kk] == 1]
        p_best = float(pvafs[pick])

        # 当前 Q 行的 PVAF
        cur_mask = 0
        for kk in cur:
            cur_mask |= 1 << (k - 1 - kk)
        if cur_mask in masks:
            p_cur = float(pvafs[masks.index(cur_mask)])
        else:
            p_cur = float(_pvaf_vectorized(xv, iv, np.array([cur_mask], dtype=np.int64))[0]) \
                if cur_mask else 0.0

        item_pvaf[it] = p_cur
        item_best[it] = p_best
        cur_attrs.append(cur)
        best_attrs.append(sug)

        if set(cur) != set(sug) and p_best - p_cur > 1e-6:
            suggestions.append(QSuggestion(
                item_idx=it,
                item_name=item_names[it],
                current_attrs=cur,
                suggested_attrs=sug,
                pvaf_current=p_cur,
                pvaf_suggested=p_best,
                attr_names=attr_names,
            ))

    suggestions.sort(key=lambda s: -s.improvement)
    return QValidationReport(
        threshold=float(pvaf_threshold),
        attr_names=attr_names,
        item_names=item_names,
        item_pvaf=item_pvaf,
        item_best_pvaf=item_best,
        item_best_attrs=best_attrs,
        item_current_attrs=cur_attrs,
        suggestions=suggestions,
        n_candidates_scanned=len(masks),
        max_attrs_per_item=(max_attrs_per_item or min(k, 4)),
    )
