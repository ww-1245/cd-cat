"""项目功能差异（DIF）分析。

在认知诊断框架下，DIF 的含义比 IRT 更具体：**同一知识状态的两名考生，
只因所属群体不同而答对同一题的概率不同**，就说明该题存在 DIF。

条件变量怎么选？
----------------
传统 MH-DIF 用**测验总分**分层。CDM 的优势是有更贴近构念的条件的量 ——
**估计出的属性模式** :math:`\\hat{\\alpha}`。于是分层集合变成
"知识状态完全相同的考生"，这比总分分层更严格、也更符合 CDM 的逻辑
（Hou, de la Torre & Nandakumar, 2014）。

统计量
------
对第 :math:`j` 题、第 :math:`s` 个属性模式层，记

.. math::

    a_s = \\#(\\text{参照组答对}),\\quad b_s = \\#(\\text{参照组答错}),\\\\
    c_s = \\#(\\text{焦点组答对}),\\quad d_s = \\#(\\text{焦点组答错})

Mantel-Haenszel 统计量（带连续性校正）::

    MH-χ² = (|Σa − ΣE[a]| − 0.5)² / ΣV[a]

其中
:math:`E[a_s] = m_{1s}n_{1s}/n_s`、
:math:`V[a_s] = m_{1s}n_{1s}(n_s-m_{1s})(n_s-n_{1s}) / (n_s^2(n_s-1))`，
:math:`m_{1s}` 为参照组人数、:math:`n_{1s}` 为答对人数。

共同比值比与效应量::

    α_MH = Σ(a_s d_s / n_s) / Σ(b_s c_s / n_s)
    Δ    = 2.35 ln(α_MH)

**符号约定**：``α_MH > 1`` 表示"同等知识状态下参照组答对的几率更高"，
因此 ``Δ > 0`` 即**有利于参照组**。这与 ETS 的 ``D-DIF = -2.35 ln α*``
完全等价——只不过 ETS 把 ``α*`` 定义成"焦点组/参照组"的比值，
本实现取倒数并把负号吸收进公式，使正负号直接可读。

ETS 分类（Dorrity & Hollingshead 的标准做法）：

===========  ==========================================
A（可忽略）  ``|Δ| < 1.0``
C（严重）    ``|Δ| ≥ 1.5`` 且 ``p < .05``
B（中等）    其余
===========  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..models import CDModel

__all__ = ["DIFItem", "DIFReport", "dif_analysis"]


# ---------------------------------------------------------------------- #
# 结果容器
# ---------------------------------------------------------------------- #
@dataclass
class DIFItem:
    """单题的 DIF 检验结果。"""

    item_idx: int
    item_name: str
    n_ref: int
    n_focal: int
    p_ref: float                      # 参照组原始答对率
    p_focal: float                    # 焦点组原始答对率
    n_strata: int                     # 有效分层数
    mh_chi2: float
    df: int
    p_value: float
    alpha_mh: float                   # 共同比值比（>1 ⇒ 参照组占优）
    delta_ets: float                  # Δ = 2.35·ln(α_MH)（正值＝利于参照组）
    ets_class: str                    # 'A' / 'B' / 'C'
    d_uniform: float = 0.0            # 未分层答对率差（描述用）

    @property
    def flagged(self) -> bool:
        """是否达到 B 级及以上（需人工复核）。"""
        return self.ets_class in ("B", "C")

    @property
    def favors(self) -> str:
        """DIF 有利于哪一组。``Δ>0`` 表示参照组占优。"""
        if not np.isfinite(self.delta_ets) or abs(self.delta_ets) < 1e-9:
            return "无"
        return "参照组" if self.delta_ets > 0 else "焦点组"

    def to_dict(self) -> dict:
        return {
            "item": self.item_name,
            "item_idx": self.item_idx,
            "n_ref": self.n_ref,
            "n_focal": self.n_focal,
            "p_ref": round(self.p_ref, 4),
            "p_focal": round(self.p_focal, 4),
            "d_uniform": round(self.d_uniform, 4),
            "n_strata": self.n_strata,
            "mh_chi2": round(self.mh_chi2, 4),
            "p_value": round(self.p_value, 6),
            "alpha_mh": round(self.alpha_mh, 4),
            "delta_ets": round(self.delta_ets, 4),
            "ets_class": self.ets_class,
            "favors": self.favors,
        }


@dataclass
class DIFReport:
    """整卷 DIF 检验结果。"""

    items: List[DIFItem]
    ref_label: str
    focal_label: str
    n_ref: int
    n_focal: int
    n_strata_used: int
    alpha: float = 0.05
    condition: str = "attribute_pattern"
    item_names: List[str] = field(default_factory=list)

    # ---- 汇总 ----
    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_flagged(self) -> int:
        return sum(1 for it in self.items if it.flagged)

    @property
    def n_C(self) -> int:
        return sum(1 for it in self.items if it.ets_class == "C")

    @property
    def n_B(self) -> int:
        return sum(1 for it in self.items if it.ets_class == "B")

    @property
    def n_A(self) -> int:
        return sum(1 for it in self.items if it.ets_class == "A")

    @property
    def n_significant(self) -> int:
        return sum(1 for it in self.items if it.p_value < self.alpha)

    def flagged(self, min_class: str = "B") -> List[DIFItem]:
        """返回达到指定等级及以上的题目。"""
        order = {"A": 0, "B": 1, "C": 2}
        floor = order.get(min_class.upper(), 1)
        return [it for it in self.items if order.get(it.ets_class, 0) >= floor]

    def largest(self, top_n: int = 10) -> List[DIFItem]:
        """按 |Δ_ETS| 降序返回效应最大的题目。"""
        return sorted(self.items, key=lambda it: -abs(it.delta_ets))[:top_n]

    # ---- 输出 ----
    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([it.to_dict() for it in self.items])

    def to_dict(self) -> dict:
        return {
            "ref_label": self.ref_label,
            "focal_label": self.focal_label,
            "n_ref": self.n_ref,
            "n_focal": self.n_focal,
            "n_strata_used": self.n_strata_used,
            "condition": self.condition,
            "alpha": self.alpha,
            "n_items": self.n_items,
            "n_A": self.n_A,
            "n_B": self.n_B,
            "n_C": self.n_C,
            "n_significant": self.n_significant,
            "items": [it.to_dict() for it in self.items],
        }

    def summary(self, top_n: int = 10) -> str:
        lines = [
            "=" * 68,
            "DIF 项目功能差异分析（MH，条件于估计属性模式）",
            "=" * 68,
            f"  参照组            : {self.ref_label}（n = {self.n_ref}）",
            f"  焦点组            : {self.focal_label}（n = {self.n_focal}）",
            f"  条件变量           : 估计属性模式（有效分层 {self.n_strata_used} 个）",
            f"  题目数            : {self.n_items}",
            f"  显著（p < {self.alpha}）  : {self.n_significant}",
            f"  ETS 分类          : A={self.n_A}  B={self.n_B}  C={self.n_C}",
        ]
        top = self.largest(top_n)
        if top:
            lines += ["", f"  效应量最大的 {len(top)} 题"]
            lines.append("  " + "-" * 60)
            lines.append(f"  {'题目':<8}{'Δ_ETS':>8}{'α_MH':>8}{'p':>9}  等级  有利")
            for it in top:
                p = "—" if not np.isfinite(it.p_value) else f"{it.p_value:.4f}"
                a = "—" if not np.isfinite(it.alpha_mh) else f"{it.alpha_mh:.3f}"
                lines.append(
                    f"  {it.item_name:<8}{it.delta_ets:>8.3f}{a:>8}{p:>9}"
                    f"   {it.ets_class:<4} {it.favors}"
                )
        if self.n_C:
            lines += ["", "  ⚠ C 级题目建议优先复核：可能是 Q 矩阵漏标、或群体间教学重点不同。"]
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 核心计算
# ---------------------------------------------------------------------- #
def _mh_for_item(score: np.ndarray, is_ref: np.ndarray, strata: np.ndarray,
                 min_stratum: int = 1) -> tuple:
    """单题的 MH 统计量；返回 (χ², α_MH, 有效层数)。"""
    sum_a = 0.0
    sum_e = 0.0
    sum_v = 0.0
    num = 0.0          # Σ a_s d_s / n_s
    den = 0.0          # Σ b_s c_s / n_s
    n_used = 0
    for s in np.unique(strata):
        m = strata == s
        ref_m = m & is_ref
        foc_m = m & ~is_ref
        n_ref_s = int(ref_m.sum())
        n_foc_s = int(foc_m.sum())
        # 分层内两组都太少的层不参与合并：这类层的比值比不稳定，且会放大方差
        if n_ref_s < min_stratum or n_foc_s < min_stratum:
            continue
        a = float(score[ref_m].sum())
        b = float(n_ref_s - a)
        c = float(score[foc_m].sum())
        d = float(n_foc_s - c)
        n_s = float(n_ref_s + n_foc_s)
        if n_s < 2:
            continue
        n_used += 1
        m1 = float(n_ref_s)          # 参照组人数
        n1 = a + c                   # 答对人数
        sum_a += a
        sum_e += m1 * n1 / n_s
        sum_v += m1 * n1 * (n_s - m1) * (n_s - n1) / (n_s * n_s * (n_s - 1.0))
        if n_s > 0:
            num += a * d / n_s
            den += b * c / n_s

    if sum_v <= 1e-12:
        return float("nan"), float("nan"), n_used
    chi2 = (abs(sum_a - sum_e) - 0.5) ** 2 / sum_v
    chi2 = max(chi2, 0.0)
    alpha_mh = num / den if den > 1e-12 else float("nan")
    return float(chi2), float(alpha_mh), n_used


def _chi2_sf(x: float, df: int = 1) -> float:
    """χ²(1) 上尾概率（闭式，避免 scipy 依赖）。"""
    if not np.isfinite(x) or x <= 0:
        return 1.0 if np.isfinite(x) else float("nan")
    # df=1 时 P(χ² > x) = erfc(sqrt(x/2))
    from math import erfc, sqrt

    return float(erfc(sqrt(x / 2.0)))


def _ets_class(delta: float, p: float, alpha: float = 0.05) -> str:
    """ETS Δ 分级。"""
    if not np.isfinite(delta):
        return "A"
    ad = abs(delta)
    if ad >= 1.5 and np.isfinite(p) and p < alpha:
        return "C"
    if ad >= 1.0:
        return "B"
    return "A"


def dif_analysis(
    model: CDModel,
    responses: np.ndarray,
    group_labels: Sequence,
    *,
    ref_group=None,
    alpha_hat: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    min_stratum: int = 1,
    alpha: float = 0.05,
    item_names: Optional[Sequence[str]] = None,
) -> DIFReport:
    """条件于估计属性模式的 MH-DIF 检验。

    Parameters
    ----------
    model : CDModel
        已标定的响应模型（仅用于推断属性模式）。
    responses : ndarray, shape (n_examinees, n_items)
        0/1 作答矩阵（可含 NaN，按缺失跳过）。
    group_labels : sequence
        每个被试的组别标签（长度 = 被试数）。
    ref_group : optional
        参照组标签；默认取出现次数最多的组。
    alpha_hat : ndarray, optional
        直接给定的属性模式 (n, K) 或模式索引 (n,)；不给则内部做 MAP 诊断。
    prior : ndarray, optional
        属性模式先验，用于 ``alpha_hat`` 的推断。
    min_stratum : int
        分层内两组各自最少人数，低于此值的层不参与合并（默认 1）。
    alpha : float
        显著性水平，用于 ETS C 级判定。
    item_names : sequence, optional

    Returns
    -------
    DIFReport

    Examples
    --------
    >>> rep = dif_analysis(model, X, group_labels=grade)   # doctest: +SKIP
    >>> rep.summary()                                      # doctest: +SKIP
    >>> rep.to_dataframe().query("ets_class != 'A'")       # doctest: +SKIP
    """
    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = resp.shape
    if j != model.n_items:
        raise ValueError(f"作答矩阵列数 ({j}) 与模型题目数 ({model.n_items}) 不一致")

    labels = np.asarray(list(group_labels), dtype=object)
    if labels.shape[0] != n:
        raise ValueError(
            f"分组标签长度 ({labels.shape[0]}) 与作答矩阵行数 ({n}) 不一致"
        )
    uniq = [g for g in dict.fromkeys(labels.tolist()) if g is not None]
    if len(uniq) < 2:
        raise ValueError("DIF 分析至少需要两个组，当前只有：" + "、".join(map(str, uniq)))

    if ref_group is None:
        counts = {g: int((labels == g).sum()) for g in uniq}
        ref_group = max(counts, key=counts.get)
    if ref_group not in uniq:
        raise ValueError(f"参照组 {ref_group!r} 不在分组标签中")

    # ---- 条件变量：估计属性模式 ----
    if alpha_hat is None:
        from . import diagnose_patterns as _diag

        diag = _diag(model, resp, prior=prior, method="MAP")
        strata = np.asarray(diag.pattern_indices, dtype=np.int64)
    else:
        pat = np.asarray(alpha_hat)
        if pat.ndim == 1:
            strata = pat.astype(np.int64)
        else:
            weights = 1 << np.arange(pat.shape[1] - 1, -1, -1)
            strata = (pat.astype(np.int64) * weights).sum(axis=1)

    is_ref = labels == ref_group
    other = [g for g in uniq if g != ref_group]
    # 焦点组若多于一个，逐个比较会膨胀多重检验；这里合并为一个"非参照组"对照，
    # 并在报告中通过 ref/focal 人数呈现。需要逐组比较时请分别调用。
    is_focal = ~is_ref

    n_ref = int(is_ref.sum())
    n_focal = int(is_focal.sum())
    if n_ref == 0 or n_focal == 0:
        raise ValueError("参照组或焦点组没有被试")

    names = list(item_names) if item_names else (
        list(model.q.item_names) if getattr(model.q, "item_names", None)
        else [f"I{i + 1}" for i in range(j)]
    )

    items: List[DIFItem] = []
    strata_used = 0
    for jj in range(j):
        col = resp[:, jj]
        obs = np.isfinite(col)
        score = np.where(obs, col, 0.0)
        if obs.sum() < 2 or (obs & is_ref).sum() == 0 or (obs & is_focal).sum() == 0:
            items.append(DIFItem(
                item_idx=jj, item_name=names[jj], n_ref=int((obs & is_ref).sum()),
                n_focal=int((obs & is_focal).sum()),
                p_ref=float("nan"), p_focal=float("nan"), n_strata=0,
                mh_chi2=float("nan"), df=1, p_value=float("nan"),
                alpha_mh=float("nan"), delta_ets=float("nan"), ets_class="A",
            ))
            continue

        chi2, alpha_mh, used = _mh_for_item(
            score[obs], is_ref[obs], strata[obs], min_stratum=min_stratum
        )
        strata_used = max(strata_used, used)
        p_val = _chi2_sf(chi2, 1)
        # Δ > 0 ⇒ 参照组占优（见模块文档的符号约定）
        delta = 2.35 * np.log(alpha_mh) if (np.isfinite(alpha_mh) and alpha_mh > 0) else float("nan")

        p_ref = float(score[obs & is_ref].mean())
        p_focal = float(score[obs & is_focal].mean())

        items.append(DIFItem(
            item_idx=jj, item_name=names[jj],
            n_ref=int((obs & is_ref).sum()), n_focal=int((obs & is_focal).sum()),
            p_ref=p_ref, p_focal=p_focal, n_strata=used,
            mh_chi2=chi2, df=1, p_value=p_val,
            alpha_mh=alpha_mh, delta_ets=float(delta),
            ets_class=_ets_class(delta, p_val, alpha),
            d_uniform=p_ref - p_focal,
        ))

    return DIFReport(
        items=items,
        ref_label=str(ref_group),
        focal_label=" + ".join(str(g) for g in other),
        n_ref=n_ref,
        n_focal=n_focal,
        n_strata_used=int(len(np.unique(strata))),
        alpha=alpha,
        item_names=names,
    )
