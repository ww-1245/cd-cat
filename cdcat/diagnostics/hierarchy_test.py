"""属性层级结构的自动检验与比较。

为什么需要它
------------
属性层级（AHM）是认知诊断里"用先决关系约束知识状态"的常见做法：
若 A1 是 A2 的先备，那么"A1 未掌握但 A2 已掌握"这种模式在现实中不该出现。
把它写进模型的好处是参数更少、分类更稳；坏处是**如果层级本身错了，
约束会把被试强行套进错误的模式**，分类准确率反而下降。

所以层级不能只靠学科专家拍板，还要让数据说话。本模块把"哪种层级结构更被
数据支持"变成一个可比较的模型选择问题：

对每个候选层级 :math:`h`，把属性模式的先验限制在合法模式集 :math:`\\mathcal{A}_h` 上
重新标定，得到边际对数似然 :math:`\\ln L_h`，再用信息准则比较：

.. math::

    \\mathrm{BIC}_h = -2\\ln L_h + (n_{\\text{item}} + |\\mathcal{A}_h| - 1)\\ln N

同时报告两个**直接的违背指标**（不依赖信息准则，更直观）：

``illegal_ratio``
    在无约束模型下按 MAP 分类，被判为"非法模式"的被试比例。
``illegal_mass``
    每名被试后验落在非法模式上的概率的平均值（比 MAP 更平滑，不受临界跳变影响）。

为什么用 BIC 而不是似然比
------------------------
受限模型位于参数空间的**边界**上（非法模式的先验被钉在 0），经典的 χ² 近似
在这种情形下偏保守，p 值会系统性偏大。所以这里以 BIC 为主判据，似然比检验
只作为参考（``lr_p`` 列），并已注明其保守性。

**注意**：BIC 比较的是"哪个层级约束更合理"，不是"层级是否存在"。若
``none`` 排第一，含义是"数据没有提供支持任何层级约束的证据"，这本身就是一个
有意义的结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..calibration import CalibrationResult, calibrate
from ..models import CDModel
from ..qmatrix import AttributeHierarchy, QMatrix

__all__ = [
    "HierarchyCandidate",
    "HierarchyComparisonReport",
    "compare_hierarchies",
    "DEFAULT_STRUCTURES",
]

#: 默认比较的结构类型
DEFAULT_STRUCTURES = ("none", "linear", "convergent", "divergent")


def _chi2_sf(x: float, df: int) -> float:
    """χ² 上尾概率。"""
    if df <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    if df == 1:
        from math import erfc, sqrt

        return float(erfc(sqrt(x / 2.0)))
    from scipy import stats

    return float(stats.chi2.sf(x, df))


def _build_default(n_attrs: int, attr_names: Optional[Sequence[str]],
                   structures: Sequence[str]) -> Dict[str, AttributeHierarchy]:
    makers = {
        "none": AttributeHierarchy.none,
        "linear": AttributeHierarchy.linear,
        "convergent": AttributeHierarchy.convergent,
        "divergent": AttributeHierarchy.divergent,
    }
    out: Dict[str, AttributeHierarchy] = {}
    for key in structures:
        key = str(key).strip().lower()
        if key not in makers:
            raise ValueError(f"未知结构类型 '{key}'，可选 {sorted(makers)}")
        if n_attrs < 2 and key != "none":
            continue          # 单属性没有层级可言
        out[key] = makers[key](n_attrs, attr_names)
    if "none" not in out:
        out = {"none": AttributeHierarchy.none(n_attrs, attr_names), **out}
    return out


# ---------------------------------------------------------------------- #
# 结果
# ---------------------------------------------------------------------- #
@dataclass
class HierarchyCandidate:
    """一个候选层级结构的标定与比较结果。"""

    label: str
    hierarchy: AttributeHierarchy
    n_legal: int
    n_params: int
    loglik: float
    aic: float
    bic: float
    aicc: float
    delta_bic: float = float("nan")
    bic_weight: float = float("nan")
    illegal_ratio: float = float("nan")
    illegal_mass: float = float("nan")
    lr_chi2: float = float("nan")
    lr_df: int = 0
    lr_p: float = float("nan")
    converged: bool = True
    error: Optional[str] = None
    model: Optional[CDModel] = field(default=None, repr=False)
    result: Optional[CalibrationResult] = field(default=None, repr=False)

    @property
    def structure(self) -> str:
        return self.hierarchy.structure

    @property
    def is_best(self) -> bool:
        return np.isfinite(self.delta_bic) and self.delta_bic <= 1e-9

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "structure": self.structure,
            "edges": [list(e) for e in self.hierarchy.edges],
            "n_legal": self.n_legal,
            "n_params": self.n_params,
            "loglik": round(float(self.loglik), 4),
            "aic": round(float(self.aic), 4),
            "bic": round(float(self.bic), 4),
            "aicc": round(float(self.aicc), 4),
            "delta_bic": (None if not np.isfinite(self.delta_bic)
                          else round(float(self.delta_bic), 4)),
            "bic_weight": (None if not np.isfinite(self.bic_weight)
                           else round(float(self.bic_weight), 4)),
            "illegal_ratio": (None if not np.isfinite(self.illegal_ratio)
                              else round(float(self.illegal_ratio), 4)),
            "illegal_mass": (None if not np.isfinite(self.illegal_mass)
                             else round(float(self.illegal_mass), 4)),
            "lr_chi2": (None if not np.isfinite(self.lr_chi2)
                        else round(float(self.lr_chi2), 4)),
            "lr_df": int(self.lr_df),
            "lr_p": (None if not np.isfinite(self.lr_p)
                     else float(f"{self.lr_p:.4g}")),
            "converged": bool(self.converged),
            "error": self.error,
        }


@dataclass
class HierarchyComparisonReport:
    """多个候选层级的横向比较。"""

    candidates: List[HierarchyCandidate]
    n_examinees: int
    n_items: int
    n_attrs: int
    model_type: str
    attr_names: List[str] = field(default_factory=list)

    # ---- 排序与查询 ----
    @property
    def ranking(self) -> List[HierarchyCandidate]:
        """按 BIC 升序（失败项排在最后）。"""
        ok = [c for c in self.candidates if np.isfinite(c.bic)]
        bad = [c for c in self.candidates if not np.isfinite(c.bic)]
        return sorted(ok, key=lambda c: c.bic) + bad

    @property
    def best(self) -> HierarchyCandidate:
        return self.ranking[0]

    @property
    def best_label(self) -> str:
        return self.best.label

    def get(self, label: str) -> HierarchyCandidate:
        for c in self.candidates:
            if c.label == label:
                return c
        raise KeyError(f"没有名为 '{label}' 的候选层级")

    # ---- 判读 ----
    def verdict(self) -> str:
        best = self.best
        if best.label == "none":
            return ("数据不支持任何层级约束：无约束模型（none）的 BIC 最低，"
                    "建议按 2^K 个模式自由估计。")
        second = self.ranking[1] if len(self.ranking) > 1 else None
        margin = (second.bic - best.bic) if second is not None else float("nan")
        tail = ""
        if np.isfinite(margin):
            if margin < 2:
                tail = f"，但对次优（{second.label}）的优势仅 ΔBIC={margin:.1f}，证据较弱"
            elif margin < 6:
                tail = f"，优于次优（{second.label}）ΔBIC={margin:.1f}，证据中等"
            else:
                tail = f"，明显优于次优（{second.label}）ΔBIC={margin:.1f}"
        ratio = best.illegal_ratio
        ratio_txt = ""
        if np.isfinite(ratio):
            ratio_txt = f"；该层级下被判为非法的被试占 {ratio:.1%}"
        return f"推荐层级结构：{best.label}（合法模式 {best.n_legal} 个）{tail}{ratio_txt}。"

    # ---- 输出 ----
    def to_dataframe(self):
        import pandas as pd

        rows = [c.to_dict() for c in self.ranking]
        df = pd.DataFrame(rows)
        keep = ["label", "structure", "n_legal", "n_params", "loglik",
                "aic", "bic", "delta_bic", "bic_weight",
                "illegal_ratio", "illegal_mass", "lr_p", "converged"]
        return df[[k for k in keep if k in df.columns]]

    def to_dict(self) -> dict:
        return {
            "n_examinees": self.n_examinees,
            "n_items": self.n_items,
            "n_attrs": self.n_attrs,
            "model_type": self.model_type,
            "attr_names": list(self.attr_names),
            "best": self.best_label,
            "verdict": self.verdict(),
            "candidates": [c.to_dict() for c in self.ranking],
        }

    def summary(self) -> str:
        head = [
            "=" * 74,
            "属性层级结构比较",
            "=" * 74,
            f"  被试数 / 题目数 / 属性数 : {self.n_examinees} / {self.n_items} / {self.n_attrs}",
            f"  响应模型                  : {self.model_type}",
            "",
            f"  {'层级':<12}{'合法模式':>9}{'参数':>7}{'-2LL':>12}"
            f"{'BIC':>12}{'ΔBIC':>8}{'权重':>7}{'非法%':>8}",
            "  " + "-" * 70,
        ]
        lines = list(head)
        for c in self.ranking:
            if not np.isfinite(c.bic):
                lines.append(f"  {c.label:<12}{'—':>9}{'—':>7}{'—':>12}"
                             f"{'—':>12}{'—':>8}{'—':>7}{'—':>8}  {c.error or '失败'}")
                continue
            lines.append(
                f"  {c.label:<12}{c.n_legal:>9}{c.n_params:>7}"
                f"{-2.0 * c.loglik:>12.2f}{c.bic:>12.2f}"
                f"{c.delta_bic:>8.2f}{c.bic_weight:>7.3f}"
                f"{(c.illegal_ratio if np.isfinite(c.illegal_ratio) else 0.0) * 100:>7.1f}%"
            )
        lines += [
            "",
            "  " + self.verdict(),
            "  提示：ΔBIC < 2 视为证据微弱，2~6 中等，> 6 明显；权重为 BIC 后验概率。",
            "  非法% = 无约束模型下 MAP 分类落在非法模式的被试比例，越低说明层级越贴合数据。",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 主函数
# ---------------------------------------------------------------------- #
def compare_hierarchies(
    q_matrix,
    responses: np.ndarray,
    model_type: str = "DINA",
    *,
    candidates: Optional[Dict[str, AttributeHierarchy]] = None,
    structures: Sequence[str] = DEFAULT_STRUCTURES,
    attr_names: Optional[Sequence[str]] = None,
    estimate_prior: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> HierarchyComparisonReport:
    """比较多个候选属性层级，按 BIC 推荐。

    Parameters
    ----------
    q_matrix : QMatrix or array-like
    responses : ndarray, shape (n_examinees, n_items)
        0/1 作答矩阵。
    model_type : str
        响应模型，默认 ``DINA``。
    candidates : dict, optional
        自定义候选，``{标签: AttributeHierarchy}``。给定时**取代** ``structures``。
    structures : sequence of str
        要自动生成的默认结构，可选 ``none`` / ``linear`` / ``convergent`` / ``divergent``。
    attr_names : sequence of str, optional
    estimate_prior : bool
        是否在合法模式集内继续做经验贝叶斯先验估计（默认是，非法模式的先验
        恒为 0，不会被"复活"）。
    max_iter, tol : 传给 EM。
    verbose : bool

    Returns
    -------
    HierarchyComparisonReport

    Examples
    --------
    >>> rep = compare_hierarchies(q, X)          # doctest: +SKIP
    >>> rep.best_label                           # doctest: +SKIP
    'linear'
    >>> print(rep.summary())                     # doctest: +SKIP
    """
    q = q_matrix if isinstance(q_matrix, QMatrix) else QMatrix(q_matrix)
    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    if resp.shape[1] != q.n_items:
        raise ValueError(
            f"作答矩阵列数 ({resp.shape[1]}) 与题目数 ({q.n_items}) 不一致"
        )
    n, n_attrs = resp.shape[0], q.n_attrs

    if candidates is None:
        pool = _build_default(n_attrs, attr_names, structures)
    else:
        pool = dict(candidates)
        if not pool:
            raise ValueError("candidates 不能为空")
        for label, h in pool.items():
            if not isinstance(h, AttributeHierarchy):
                raise TypeError(f"候选 '{label}' 不是 AttributeHierarchy")
            if h.n_attrs != n_attrs:
                raise ValueError(
                    f"候选 '{label}' 的属性数 ({h.n_attrs}) 与 Q 矩阵 ({n_attrs}) 不一致"
                )
    if len(pool) < 2:
        raise ValueError("至少需要两个候选层级才能比较")

    # ---- 无约束模型的 MAP，用来算"非法模式"指标 ----
    base_label = next((k for k in pool if pool[k].is_empty), None)
    illegal_ratio: Dict[str, float] = {}
    illegal_mass: Dict[str, float] = {}
    for label, h in pool.items():
        if h.is_empty:
            illegal_ratio[label] = 0.0
            illegal_mass[label] = 0.0

    out: List[HierarchyCandidate] = []
    ref: Optional[HierarchyCandidate] = None

    for label, h in pool.items():
        legal = np.asarray(h.legal_mask(), dtype=bool)
        prior = legal.astype(np.float64)
        prior = prior / prior.sum()
        try:
            model, res = calibrate(
                q, resp, model_type=model_type,
                prior=prior, estimate_prior=estimate_prior,
                lock_prior_zeros=True,
                max_iter=max_iter, tol=tol,
            )
        except Exception as exc:                       # 候选不成立时不让整份报告失败
            out.append(HierarchyCandidate(
                label=label, hierarchy=h, n_legal=int(legal.sum()),
                n_params=0, loglik=float("nan"), aic=float("nan"),
                bic=float("nan"), aicc=float("nan"), error=str(exc),
            ))
            continue

        cand = HierarchyCandidate(
            label=label,
            hierarchy=h,
            n_legal=int(legal.sum()),
            n_params=int(getattr(res, "n_params", 0)),
            loglik=float(res.log_likelihood),
            aic=float(res.aic),
            bic=float(res.bic),
            aicc=float(res.aicc),
            converged=bool(getattr(res, "converged", True)),
            model=model,
            result=res,
        )
        out.append(cand)
        if h.is_empty:
            ref = cand

        # 非法模式比例（用无约束模型的后验判定）
        if ref is not None and not h.is_empty:
            post = _posterior_matrix(ref.model, resp, ref.result)
            bad = ~legal
            mass = post[:, bad].sum(axis=1)
            illegal_mass[label] = float(mass.mean())
            map_idx = post.argmax(axis=1)
            illegal_ratio[label] = float(bad[map_idx].mean())

    # 若自定义候选里没有"无约束"，用第一个候选作基准算非法比例
    if ref is None:
        first = next((c for c in out if c.model is not None), None)
        if first is not None:
            post = _posterior_matrix(first.model, resp, first.result)
            for c in out:
                if c.hierarchy.is_empty:
                    continue
                bad = ~np.asarray(c.hierarchy.legal_mask(), dtype=bool)
                mass = post[:, bad].sum(axis=1)
                illegal_mass[c.label] = float(mass.mean())
                illegal_ratio[c.label] = float(bad[post.argmax(axis=1)].mean())

    for c in out:
        c.illegal_ratio = illegal_ratio.get(c.label, float("nan"))
        c.illegal_mass = illegal_mass.get(c.label, float("nan"))

    # ---- 似然比（相对无约束模型）----
    if ref is not None and np.isfinite(ref.loglik):
        for c in out:
            if c is ref or not np.isfinite(c.loglik):
                continue
            c.lr_chi2 = max(2.0 * (ref.loglik - c.loglik), 0.0)
            c.lr_df = max(ref.n_legal - c.n_legal, 0)
            c.lr_p = _chi2_sf(c.lr_chi2, c.lr_df)

    # ---- BIC 差值与权重 ----
    finite = [c for c in out if np.isfinite(c.bic)]
    if finite:
        best_bic = min(c.bic for c in finite)
        for c in finite:
            c.delta_bic = c.bic - best_bic
        w = np.array([np.exp(-0.5 * c.delta_bic) for c in finite])
        w = w / w.sum()
        for c, wi in zip(finite, w):
            c.bic_weight = float(wi)

    if verbose:
        print(f"比较 {len(out)} 个候选层级……完成")

    return HierarchyComparisonReport(
        candidates=out,
        n_examinees=n,
        n_items=q.n_items,
        n_attrs=n_attrs,
        model_type=str(model_type),
        attr_names=list(attr_names) if attr_names else [],
    )


def _posterior_matrix(model: CDModel, resp: np.ndarray, res: CalibrationResult) -> np.ndarray:
    """用标定得到的属性先验计算每名被试的模式后验，形状 ``(n, 2^K)``。"""
    prob = np.asarray(model._get_prob(), dtype=np.float64)
    obs = np.isfinite(resp)
    x = np.where(obs, resp, 0.0)
    ll = (x @ np.log(prob).T + (obs - x) @ np.log1p(-prob).T)
    prior = getattr(res, "attr_prior", None)
    if prior is None:
        prior = np.full(prob.shape[0], 1.0 / prob.shape[0])
    prior = np.clip(np.asarray(prior, dtype=np.float64).ravel(), 1e-300, None)
    ll = ll + np.log(prior)
    ll -= ll.max(axis=1, keepdims=True)
    post = np.exp(ll)
    return post / post.sum(axis=1, keepdims=True)
