"""在线标定：给新进题库的题目估参数，不必重新组织一次完整标定。

场景
----
题库要扩充，新写了 20 道题。为这 20 道题单独做一次标定需要一个几百人的
专项样本；但实际考试里它们已经跟着老题一起考过了。既然老题已经标定好、
考生的知识状态也能诊出来，就可以"拿考生的诊断结果当已知量"去反推新题的
slip / guess——这就是在线标定（online calibration）。

做法
----
对新题 :math:`j`，把考生的属性后验 :math:`p(\\alpha\\mid X^{\\text{old}}_i)`
当作权重，分别统计"该题应该会做的人"和"该题不该会做的人"：

.. math::

    n_{1j} = \\sum_i \\sum_s w_{is}\\,\\eta_{sj}, \\qquad
    r_{1j} = \\sum_i \\sum_s w_{is}\\,\\eta_{sj}\\,x_{ij}

    \\hat{\\sigma}_j = 1 - \\frac{r_{1j}}{n_{1j}}, \\qquad
    \\hat{g}_j = \\frac{r_{0j}}{n_{0j}}

其中 :math:`\\eta_{sj}` 是理想作答（该模式是否具备作答该题所需的全部属性），
:math:`n_{0j} = N - n_{1j}`、:math:`r_{0j} = \\sum_i x_{ij} - r_{1j}`。
这等价于在"题目参数自由、属性分布固定"下做一步 MMLE。

为什么加 Beta 平滑
------------------
新题常常只被几十个人做过，"应该会做的人"可能只有个位数；直接算比例会得到
0 或 1 这种荒谬的 slip。用 Beta 先验做收缩：:math:`\\hat\\sigma \\sim
\\mathrm{Beta}(a + n_1 - r_1,\\; b + r_1)`，样本少时向后验均值（默认
Beta(1,4)，均值 0.2）收缩，样本多时由数据主导。标准误与置信区间直接取该
Beta 后验的解析解，不需要 bootstrap。

什么时候结果不能用
------------------
- 该题在样本里"应会做"的人太少（:math:`n_1` 小于阈值）→ slip 不可信；
- 该题"不该会做"的人太少（:math:`n_0` 小）→ guess 不可信；
- 更根本的：**如果考生整体都没掌握这些属性，无论多少人都标不出 slip**。
  所以报告里同时给 ``n_masters``，这个数字比样本量更能说明问题。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..models import CDModel

__all__ = ["NewItemResult", "OnlineCalibrationReport", "calibrate_new_items"]

_EPS = 1e-10


def _beta_summary(a: float, b: float, level: float) -> Tuple[float, float, float]:
    """Beta 后验的均值、标准误、双侧区间。"""
    total = a + b
    mean = a / total if total > 0 else float("nan")
    var = (a * b) / (total * total * (total + 1.0)) if total > 0 else float("nan")
    se = float(np.sqrt(max(var, 0.0)))
    try:
        from scipy import stats

        lo, hi = stats.beta.ppf([(1 - level) / 2, 1 - (1 - level) / 2], a, b)
        ci = (float(lo), float(hi))
    except Exception:                                     # scipy 不可用时只给均值
        ci = (float("nan"), float("nan"))
    return float(mean), se, ci


# ---------------------------------------------------------------------- #
# 结果
# ---------------------------------------------------------------------- #
@dataclass
class NewItemResult:
    """单道新题的在线标定结果。"""

    item_idx: int
    item_name: str
    n_masters: float                 # 期望"应会做"人数
    n_nonmasters: float              # 期望"不应会做"人数
    n_observed: int                  # 实际作答人数（非缺失）
    slip: float
    guess: float
    slip_se: float
    guess_se: float
    slip_ci: Tuple[float, float]
    guess_ci: Tuple[float, float]
    status: str = "ok"               # ok / unstable / unusable
    notes: List[str] = field(default_factory=list)

    @property
    def discrimination(self) -> float:
        return float(1.0 - self.slip - self.guess)

    @property
    def usable(self) -> bool:
        return self.status != "unusable"

    def to_dict(self) -> dict:
        return {
            "item_idx": int(self.item_idx),
            "item": self.item_name,
            "n_masters": round(float(self.n_masters), 2),
            "n_nonmasters": round(float(self.n_nonmasters), 2),
            "n_observed": int(self.n_observed),
            "slip": round(float(self.slip), 4),
            "guess": round(float(self.guess), 4),
            "slip_se": round(float(self.slip_se), 4),
            "guess_se": round(float(self.guess_se), 4),
            "slip_ci": [round(float(v), 4) for v in self.slip_ci],
            "guess_ci": [round(float(v), 4) for v in self.guess_ci],
            "discrimination": round(self.discrimination, 4),
            "status": self.status,
            "notes": list(self.notes),
        }


@dataclass
class OnlineCalibrationReport:
    """一批新题的在线标定结果。"""

    items: List[NewItemResult]
    n_examinees: int
    n_old_items: int
    n_new_items: int
    beta_prior: Tuple[float, float]
    min_effective_n: float
    ci_level: float
    method: str = "posterior_weighted_mmle"
    attr_names: List[str] = field(default_factory=list)

    # ---- 查询 ----
    @property
    def usable_items(self) -> List[NewItemResult]:
        return [it for it in self.items if it.usable]

    @property
    def flagged(self) -> List[NewItemResult]:
        return [it for it in self.items if it.status != "ok"]

    def get(self, item_idx: int) -> NewItemResult:
        for it in self.items:
            if it.item_idx == item_idx:
                return it
        raise KeyError(f"没有索引为 {item_idx} 的新题")

    # ---- 汇总 ----
    def verdict(self) -> str:
        bad = self.flagged
        if not bad:
            return f"{len(self.items)} 道新题全部完成标定，样本量充足。"
        unusable = [it for it in bad if it.status == "unusable"]
        unstable = [it for it in bad if it.status == "unstable"]
        parts = []
        if unstable:
            parts.append(f"{len(unstable)} 道样本不足（估计不稳定）")
        if unusable:
            parts.append(f"{len(unusable)} 道无法标定")
        return "、".join(parts) + "，详见 notes。建议补测或改用专家赋值。"

    # ---- 导出 ----
    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([it.to_dict() for it in self.items])

    def to_dict(self) -> dict:
        return {
            "n_examinees": self.n_examinees,
            "n_old_items": self.n_old_items,
            "n_new_items": self.n_new_items,
            "method": self.method,
            "beta_prior": list(self.beta_prior),
            "min_effective_n": self.min_effective_n,
            "ci_level": self.ci_level,
            "verdict": self.verdict(),
            "n_flagged": len(self.flagged),
            "items": [it.to_dict() for it in self.items],
        }

    def summary(self) -> str:
        lines = [
            "=" * 76,
            "新题在线标定",
            "=" * 76,
            f"  被试数              : {self.n_examinees}",
            f"  老题（用于诊断）/ 新题: {self.n_old_items} / {self.n_new_items}",
            f"  方法                : {self.method}",
            f"  Beta 先验           : Beta{tuple(round(v, 2) for v in self.beta_prior)}",
            f"  样本充足阈值        : n_eff ≥ {self.min_effective_n:g}",
            "",
            f"  {'题目':<10}{'应会':>8}{'不应会':>8}{'slip':>9}{'guess':>9}"
            f"{'区分度':>8}  状态",
            "  " + "-" * 72,
        ]
        for it in self.items:
            lines.append(
                f"  {it.item_name:<10}{it.n_masters:>8.1f}{it.n_nonmasters:>8.1f}"
                f"{it.slip:>9.3f}{it.guess:>9.3f}{it.discrimination:>8.3f}"
                f"  {'正常' if it.status == 'ok' else ('不稳定' if it.status == 'unstable' else '不可用')}"
            )
        notes = [it for it in self.items if it.notes]
        if notes:
            lines += ["", "  说明:"]
            for it in notes:
                lines.append(f"    {it.item_name}: {'；'.join(it.notes)}")
        lines += ["", "  " + self.verdict()]
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 主函数
# ---------------------------------------------------------------------- #
def _posterior(model: CDModel, resp: np.ndarray, items: np.ndarray,
               prior: Optional[np.ndarray]) -> np.ndarray:
    """只用指定题目计算属性模式后验，形状 ``(n, 2^K)``。"""
    prob = np.asarray(model._get_prob(), dtype=np.float64)[:, items]
    sub = resp[:, items]
    obs = np.isfinite(sub)
    x = np.where(obs, sub, 0.0)
    ll = x @ np.log(np.clip(prob, _EPS, 1.0)).T \
        + (obs - x) @ np.log(np.clip(1.0 - prob, _EPS, 1.0)).T
    s_total = prob.shape[0]
    if prior is None:
        pri = np.full(s_total, 1.0 / s_total)
    else:
        pri = np.clip(np.asarray(prior, dtype=np.float64).ravel(), 1e-300, None)
        if pri.size != s_total:
            raise ValueError(f"prior 长度应为 {s_total}")
        pri = pri / pri.sum()
    ll = ll + np.log(pri)
    ll -= ll.max(axis=1, keepdims=True)
    post = np.exp(ll)
    return post / post.sum(axis=1, keepdims=True)


def calibrate_new_items(
    model: CDModel,
    responses: np.ndarray,
    new_items: Sequence[int],
    *,
    old_items: Optional[Sequence[int]] = None,
    prior: Optional[np.ndarray] = None,
    alpha_hat: Optional[np.ndarray] = None,
    slip_prior: Tuple[float, float] = (1.0, 4.0),
    guess_prior: Tuple[float, float] = (1.0, 4.0),
    min_effective_n: float = 10.0,
    ci_level: float = 0.95,
    item_names: Optional[Sequence[str]] = None,
) -> OnlineCalibrationReport:
    """用已诊断的考生给新题标定 slip / guess。

    Parameters
    ----------
    model : CDModel
        已标定好的模型。其 Q 矩阵须覆盖**全部**题目（含新题），但新题位置的
        参数可以是占位值——本函数不会用到它们。
    responses : ndarray, shape (n_examinees, n_items)
        全部题目的 0/1 作答矩阵，允许 ``np.nan``。
    new_items : sequence of int
        要标定的新题索引（相对 ``responses`` 的列）。
    old_items : sequence of int, optional
        用于诊断考生知识状态的老题索引，默认取"除新题外的全部题目"。
    prior : ndarray, optional
        属性模式先验（长度 ``2^K``），默认均匀。
    alpha_hat : ndarray, optional
        直接给定属性后验 ``(n, 2^K)``（每行和为 1），跳过内部诊断。
    slip_prior, guess_prior : (float, float)
        Beta 先验的 ``(a, b)``，默认 ``(1, 4)`` 即均值 0.2 的弱先验。
        样本充足时几乎不影响结果；样本少时防止出现 0 / 1 的极端估计。
    min_effective_n : float
        判别"样本充足"的阈值，作用于 ``min(n_masters, n_nonmasters)``。
    ci_level : float
        置信水平。
    item_names : sequence of str, optional
        仅用于展示。

    Returns
    -------
    OnlineCalibrationReport

    Examples
    --------
    >>> rep = calibrate_new_items(model, X, new_items=range(30, 40))  # doctest: +SKIP
    >>> rep.to_dataframe()                                           # doctest: +SKIP
    >>> rep.flagged                                                  # doctest: +SKIP
    """
    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j_total = resp.shape
    if j_total != model.n_items:
        raise ValueError(
            f"作答矩阵列数 ({j_total}) 与模型题目数 ({model.n_items}) 不一致"
        )

    new_idx = np.array(sorted({int(i) for i in new_items}), dtype=int)
    if new_idx.size == 0:
        raise ValueError("new_items 不能为空")
    if new_idx.min() < 0 or new_idx.max() >= j_total:
        raise ValueError(f"new_items 超出范围 [0, {j_total})")

    if old_items is None:
        mask = np.ones(j_total, dtype=bool)
        mask[new_idx] = False
        old_idx = np.where(mask)[0]
    else:
        old_idx = np.array(sorted({int(i) for i in old_items}), dtype=int)
        if old_idx.size == 0:
            raise ValueError("old_items 不能为空")
        if np.intersect1d(old_idx, new_idx).size:
            raise ValueError("old_items 与 new_items 不能重叠")
        if old_idx.min() < 0 or old_idx.max() >= j_total:
            raise ValueError(f"old_items 超出范围 [0, {j_total})")

    # ---- 考生知识状态 ----
    if alpha_hat is not None:
        post = np.asarray(alpha_hat, dtype=np.float64)
        if post.ndim != 2 or post.shape[0] != n:
            raise ValueError(f"alpha_hat 形状应为 ({n}, 2^K)")
        post = np.clip(post, 0.0, None)
        rs = post.sum(axis=1, keepdims=True)
        post = np.divide(post, np.where(rs > 0, rs, 1.0))
    else:
        post = _posterior(model, resp, old_idx, prior)

    # ---- 理想作答（该模式是否具备作答该题的全部属性）----
    mode = getattr(model, "ideal_mode", "conjunctive")
    eta_all = np.asarray(model.q.ideal_response(mode), dtype=np.float64)   # (S, J)

    names = list(item_names) if item_names else []
    results: List[NewItemResult] = []
    for j in new_idx:
        j = int(j)
        col = resp[:, j]
        valid = np.isfinite(col)
        x = np.where(valid, col, 0.0)

        eta = eta_all[:, j]                                    # (S,)
        w1 = post @ eta                                        # (n,) 期望"会做"
        n1 = float(w1[valid].sum())
        r1 = float((w1[valid] * x[valid]).sum())
        n0 = float(valid.sum()) - n1
        r0 = float(x[valid].sum()) - r1

        name = names[j] if j < len(names) else f"I{j + 1}"
        notes: List[str] = []
        item = NewItemResult(
            item_idx=j, item_name=name,
            n_masters=n1, n_nonmasters=n0, n_observed=int(valid.sum()),
            slip=float("nan"), guess=float("nan"),
            slip_se=float("nan"), guess_se=float("nan"),
            slip_ci=(float("nan"), float("nan")),
            guess_ci=(float("nan"), float("nan")),
        )

        # ---- slip：Beta(a + 错答数, b + 答对数) ----
        if n1 <= 0:
            item.status = "unusable"
            notes.append("样本中没有考生被诊断为掌握该题所需属性，slip 无法估计")
        else:
            a = slip_prior[0] + (n1 - r1)
            b = slip_prior[1] + r1
            item.slip, item.slip_se, item.slip_ci = _beta_summary(a, b, ci_level)
            if n1 < min_effective_n:
                notes.append(f"应会做人数仅 {n1:.1f}，slip 估计不稳定")

        # ---- guess：Beta(a + 误答对数, b + 答错数) ----
        if n0 <= 0:
            item.status = "unusable"
            notes.append("样本中所有考生都具备该题所需属性，guess 无法估计")
        else:
            a = guess_prior[0] + r0
            b = guess_prior[1] + (n0 - r0)
            item.guess, item.guess_se, item.guess_ci = _beta_summary(a, b, ci_level)
            if n0 < min_effective_n:
                notes.append(f"不应会做人数仅 {n0:.1f}，guess 估计不稳定")

        if item.status != "unusable" and notes:
            item.status = "unstable"
        if item.status != "unusable" and np.isfinite(item.slip) and np.isfinite(item.guess):
            if item.slip + item.guess > 0.95:
                item.status = "unstable"
                notes.append("slip+guess>0.95，该题几乎不提供区分信息")
        item.notes = notes
        results.append(item)

    if not results:
        raise ValueError("没有可标定的新题")

    return OnlineCalibrationReport(
        items=results,
        n_examinees=n,
        n_old_items=int(old_idx.size),
        n_new_items=int(new_idx.size),
        beta_prior=tuple(slip_prior),          # 报告里以 slip 先验为代表
        min_effective_n=float(min_effective_n),
        ci_level=float(ci_level),
        attr_names=list(getattr(model.q, "attr_names", None) or []),
    )
