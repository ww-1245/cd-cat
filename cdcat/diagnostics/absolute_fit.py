"""绝对拟合检验与局部依赖分析。

相对拟合指数（AIC / BIC / CFI / RMSEA）只能回答"哪个模型更好"，
无法回答"这个模型本身够不够好"。本模块补上三件审稿人必问的证据：

1. **题目级绝对拟合**（:class:`ItemFitResult`）
   把被试按"MAP 属性估计 × 剩余得分分箱"划分为若干单元，比较每道题在
   各单元内的**观测答对率**与**模型预测答对率**，给出 Pearson ``χ²``、
   似然比 ``G²``、自由度与 p 值，以及不受样本量影响的效应量
   ``max_abs_residual``（观测与预测答对率的最大绝对差）。

2. **局部依赖 Q3**（:class:`LocalDependenceResult`）
   Yen (1984) 的 Q3 统计量：模型若正确，控制住属性后题目之间不应再有
   相关。Q3 即**残差相关**，显著偏高的题目对说明存在局部依赖，通常意味着
   Q 矩阵漏写了共同属性。

3. **残差结构**（:class:`AbsoluteFitResult`）
   按潜在组汇总观测/预测答对率，定位系统性偏差集中在哪些题、哪些组。

重要说明
--------
- 题目参数是用**同一批数据**估计的，因此 ``χ²`` / ``G²`` 的 p 值只是
  **近似**参考（真实分布比名义分布更保守）。判读时请优先看
  ``max_abs_residual`` 这类效应量，p 值作为辅助。
- Q3 的阈值 0.2 来自 Yen (1984) 的经验建议，仅作筛查起点；
  题目数很多时天然会有较大的 Q3 抽样波动，应结合题目内容判断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..models import CDModel

__all__ = [
    "ItemFitResult",
    "item_fit",
    "LocalDependenceResult",
    "local_dependence",
    "AbsoluteFitResult",
    "analyse_absolute_fit",
]


# ====================================================================== #
# 内部工具
# ====================================================================== #
def _resolve_alpha_hat(
    model: CDModel,
    responses: np.ndarray,
    alpha_hat: Optional[np.ndarray],
    prior: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (MAP 模式索引, P(X=1|α̂) 矩阵)，形状 (n,) 与 (n, J)。"""
    from . import diagnose_patterns

    if alpha_hat is None:
        diag = diagnose_patterns(model, responses, prior=prior)
        idx = diag.pattern_indices
    else:
        a = np.asarray(alpha_hat, dtype=np.int8)
        if a.ndim != 2 or a.shape[1] != model.q.n_attrs:
            raise ValueError(
                f"alpha_hat 形状应为 (n_examinees, {model.q.n_attrs})"
            )
        patterns = model.q.attribute_patterns().astype(np.int8)
        lookup = {tuple(row): i for i, row in enumerate(patterns)}
        idx = np.array([lookup[tuple(row)] for row in a], dtype=np.int64)

    prob = model.prob_matrix()              # (S, J)
    return idx, prob[idx]                   # (n,), (n, J)


def _rest_score_bins(rest: np.ndarray, n_bins: int) -> np.ndarray:
    """按剩余得分做等频分箱，返回 0..n_bins-1 的箱号。"""
    if n_bins <= 1 or np.unique(rest).size <= 1:
        return np.zeros(rest.shape[0], dtype=np.int64)
    edges = np.quantile(rest, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.searchsorted(edges, rest, side="right").astype(np.int64)


def _chi2_p_value(stat: float, df: int) -> float:
    """卡方上尾概率（复用 diagnostics 中自实现的数值例程）。"""
    from . import likelihood_ratio_test

    if df <= 0:
        return 1.0
    # likelihood_ratio_test 的统计量 = 2·Δll，这里直接构造等价输入
    out = likelihood_ratio_test(0.0, 0, stat / 2.0, df)
    return float(out["p_value"])


# ====================================================================== #
# 1. 题目级绝对拟合
# ====================================================================== #
@dataclass
class ItemFitResult:
    """题目级绝对拟合检验结果。"""

    chi_square: np.ndarray
    g_square: np.ndarray
    df: np.ndarray
    p_value: np.ndarray
    p_value_g: np.ndarray
    n_cells: np.ndarray
    max_abs_residual: np.ndarray
    rmse_residual: np.ndarray
    item_names: List[str] = field(default_factory=list)
    n_bins: int = 5

    # ------------------------------------------------------------------ #
    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame({
            "题号": self.item_names,
            "χ²": np.round(self.chi_square, 3),
            "G²": np.round(self.g_square, 3),
            "df": self.df,
            "p值": np.round(self.p_value, 4),
            "单元数": self.n_cells,
            "最大残差": np.round(self.max_abs_residual, 4),
            "残差RMSE": np.round(self.rmse_residual, 4),
        })

    def poorly_fitting(self, alpha: float = 0.05):
        """返回未通过检验的题目（按 p 值升序）。"""
        mask = self.p_value < alpha
        tbl = self.to_dataframe()
        return tbl[mask].sort_values("p值").reset_index(drop=True)

    def worst_items(self, top_n: int = 5):
        """按最大残差排序的最差题目。"""
        order = np.argsort(-self.max_abs_residual)[:top_n]
        tbl = self.to_dataframe()
        return tbl.iloc[order].reset_index(drop=True)

    def summary(self) -> str:
        n = len(self.chi_square)
        bad = int((self.p_value < 0.05).sum())
        lines = [
            "题目级绝对拟合检验",
            "-" * 56,
            f"  题目数                : {n}",
            f"  p<0.05 的题目数       : {bad} ({bad / max(n, 1):.1%})",
            f"  最大残差 均值 / 最大  : "
            f"{self.max_abs_residual.mean():.4f} / {self.max_abs_residual.max():.4f}",
        ]
        if bad:
            worst = np.argsort(self.p_value)[: min(bad, 8)]
            lines.append("")
            lines.append("  拟合最差的题目:")
            for i in worst:
                nm = self.item_names[i] if i < len(self.item_names) else f"I{i+1}"
                lines.append(
                    f"    {nm:<12s} χ²={self.chi_square[i]:8.3f}  "
                    f"df={self.df[i]:3d}  p={self.p_value[i]:.4f}  "
                    f"最大残差={self.max_abs_residual[i]:.4f}"
                )
        return "\n".join(lines)


def item_fit(
    model: CDModel,
    responses: np.ndarray,
    alpha_hat: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    n_bins: int = 5,
    min_expected: float = 1.0,
    item_names: Optional[Sequence[str]] = None,
) -> ItemFitResult:
    """题目级 Pearson χ² / 似然比 G² 绝对拟合检验。

    Parameters
    ----------
    model : CDModel
        已标定的响应模型。
    responses : ndarray, shape (n, J)
        0/1 作答矩阵。
    alpha_hat : ndarray, optional
        (n, K) 的属性模式估计；不给则内部用 MAP 估计。
    prior : ndarray, optional
        传给 MAP 估计的属性先验。
    n_bins : int
        在每个"MAP 属性组"内按剩余得分再分的箱数（默认 5）。
        ``n_bins=1`` 表示只用属性分组。
    min_expected : float
        期望频数低于此值的单元会被剔除。
    item_names : sequence of str, optional

    Returns
    -------
    ItemFitResult
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = responses.shape
    q = model.q
    if j != q.n_items:
        raise ValueError(f"作答矩阵列数 {j} 与题目数 {q.n_items} 不符")

    idx, p_hat = _resolve_alpha_hat(model, responses, alpha_hat, prior)
    patterns = q.attribute_patterns()
    alpha_mat = patterns[idx]                      # (n, K)

    chi2 = np.zeros(j)
    g2 = np.zeros(j)
    df = np.zeros(j, dtype=int)
    p_val = np.ones(j)
    p_val_g = np.ones(j)
    n_cells = np.zeros(j, dtype=int)
    max_res = np.zeros(j)
    rmse_res = np.zeros(j)

    # 缺失作答：该题不进入统计；剩余得分按"已作答题目数"等比还原到完整题量尺度，
    # 这样无缺失时结果与原先完全一致，有缺失时分箱仍然可比。
    obs_mask = np.isfinite(responses)
    resp_filled = np.where(obs_mask, responses, 0.0)
    n_obs_row = obs_mask.sum(axis=1)
    rest_all = resp_filled.sum(axis=1)             # 已作答部分得分（缺失计 0 但不进分母）

    for item in range(j):
        required = np.where(q.matrix[item] == 1)[0]
        r = len(required)
        if r == 0:
            # 该题不考查任何属性：全样本一个预测概率
            group = np.zeros(n, dtype=np.int64)
            n_group = 1
        else:
            sub = alpha_mat[:, required].astype(np.int64)
            # 组号 = 二进制展开（首位为最高位，与 QMatrix 行序一致）
            group = np.zeros(n, dtype=np.int64)
            for k in range(r):
                group = (group << 1) | sub[:, k]
            n_group = 1 << r

        # 只保留该题有作答的被试
        keep = obs_mask[:, item]
        n_other = np.maximum(n_obs_row - 1, 1)
        rest_obs = rest_all - resp_filled[:, item]
        rest = rest_obs * (j - 1) / n_other         # 等比还原到完整尺度
        bin_idx = _rest_score_bins(rest[keep], n_bins)
        cell = group[keep] * max(n_bins, 1) + bin_idx

        x_j = responses[keep, item]
        p_j = np.clip(p_hat[keep, item], 1e-6, 1 - 1e-6)

        stat_c = 0.0
        stat_g = 0.0
        used = 0
        res_list: List[float] = []

        for c in np.unique(cell):
            m = cell == c
            n_c = float(m.sum())
            if n_c < 1:
                continue
            o = float(x_j[m].sum())
            exp = float(p_j[m].sum())            # 期望答对人数
            exp0 = n_c - exp
            if exp < min_expected or exp0 < min_expected:
                continue
            used += 1
            stat_c += (o - exp) ** 2 / exp
            stat_c += ((n_c - o) - exp0) ** 2 / exp0
            if o > 0:
                stat_g += 2.0 * o * np.log(o / exp)
            if (n_c - o) > 0:
                stat_g += 2.0 * (n_c - o) * np.log((n_c - o) / exp0)
            res_list.append(abs(o / n_c - exp / n_c))

        chi2[item] = stat_c
        g2[item] = max(stat_g, 0.0)
        n_cells[item] = used
        df[item] = max(used - 1, 0)
        p_val[item] = _chi2_p_value(stat_c, int(df[item]))
        p_val_g[item] = _chi2_p_value(max(stat_g, 0.0), int(df[item]))
        if res_list:
            max_res[item] = float(np.max(res_list))
            rmse_res[item] = float(np.sqrt(np.mean(np.square(res_list))))

    names = list(item_names) if item_names is not None else list(q.item_names or [])
    return ItemFitResult(
        chi_square=chi2,
        g_square=g2,
        df=df,
        p_value=p_val,
        p_value_g=p_val_g,
        n_cells=n_cells,
        max_abs_residual=max_res,
        rmse_residual=rmse_res,
        item_names=names,
        n_bins=int(n_bins),
    )


# ====================================================================== #
# 2. 局部依赖 Q3
# ====================================================================== #
@dataclass
class LocalDependenceResult:
    """Yen Q3 局部依赖分析结果。"""

    q3: np.ndarray                       # (J, J) 残差相关矩阵
    threshold: float = 0.2
    mean_q3: float = 0.0
    max_q3: float = 0.0
    n_flagged: int = 0
    pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    item_names: List[str] = field(default_factory=list)

    def top_pairs(self, n: int = 10):
        """Q3 最高的 n 对题目（含题号）。"""
        import pandas as pd

        rows = []
        for a, b, v in self.pairs[:n]:
            rows.append({
                "题号A": self._name(a),
                "题号B": self._name(b),
                "Q3": round(v, 4),
            })
        return pd.DataFrame(rows)

    def _name(self, i: int) -> str:
        return self.item_names[i] if i < len(self.item_names) else f"I{i + 1}"

    def to_dataframe(self):
        import pandas as pd

        names = self.item_names or [f"I{i + 1}" for i in range(self.q3.shape[0])]
        return pd.DataFrame(self.q3, index=names, columns=names)

    def flagged_pairs(self, top_n: Optional[int] = None):
        """Q3 超过阈值的题目对。"""
        import pandas as pd

        sel = self.pairs if top_n is None else self.pairs[:top_n]
        sel = [(a, b, v) for a, b, v in sel if v > self.threshold]
        return pd.DataFrame([
            {"题号A": self._name(a), "题号B": self._name(b), "Q3": round(v, 4)}
            for a, b, v in sel
        ])

    def summary(self) -> str:
        lines = [
            "局部依赖分析（Yen Q3）",
            "-" * 56,
            f"  题目数          : {self.q3.shape[0]}",
            f"  题目对数        : {len(self.pairs)}",
            f"  平均 Q3         : {self.mean_q3:.4f}",
            f"  最大 Q3         : {self.max_q3:.4f}",
            f"  超阈值(>{self.threshold}) 对数: {self.n_flagged}",
        ]
        if self.n_flagged:
            lines.append("")
            lines.append("  Q3 最高的题目对（可能存在局部依赖）:")
            for a, b, v in self.pairs[: min(self.n_flagged, 8)]:
                lines.append(
                    f"    {self._name(a):<10s} × {self._name(b):<10s} Q3={v:.4f}"
                )
        else:
            lines.append("")
            lines.append("  未检出超过阈值的局部依赖对。")
        return "\n".join(lines)


def local_dependence(
    model: CDModel,
    responses: np.ndarray,
    alpha_hat: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    threshold: float = 0.2,
    item_names: Optional[Sequence[str]] = None,
) -> LocalDependenceResult:
    """Yen (1984) Q3 局部依赖分析。

    残差定义为 ``d_ij = X_ij - P(X_ij=1 | α̂_i)``；Q3 为残差的题目间相关。
    模型正确时 Q3 应在 0 附近小幅波动；显著偏高说明该对题目在控制属性后
    仍共享额外变异 —— 典型原因是 Q 矩阵漏标了它们共同考查的属性。

    Returns
    -------
    LocalDependenceResult
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = responses.shape

    idx, p_hat = _resolve_alpha_hat(model, responses, alpha_hat, prior)

    # 残差相关系数采用「配对删除」：某题缺失时，该被试不参与这道题的
    # 均值/标准差，也不参与涉及这道题的题目对。做法是把缺失残差置 0，
    # 再用共观测计数矩阵归一化 —— 完全数据下与 np.corrcoef 结果一致。
    obs = np.isfinite(responses)
    resid = np.where(obs, np.nan_to_num(responses, nan=0.0) - p_hat, 0.0)   # (n, J)
    cnt = obs.sum(axis=0).astype(np.float64)                                # (J,)
    mean = resid.sum(axis=0) / np.maximum(cnt, 1e-12)
    centered = np.where(obs, resid - mean, 0.0)
    var = (centered ** 2).sum(axis=0) / np.maximum(cnt, 1e-12)
    sd = np.sqrt(np.maximum(var, 0.0))

    valid = (sd > 1e-12) & (cnt >= 3)
    q3 = np.zeros((j, j))
    if valid.sum() >= 2:
        z = centered[:, valid] / sd[valid][None, :]
        obs_v = obs[:, valid].astype(np.float64)
        pair_cnt = obs_v.T @ obs_v                        # 每题对共观测人数
        s_mat = z.T @ z
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(pair_cnt >= 3, s_mat / np.maximum(pair_cnt, 1e-12), 0.0)
        q3[np.ix_(valid, valid)] = c
    np.fill_diagonal(q3, 0.0)

    pairs: List[Tuple[int, int, float]] = []
    for a in range(j):
        for b in range(a + 1, j):
            pairs.append((a, b, float(q3[a, b])))
    pairs.sort(key=lambda t: -t[2])

    vals = np.array([v for _, _, v in pairs]) if pairs else np.zeros(1)
    names = list(item_names) if item_names is not None else list(model.q.item_names or [])
    return LocalDependenceResult(
        q3=q3,
        threshold=float(threshold),
        mean_q3=float(vals.mean()),
        max_q3=float(vals.max()),
        n_flagged=int((vals > threshold).sum()),
        pairs=pairs,
        item_names=names,
    )


# ====================================================================== #
# 3. 汇总入口
# ====================================================================== #
@dataclass
class AbsoluteFitResult:
    """绝对拟合分析总结果。"""

    item_fit: ItemFitResult
    local_dependence: LocalDependenceResult
    observations: int = 0
    items: int = 0

    def summary(self) -> str:
        return "\n\n".join([
            self.item_fit.summary(),
            self.local_dependence.summary(),
        ])


def analyse_absolute_fit(
    model: CDModel,
    responses: np.ndarray,
    alpha_hat: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    n_bins: int = 5,
    q3_threshold: float = 0.2,
    item_names: Optional[Sequence[str]] = None,
) -> AbsoluteFitResult:
    """一次跑完题目级绝对拟合 + 局部依赖 Q3 分析。

    Examples
    --------
    >>> res = analyse_absolute_fit(model, responses)   # doctest: +SKIP
    >>> res.item_fit.poorly_fitting().shape[0]         # doctest: +SKIP
    0
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")

    idx, _ = _resolve_alpha_hat(model, responses, alpha_hat, prior)
    alpha_mat = model.q.attribute_patterns()[idx]

    return AbsoluteFitResult(
        item_fit=item_fit(
            model, responses, alpha_hat=alpha_mat, n_bins=n_bins,
            item_names=item_names,
        ),
        local_dependence=local_dependence(
            model, responses, alpha_hat=alpha_mat,
            threshold=q3_threshold, item_names=item_names,
        ),
        observations=int(responses.shape[0]),
        items=int(responses.shape[1]),
    )
