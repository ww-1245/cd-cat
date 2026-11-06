"""CDM 标定结果可视化（matplotlib 版）。

面向脚本 / 报告场景，生成可直接嵌入论文的静态图。
所有函数返回 ``matplotlib.figure.Figure``，调用方可自行 ``savefig``。

    from cdcat.calibration_plots import calibration_report_figure
    fig = calibration_report_figure(analysis)
    fig.savefig("report.png", dpi=150)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "plot_ll_convergence",
    "plot_parameter_recovery",
    "plot_item_discrimination",
    "plot_fit_comparison",
    "plot_mastery_proportion",
    "plot_attr_reliability",
    "plot_pattern_heatmap",
    "calibration_report_figure",
    "plot_item_fit",
    "plot_q3_heatmap",
    "plot_attr_prior",
    "absolute_fit_report_figure",
]

_PALETTE = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#BA7517", "#D4537E", "#7F77DD"]


def _mpl():
    """延迟导入 matplotlib 并设置中文字体。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _boxplot_compat(ax, data, labels):
    """兼容 matplotlib 新旧版本的 boxplot 标签参数名。

    matplotlib < 3.9 用 ``labels=``，>= 3.9 改为 ``tick_labels=``。
    """
    try:
        return ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    except TypeError:
        return ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)


# ====================================================================== #
def plot_ll_convergence(
    results: Dict[str, object],
    title: str = "边际对数似然收敛曲线",
):
    """多个模型的 EM 收敛曲线。

    Parameters
    ----------
    results : dict[str, CalibrationResult]
        模型名 -> 标定结果。
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (name, res) in enumerate(results.items()):
        hist = res.log_likelihood_history
        ax.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3,
                color=_PALETTE[k % len(_PALETTE)], label=name, linewidth=2)
    ax.set_xlabel("EM 迭代次数")
    ax.set_ylabel("边际对数似然")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_parameter_recovery(
    estimated: np.ndarray,
    true: np.ndarray,
    param_name: str = "slip",
    ax=None,
):
    """参数恢复散点图（估计值 vs 真值，附 45° 参考线）。"""
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 5))
    else:
        fig = ax.figure
    e = np.asarray(estimated, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    ax.scatter(t, e, alpha=0.6, s=26, color="#534AB7", edgecolors="none")
    lim = [0, max(0.55, t.max() * 1.15, e.max() * 1.15)]
    ax.plot(lim, lim, "--", color="#B4B2A9", linewidth=1.2, label="理想恢复线")
    if len(t) > 1 and np.std(t) > 1e-9 and np.std(e) > 1e-9:
        r = np.corrcoef(e, t)[0, 1]
        mae = np.abs(e - t).mean()
        ax.text(0.04, 0.93, f"r = {r:.4f}\nMAE = {mae:.4f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", fc="#F7F6FB", ec="#D8D5EA"))
    ax.set_xlabel(f"真实 {param_name}")
    ax.set_ylabel(f"估计 {param_name}")
    ax.set_title(f"{param_name} 参数恢复")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(alpha=0.25)
    return fig


def plot_item_discrimination(
    discrimination: np.ndarray,
    item_names: Optional[Sequence[str]] = None,
    ax=None,
):
    """题目区分度条形图（按降序）。"""
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    d = np.asarray(discrimination, dtype=float).ravel()
    order = np.argsort(-d)
    names = [item_names[i] if item_names else f"I{i+1}" for i in order]
    vals = d[order]
    colors = ["#1D9E75" if v >= 0.5 else "#D85A30" if v < 0.3 else "#BA7517" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=max(4, min(9, 200 // max(len(vals), 1))))
    ax.invert_yaxis()
    ax.set_xlabel("区分度 = 1 - slip - guess")
    ax.set_title("题目区分度")
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.25, axis="x")
    return fig


def plot_fit_comparison(fits: Dict[str, object]):
    """模型拟合指标对比（AIC / BIC / AICc 分组柱状）。"""
    plt = _mpl()
    names = list(fits.keys())
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w, [fits[n].aic for n in names], w, label="AIC", color="#534AB7")
    ax.bar(x, [fits[n].bic for n in names], w, label="BIC", color="#1D9E75")
    ax.bar(x + w, [fits[n].aicc for n in names], w, label="AICc", color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("信息准则（越小越好）")
    ax.set_title("模型拟合指标对比")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


def plot_mastery_proportion(
    proportion: np.ndarray,
    attr_names: Optional[Sequence[str]] = None,
    ax=None,
):
    """各属性掌握比例。"""
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.4))
    else:
        fig = ax.figure
    prop = np.asarray(proportion, dtype=float).ravel()
    names = [attr_names[i] if attr_names and i < len(attr_names) else f"A{i+1}"
             for i in range(len(prop))]
    ax.bar(names, prop, color="#1D9E75", width=0.6)
    for i, v in enumerate(prop):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(0.5, ls=":", color="#B4B2A9", linewidth=1)
    ax.set_ylabel("掌握比例")
    ax.set_ylim(0, 1.1)
    ax.set_title("各属性掌握比例")
    ax.grid(alpha=0.25, axis="y")
    return fig


def plot_attr_reliability(
    reliability: np.ndarray,
    attr_names: Optional[Sequence[str]] = None,
    threshold: float = 0.6,
    ax=None,
):
    """属性级分类信度分布（箱线图）。"""
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.4))
    else:
        fig = ax.figure
    rel = np.asarray(reliability, dtype=float)
    names = [attr_names[i] if attr_names and i < len(attr_names) else f"A{i+1}"
             for i in range(rel.shape[1])]
    data = [rel[:, k] for k in range(rel.shape[1])]
    bp = _boxplot_compat(ax, data, names)
    for patch, c in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(threshold, ls="--", color="#D85A30", linewidth=1.2,
               label=f"信度阈值 {threshold}")
    ax.set_ylabel("属性分类信度")
    ax.set_ylim(0, 1.05)
    ax.set_title("属性级分类信度分布")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    return fig


def plot_pattern_heatmap(
    diagnosis,
    title: str = "属性掌握模式频次",
    ax=None,
):
    """诊断出的知识状态频次热力图（按模式索引）。"""
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.4))
    else:
        fig = ax.figure
    dist = diagnosis.pattern_distribution()
    s = diagnosis.posterior.shape[1]
    counts = np.zeros(s)
    for idx, c in dist.items():
        counts[idx] = c
    ax.bar(range(s), counts, color="#7F77DD", width=0.75)
    ax.set_xlabel("属性模式索引（二进制编码）")
    ax.set_ylabel("诊断人数")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    if s > 20:
        step = max(1, s // 20)
        ax.set_xticks(range(0, s, step))
    return fig


# ====================================================================== #
def calibration_report_figure(analysis, max_items_shown: int = 60):
    """生成一张包含 6 个子图的标准标定报告图。

    Parameters
    ----------
    analysis : CDMAnalysis
        :func:`cdcat.pipeline.run_cdm_analysis` 的输出。
    """
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # 1) LL 收敛
    ax = axes[0, 0]
    for k, (name, res) in enumerate(analysis.models.items()):
        hist = res.log_likelihood_history
        ax.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3,
                color=_PALETTE[k % len(_PALETTE)], label=name, linewidth=2)
    ax.set_xlabel("EM 迭代次数")
    ax.set_ylabel("边际对数似然")
    ax.set_title("EM 收敛曲线")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # 2) 题目区分度（最优模型）
    best = analysis.best_model
    res_best = analysis.models[best]
    disc = 1.0 - res_best.slip - res_best.guess
    order = np.argsort(-disc)[:max_items_shown]
    ax = axes[0, 1]
    names = [analysis.q_matrix.item_names[i] for i in order]
    vals = disc[order]
    colors = ["#1D9E75" if v >= 0.5 else "#D85A30" if v < 0.3 else "#BA7517" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=max(4, min(9, 200 // max(len(vals), 1))))
    ax.invert_yaxis()
    ax.set_xlabel("区分度")
    ax.set_title(f"题目区分度（{best}）")
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.25, axis="x")

    # 3) 拟合指标对比
    ax = axes[0, 2]
    names_m = list(analysis.fits.keys())
    x = np.arange(len(names_m))
    w = 0.26
    ax.bar(x - w, [analysis.fits[n].aic for n in names_m], w, label="AIC", color="#534AB7")
    ax.bar(x, [analysis.fits[n].bic for n in names_m], w, label="BIC", color="#1D9E75")
    ax.bar(x + w, [analysis.fits[n].aicc for n in names_m], w, label="AICc", color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(names_m)
    ax.set_ylabel("信息准则")
    ax.set_title("模型拟合指标对比")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # 4) 属性掌握比例
    ax = axes[1, 0]
    diag = analysis.diagnosis[best]
    prop = diag.mastery_proportion()
    anames = diag.attr_names or [f"A{i+1}" for i in range(len(prop))]
    ax.bar(anames, prop, color="#1D9E75", width=0.6)
    for i, v in enumerate(prop):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, ls=":", color="#B4B2A9", linewidth=1)
    ax.set_ylabel("掌握比例")
    ax.set_ylim(0, 1.1)
    ax.set_title("各属性掌握比例")
    ax.grid(alpha=0.25, axis="y")

    # 5) 属性分类信度
    ax = axes[1, 1]
    rel = diag.attr_reliability
    data = [rel[:, k] for k in range(rel.shape[1])]
    bp = _boxplot_compat(ax, data, anames)
    for patch, c in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(0.6, ls="--", color="#D85A30", linewidth=1.2, label="阈值 0.6")
    ax.set_ylabel("分类信度")
    ax.set_ylim(0, 1.05)
    ax.set_title("属性级分类信度")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # 6) 参数恢复（若有真值）
    ax = axes[1, 2]
    if analysis.recovery:
        # 取最优模型的 slip 恢复（需由调用方保存真值，这里退化为展示区分度分布）
        d = disc
        ax.hist(d, bins=min(20, max(5, len(d) // 3)), color="#534AB7",
                alpha=0.75, edgecolor="white")
        ax.axvline(d.mean(), ls="--", color="#D85A30", linewidth=1.5,
                   label=f"均值 {d.mean():.3f}")
        ax.set_xlabel("区分度")
        ax.set_ylabel("题目数")
        ax.set_title("区分度分布")
        ax.legend(fontsize=9)
    else:
        ax.hist(disc, bins=min(20, max(5, len(disc) // 3)), color="#534AB7",
                alpha=0.75, edgecolor="white")
        ax.set_xlabel("区分度")
        ax.set_ylabel("题目数")
        ax.set_title("区分度分布")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"CDM 标定分析报告 · 最优模型 {best} · "
        f"{analysis.q_matrix.n_items} 题 × {analysis.q_matrix.n_attrs} 属性",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ====================================================================== #
# 绝对拟合与局部依赖
# ====================================================================== #
def plot_item_fit(item_fit_result, top_n: int = 25, ax=None):
    """题目级绝对拟合：最差 top_n 题的 p 值横向条形图（含 0.05 参考线）。

    采用横向条形图，题号直接标注在纵轴，避免题量较大时 x 轴标签重叠。
    注意 p 值会被大样本放大，判读时请配合
    ``item_fit_result.max_abs_residual``（效应量）一起看。
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
    else:
        fig = ax.figure
    p = np.asarray(item_fit_result.p_value, dtype=float)
    names = item_fit_result.item_names or [f"I{i+1}" for i in range(len(p))]
    order = np.argsort(p)[:top_n]                    # 最差（p 最小）在前

    y = np.arange(len(order))[::-1]                  # 最差画在最上方
    ax.barh(y, p[order], color="#534AB7", height=0.7, label="p 值")
    ax.axvline(0.05, ls="--", color="#D85A30", linewidth=1.2, label="p = 0.05")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("拟合 p 值")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"题目级绝对拟合（最差 {len(order)} 题，p 越小越差）")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    return fig


def plot_q3_heatmap(ld_result, ax=None):
    """Q3 残差相关热力图（上三角），高值即潜在局部依赖。

    色标以 **0 为中心对称**，使"无局部依赖"的噪声呈现为中性色，
    正值（潜在局部依赖）才向暖色偏移，避免视觉误判。
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
    else:
        fig = ax.figure
    data = np.asarray(ld_result.q3, dtype=float)
    j = data.shape[0]
    upper = data[np.triu_indices(j, 1)] if j > 1 else np.array([0.0])
    # 对称色标：至少 ±0.5，保证 0 落在色标正中
    lim = max(0.5, float(np.ceil(np.abs(upper).max() * 10.0) / 10.0))
    q3 = np.ma.array(data, mask=np.tril(np.ones_like(data, dtype=bool), k=0))

    # 中性中心 + 红色表警示：Q3≈0 呈近白，高正值转红
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "cdc_q3", ["#6366F1", "#F8F8FC", "#E11D48"]
    ).with_extremes(bad="#FFFFFF")
    im = ax.imshow(q3, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(f"Q3 局部依赖热力图（阈值 {ld_result.threshold}，"
                 f"{ld_result.n_flagged} 对超标）")
    ax.set_xlabel("题目索引")
    ax.set_ylabel("题目索引")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Q3")
    if j > 15:
        step = max(1, round(j / 15))
        ticks = list(range(0, j, step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=8)
    return fig


def plot_attr_prior(model_result, attr_names=None, ax=None):
    """属性模式先验 π(α) 的分布（经验贝叶斯估计结果）。"""
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.4))
    else:
        fig = ax.figure
    prior = np.asarray(model_result.attr_prior, dtype=float)
    if prior is None or prior.size == 0:
        ax.set_title("属性先验 π(α)（无数据）")
        return fig
    s = prior.size
    ax.bar(range(s), prior, color="#1D9E75", width=0.8)
    ax.axhline(1.0 / s, ls="--", color="#D85A30", linewidth=1.2,
               label=f"均匀先验 1/{s} = {1.0/s:.4f}")
    ax.set_xlabel("属性模式索引（二进制编码）")
    ax.set_ylabel("π(α)")
    ax.set_title("属性模式先验 π(α)（经验贝叶斯估计）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    if s > 20:
        step = max(1, s // 20)
        ax.set_xticks(range(0, s, step))
    return fig


def absolute_fit_report_figure(analysis, max_items_shown: int = 25):
    """绝对拟合综合报告图（3 面板）：题目 p 值 / Q3 热力图 / 属性先验。"""
    plt = _mpl()
    fig = plt.figure(figsize=(17, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.1], wspace=0.28)
    af = analysis.absolute_fit
    best = analysis.best_model

    ax1 = fig.add_subplot(gs[0, 0])
    plot_item_fit(af.item_fit, top_n=max_items_shown, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_q3_heatmap(af.local_dependence, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_attr_prior(analysis.models[best], ax=ax3)

    fig.suptitle(
        f"绝对拟合与局部依赖诊断 · 模型 {best} · "
        f"{analysis.q_matrix.n_items} 题 × {analysis.q_matrix.n_attrs} 属性",
        fontsize=14, fontweight="bold",
    )
    # 含 colorbar 的布局用 subplots_adjust 更稳，tight_layout 会告警
    fig.subplots_adjust(left=0.06, right=0.96, top=0.84, bottom=0.16)
    return fig
