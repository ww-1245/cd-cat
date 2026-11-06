"""报告级静态图（matplotlib）。

与 ``calibration_plots`` 的区别：这里画的是**群体层面**与**仿真收敛**，
供命令行导出 PNG、写进论文或放进教学汇报。

所有函数返回 ``matplotlib.figure.Figure``，由调用方决定保存路径。
matplotlib 为可选依赖，未安装时导入本模块会抛 ``ImportError``，
调用方需要自行捕获（CLI 已做处理）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .calibration_plots import _mpl

#: 与 Web 界面一致的语义配色
BRAND = "#4F46E5"
TEAL = "#0D9488"
ROSE = "#E11D48"
AMBER = "#D97706"
BLUE = "#2563EB"
VIOLET = "#7C3AED"
PALETTE = [BRAND, TEAL, AMBER, BLUE, ROSE, VIOLET, "#0891B2"]

__all__ = ["plot_group_report", "plot_convergence", "plot_attribute_mastery"]


def _ratio_color(rate: float) -> str:
    """掌握率着色：低 = 警示红，中 = 琥珀，高 = 青绿。"""
    if rate < 0.6:
        return ROSE
    if rate < 0.85:
        return AMBER
    return TEAL


def plot_attribute_mastery(
    report,
    ax=None,
    threshold: float = 0.85,
):
    """属性掌握率横向条形图（薄弱在上，便于一眼看出补课重点）。"""
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.55 * report.n_attrs + 2))

    order = np.argsort(report.attr_mastery_rate, kind="stable")
    names = [report.attr_names[i] for i in order]
    rates = report.attr_mastery_rate[order]
    colors = [_ratio_color(float(r)) for r in rates]

    ax.barh(np.arange(len(order)), rates, color=colors, height=0.62)
    ax.axvline(threshold, color="#9CA3AF", ls="--", lw=1, zorder=0)
    ax.text(threshold, len(order) - 0.4, f" 达标线 {threshold:.0%}",
            color="#6B7280", fontsize=9, va="center")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("掌握率")
    ax.set_title("各属性掌握率（由低到高）", fontweight="bold")
    ax.grid(axis="x", alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    for i, r in enumerate(rates):
        ax.text(min(float(r) + 0.02, 0.97), i, f"{r:.1%}", va="center", fontsize=9)
    return ax.figure


def plot_group_report(report, threshold: float = 0.85):
    """群体报告四联图：属性掌握率 / 掌握概率分布 / 典型模式 / 雷达图。

    Parameters
    ----------
    report : GroupReport
    threshold : float
        属性掌握率的"达标线"，仅用于图上的参考竖线。
    """
    plt = _mpl()
    fig = plt.figure(figsize=(14.5, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30)

    # ---- (1) 属性掌握率 ----
    ax1 = fig.add_subplot(gs[0, 0])
    plot_attribute_mastery(report, ax=ax1, threshold=threshold)

    # ---- (2) 掌握概率的确定性分档 ----
    # 用「确定掌握 / 模棱两可 / 确定未掌握」三段堆叠条，比箱线更直观：
    # 中间那段越长，说明该属性还有很多人需要加题才能定论。
    ax2 = fig.add_subplot(gs[0, 1])
    k = report.n_attrs
    order = np.argsort(report.attr_mastery_rate, kind="stable")
    lo_band, hi_band = 0.2, 0.8
    seg_lo, seg_mid, seg_hi = [], [], []
    for j in order:
        p = report.probabilities[:, j]
        seg_lo.append(float((p < lo_band).mean()))
        seg_mid.append(float(((p >= lo_band) & (p <= hi_band)).mean()))
        seg_hi.append(float((p > hi_band).mean()))
    y = np.arange(len(order))
    ax2.barh(y, seg_hi, color=TEAL, height=0.62, label="确定掌握 (P>0.8)")
    ax2.barh(y, seg_mid, left=seg_hi, color=AMBER, height=0.62, label="模棱两可 (0.2~0.8)")
    ax2.barh(y, seg_lo, left=np.array(seg_hi) + np.array(seg_mid),
             color=ROSE, height=0.62, label="确定未掌握 (P<0.2)")
    ax2.set_yticks(y)
    ax2.set_yticklabels([report.attr_names[j] for j in order])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("被试占比")
    ax2.set_title("各属性诊断的确定性分档\n（橙色越长＝越需要加题）", fontweight="bold")
    ax2.grid(axis="x", alpha=0.25, ls=":")
    ax2.set_axisbelow(True)
    # 图例放到坐标轴**下方**：每个条形都横跨 0~1，轴内没有不遮挡数据的位置。
    ax2.legend(
        fontsize=8.5, frameon=False, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, -0.13), handlelength=1.2, columnspacing=1.4,
    )
    for i, v in enumerate(seg_mid):
        if v > 0.04:
            ax2.text(seg_hi[i] + v / 2, i, f"{v:.0%}", va="center", ha="center",
                     fontsize=8.5, color="#3F2D00")

    # ---- (3) 典型掌握模式 ----
    ax3 = fig.add_subplot(gs[1, 0])
    top = [i for i in report.most_common_patterns(8) if report.pattern_freq[i] > 0]
    if top:
        labels, vals, info = [], [], []
        for idx in top[::-1]:
            bits = format(int(idx), f"0{k}b")
            labels.append(bits)
            vals.append(report.pattern_freq[idx])
            names = "、".join(report.attr_names[t] for t, b in enumerate(bits) if b == "1")
            info.append(names or "全未掌握")
        bars = ax3.barh(np.arange(len(vals)), vals, color=BRAND, height=0.62, alpha=0.85)
        ax3.set_yticks(np.arange(len(vals)))
        ax3.set_yticklabels(labels, family="monospace")
        ax3.set_xlabel("群体占比")
        ax3.set_title("典型掌握模式频次（TOP 8）", fontweight="bold")
        ax3.grid(axis="x", alpha=0.25, ls=":")
        ax3.set_axisbelow(True)
        for bar, v, txt in zip(bars, vals, info):
            ax3.text(v + max(vals) * 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{v:.1%}  {txt}", va="center", fontsize=8.5)
        ax3.set_xlim(0, max(vals) * 1.75)   # 给右侧的长中文标注留出空间
    else:
        ax3.text(0.5, 0.5, "无模式数据", ha="center", va="center",
                 transform=ax3.transAxes, color="#6B7280")
        ax3.set_axisbelow(True)

    # ---- (4) 雷达图 ----
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    by_group = report.group_attr_mastery()
    series = {"全班": report.attr_mastery_rate}
    if by_group:
        for g, rates in list(by_group.items())[:4]:
            series[str(g)] = rates
    for i, (name, rates) in enumerate(series.items()):
        vals = np.asarray(rates, dtype=float).tolist()
        vals_closed = vals + vals[:1]
        color = PALETTE[i % len(PALETTE)]
        ax4.plot(angles_closed, vals_closed, color=color, lw=2, label=name,
                 marker="o", markersize=4)
        if i == 0:
            ax4.fill(angles_closed, vals_closed, color=color, alpha=0.14)
    ax4.set_xticks(angles)
    ax4.set_xticklabels(report.attr_names, fontsize=9)
    ax4.tick_params(axis="x", pad=-1)          # 标签内收，避免超出画布被裁
    ax4.set_ylim(0, 1.02)
    ax4.set_yticks([0.5, 1.0])
    ax4.set_yticklabels(["0.50", "1.00"], fontsize=8, color="#6B7280")
    # 径向刻度放在两个属性标签之间的空档（135°），既不压顶部也不压左侧
    ax4.set_rlabel_position(135)
    ax4.set_title("属性掌握雷达图", fontweight="bold", pad=18)
    if len(series) > 1:
        ax4.legend(loc="upper right", bbox_to_anchor=(1.28, 1.14), fontsize=9, frameon=False)

    fig.suptitle(
        f"群体诊断报告 · {report.label}   "
        f"（N={report.n_examinees}，人均掌握 {report.mean_mastered:.2f}/{k} 个属性）",
        fontsize=13.5, fontweight="bold", y=0.975,
    )
    return fig


def plot_convergence(result, attr_names: Optional[Sequence[str]] = None):
    """仿真收敛三连图：属性边际收敛 / 用题量分布 / 题目曝光分布。"""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # ---- (1) 属性边际概率收敛曲线 ----
    ax = axes[0]
    curve = result.marginal_convergence_curve()
    if curve.size:
        names = list(attr_names) if attr_names else [
            f"属性{k + 1}" for k in range(curve.shape[1])
        ]
        for j in range(curve.shape[1]):
            ax.plot(np.arange(1, curve.shape[0] + 1), curve[:, j],
                    color=PALETTE[j % len(PALETTE)], lw=1.8, label=names[j])
        ax.axhline(0.5, color="#9CA3AF", ls="--", lw=1)
        ax.legend(fontsize=8.5, frameon=False, ncol=2)
    else:
        ax.text(0.5, 0.5, "无收敛记录", ha="center", va="center",
                transform=ax.transAxes, color="#6B7280")
    ax.set_xlabel("已作答题数")
    ax.set_ylabel("平均边际掌握概率")
    ax.set_title("属性边际概率收敛", fontweight="bold")
    ax.grid(alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    # ---- (2) 用题量分布 ----
    ax = axes[1]
    lens = np.array([r.n_answered for r in result.records], dtype=float)
    if lens.size:
        bins = min(20, max(5, len(np.unique(lens))))
        ax.hist(lens, bins=bins, color=TEAL, alpha=0.82, edgecolor="white")
        ax.axvline(lens.mean(), color=ROSE, ls="--", lw=1.6,
                   label=f"均值 {lens.mean():.1f}")
        ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel("用题量")
    ax.set_ylabel("人数")
    ax.set_title("用题量分布", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    # ---- (3) 题目曝光分布 ----
    ax = axes[2]
    usage = result.item_exposure_rate()
    if usage.size:
        order = np.argsort(-usage)
        ax.bar(np.arange(len(order)), usage[order] * 100,
               color=AMBER, alpha=0.85, width=0.9)
        ax.axhline(100.0 / len(order), color="#6B7280", ls="--", lw=1,
                   label=f"均匀曝光 {100.0 / len(order):.2f}%")
        ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel("题目（按曝光率降序）")
    ax.set_ylabel("曝光率 (%)")
    ax.set_title("题目曝光分布", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    fig.suptitle(
        f"仿真诊断 · PMR={result.pmr():.3f}  AMR={result.amr():.3f}  "
        f"平均题量={result.avg_items():.2f}  曝光SD={result.exposure_uniformity():.4f}",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    return fig
