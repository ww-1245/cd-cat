"""CDM 标定相关图表的 Plotly 版本（供 Streamlit 界面使用）。

与 ``cdcat.calibration_plots``（matplotlib 静态图，面向报告）互为补充：
本模块输出交互式 Plotly Figure，便于在 Web 界面缩放、悬停查看。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "fig_ll_convergence",
    "fig_param_recovery",
    "fig_item_discrimination",
    "fig_fit_comparison",
    "fig_mastery_proportion",
    "fig_attr_reliability",
    "fig_pattern_distribution",
    "fig_item_fit",
    "fig_q3_heatmap",
    "fig_attr_prior",
    # 群体报告
    "fig_attr_mastery_bar",
    "fig_mastery_certainty",
    "fig_group_patterns",
    "fig_group_radar",
    "fig_group_compare",
    # 多级评分
    "fig_poly_category_probs",
    "fig_poly_step_recovery",
    # 知识地图与仪表
    "fig_attribute_map",
    "fig_quality_gauge",
    "set_theme",
    "FONT_FAMILY",
    "PALETTE",
]

# ---------------------------------------------------------------------- #
# 视觉主题
# ---------------------------------------------------------------------- #
FONT_FAMILY = (
    'Inter, "SF Pro Text", "PingFang SC", "Hiragino Sans GB", '
    '"Microsoft YaHei", -apple-system, BlinkMacSystemFont, "Segoe UI", '
    "Roboto, sans-serif"
)

#: 主题色板：靛蓝主色 + 青绿/琥珀/玫红点缀，兼顾对比度与和谐度
PALETTE = ["#4F46E5", "#0D9488", "#D97706", "#2563EB", "#E11D48", "#7C3AED", "#0891B2"]

_GRID = "#EDEDF6"
_AXIS_LINE = "#E3E3EF"
_INK = "#4E4E6B"


def _build_template(dark: bool = False) -> go.layout.Template:
    """构建统一的 Plotly 主题模板。

    模板在 Plotly 中优先级**最低**，因此各图里显式设置的
    ``height`` / ``margin`` / 轴标题等仍会生效，不会被覆盖。

    深色版只换文字与网格颜色；底色一律透明，由 Streamlit 的页面背景透出，
    这样切换主题时图与页面不会出现"色块不一致"。
    """
    if dark:
        ink, grid, axis_line = "#C7C7DC", "#2C2C48", "#3A3A58"
        tick, title_c, hover_bg, hover_bd, hover_fg = (
            "#9A9AB8", "#ECECF8", "#1C1C31", "#3A3A58", "#E6E6F2")
    else:
        ink, grid, axis_line = _INK, _GRID, _AXIS_LINE
        tick, title_c, hover_bg, hover_bd, hover_fg = (
            "#6C6C87", "#1A1A33", "#FFFFFF", "#E3E3EF", "#2A2A44")

    axis = dict(
        gridcolor=grid,
        zeroline=False,
        linecolor=axis_line,
        tickfont=dict(size=11.5, color=tick),
        title=dict(font=dict(size=12, color=tick)),
        automargin=True,
    )
    return go.layout.Template(
        layout=go.Layout(
            colorway=PALETTE,
            font=dict(family=FONT_FAMILY, size=12.5, color=ink),
            title=dict(font=dict(size=13.5, color=title_c), x=0.01, xanchor="left"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor=hover_bg,
                bordercolor=hover_bd,
                font=dict(family=FONT_FAMILY, size=12, color=hover_fg),
            ),
            legend=dict(borderwidth=0, font=dict(size=11.5)),
            xaxis=dict(axis),
            yaxis=dict(axis),
        )
    )


#: 当前是否为深色主题（由 :func:`set_theme` 维护）
_DARK_MODE = False


def use_cdc_theme(dark: Optional[bool] = None) -> None:
    """注册 ``cdc`` / ``cdc_dark`` 两套主题并启用其中一套。"""
    global _DARK_MODE
    if dark is not None:
        _DARK_MODE = bool(dark)
    pio.templates["cdc"] = _build_template(False)
    pio.templates["cdc_dark"] = _build_template(True)
    pio.templates.default = "cdc_dark" if _DARK_MODE else "cdc"


def set_theme(dark: bool) -> None:
    """切换图表主题（与页面深浅色开关联动）。"""
    use_cdc_theme(bool(dark))


def is_dark() -> bool:
    return _DARK_MODE

def ink() -> str:
    """当前主题下的正文色（图表里的数值标注要跟着主题走）。"""
    return "#E4E4F4" if _DARK_MODE else "#3A3A56"



use_cdc_theme()

_BASE_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


def fig_ll_convergence(results: Dict[str, object]) -> go.Figure:
    """多模型 EM 收敛曲线。"""
    fig = go.Figure()
    for k, (name, res) in enumerate(results.items()):
        hist = res.log_likelihood_history
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(hist) + 1)),
                y=hist,
                mode="lines+markers",
                name=name,
                line=dict(color=PALETTE[k % len(PALETTE)], width=2.4),
                marker=dict(size=4),
                hovertemplate="迭代 %{x}<br>logL=%{y:.2f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="EM 迭代次数", yaxis_title="边际对数似然",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_BASE_LAYOUT,
    )
    return fig


def fig_param_recovery(
    estimated: np.ndarray, true: np.ndarray, param_name: str = "slip"
) -> go.Figure:
    """参数恢复散点图（含 45° 参考线）。"""
    e = np.asarray(estimated, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    lim = [0, float(max(0.55, t.max() * 1.15, e.max() * 1.15))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t, y=e, mode="markers", name=f"{param_name}",
            marker=dict(color="#534AB7", size=7, opacity=0.65),
            hovertemplate="真值 %{x:.3f}<br>估计 %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lim, y=lim, mode="lines", name="理想恢复线",
            line=dict(color="#B4B2A9", dash="dash", width=1.4),
        )
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title=f"真实 {param_name}", range=lim),
        yaxis=dict(title=f"估计 {param_name}", range=lim),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_BASE_LAYOUT,
    )
    return fig


def fig_item_discrimination(
    discrimination: np.ndarray,
    item_names: Optional[Sequence[str]] = None,
    top_n: int = 30,
) -> go.Figure:
    """题目区分度条形图（按降序取前 top_n）。"""
    d = np.asarray(discrimination, dtype=float).ravel()
    order = np.argsort(-d)[:top_n]
    names = [item_names[i] if item_names else f"I{i+1}" for i in order]
    vals = d[order]
    colors = ["#1D9E75" if v >= 0.5 else "#D85A30" if v < 0.3 else "#BA7517" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals, y=names, orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        )
    )
    fig.update_layout(
        height=max(340, 20 * len(names)),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="区分度 = 1 - slip - guess", range=[0, 1.05]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def fig_fit_comparison(fits: Dict[str, object]) -> go.Figure:
    """模型拟合指标对比（AIC / BIC / AICc）。"""
    names = list(fits.keys())
    fig = go.Figure()
    for label, key, color in [("AIC", "aic", "#534AB7"),
                              ("BIC", "bic", "#1D9E75"),
                              ("AICc", "aicc", "#D85A30")]:
        fig.add_trace(
            go.Bar(
                x=names, y=[getattr(fits[n], key) for n in names],
                name=label, marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group", height=360, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="信息准则（越小越好）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_BASE_LAYOUT,
    )
    return fig


def fig_mastery_proportion(
    proportion: np.ndarray, attr_names: Optional[Sequence[str]] = None
) -> go.Figure:
    """各属性掌握比例。"""
    prop = np.asarray(proportion, dtype=float).ravel()
    names = [attr_names[i] if attr_names and i < len(attr_names) else f"A{i+1}"
             for i in range(len(prop))]
    # 数值标注单独成一层：直接挂在柱顶会被 0.5 参考线穿过——柱高本来就在
    # 0.47~0.52 一带，几乎必然与虚线相撞，数字看起来像被划掉。
    # 这里统一抬到 max(柱高, 参考线) 之上 0.045，永远不会压线。
    label_y = np.maximum(prop, 0.5) + 0.045
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names, y=prop, marker_color="#1D9E75",
            hovertemplate="%{x}<br>掌握比例 %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=names, y=label_y, mode="text",
            text=[f"{v:.3f}" for v in prop],
            textfont=dict(color=ink(), size=11.5),
            cliponaxis=False, hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#B4B2A9", layer="below")
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="掌握比例", range=[0, 1.22]),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def fig_attr_reliability(
    reliability: np.ndarray, attr_names: Optional[Sequence[str]] = None
) -> go.Figure:
    """属性级分类信度的箱线图。"""
    rel = np.asarray(reliability, dtype=float)
    names = [attr_names[i] if attr_names and i < len(attr_names) else f"A{i+1}"
             for i in range(rel.shape[1])]
    fig = go.Figure()
    for k in range(rel.shape[1]):
        fig.add_trace(
            go.Box(
                y=rel[:, k], name=names[k],
                marker_color=PALETTE[k % len(PALETTE)],
                boxpoints=False,
            )
        )
    fig.add_hline(y=0.6, line_dash="dash", line_color="#D85A30",
                  annotation_text="信度阈值 0.6")
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="属性分类信度", range=[0, 1.05]),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def fig_pattern_distribution(diagnosis) -> go.Figure:
    """诊断出的知识状态频次分布。"""
    dist = diagnosis.pattern_distribution()
    s = diagnosis.posterior.shape[1]
    counts = np.zeros(s, dtype=int)
    for idx, c in dist.items():
        counts[idx] = c
    fig = go.Figure(
        go.Bar(
            x=list(range(s)), y=counts, marker_color="#7F77DD",
            hovertemplate="模式 %{x}<br>人数 %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="属性模式索引（二进制编码）",
        yaxis_title="诊断人数",
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


# ====================================================================== #
# 绝对拟合与局部依赖
# ====================================================================== #
def fig_item_fit(item_fit_result, top_n: int = 30) -> go.Figure:
    """题目级绝对拟合：最差 top_n 题的 p 值（越小越差）。"""
    p = np.asarray(item_fit_result.p_value, dtype=float)
    res = np.asarray(item_fit_result.max_abs_residual, dtype=float)
    names = (item_fit_result.item_names
             or [f"I{i+1}" for i in range(len(p))])
    order = np.argsort(p)[:top_n]
    colors = ["#D85A30" if p[i] < 0.05 else "#534AB7" for i in order]

    fig = go.Figure(
        go.Bar(
            x=[names[i] for i in order],
            y=p[order],
            marker_color=colors,
            customdata=[[res[i], int(item_fit_result.df[i])] for i in order],
            hovertemplate=("题目 %{x}<br>p 值 %{y:.4f}<br>"
                           "最大残差 %{customdata[0]:.4f}<br>"
                           "df %{customdata[1]}<extra></extra>"),
        )
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="#D85A30",
                  annotation_text="p = 0.05")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="题目（按 p 值升序）", yaxis_title="拟合 p 值",
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def fig_q3_heatmap(ld_result) -> go.Figure:
    """Q3 残差相关热力图（上三角，色标以 0 为中心对称）。"""
    q3 = np.asarray(ld_result.q3, dtype=float).copy()
    j = q3.shape[0]
    upper = q3[np.triu_indices(j, 1)] if j > 1 else np.array([0.0])
    lim = max(0.5, float(np.ceil(np.abs(upper).max() * 10.0) / 10.0))
    # 只保留上三角，下三角置 NaN（不显示）
    mask = np.tril(np.ones_like(q3, dtype=bool), k=0)
    q3[mask] = np.nan
    names = (ld_result.item_names
             or [f"I{i+1}" for i in range(j)])

    # 中性中心 + 红色表警示：Q3≈0 呈近中性（无局部依赖），高正值转红。
    # 深色主题下中心色跟着压暗，否则整张矩阵会变成一块发光的白板。
    mid = "#181828" if _DARK_MODE else "#F8F8FC"
    fig = go.Figure(
        go.Heatmap(
            z=q3, x=names, y=names,
            colorscale=[[0.0, "#6366F1"], [0.5, mid], [1.0, "#E11D48"]],
            zmin=-lim, zmax=lim,
            colorbar=dict(title="Q3", outlinewidth=0, thickness=12),
            hovertemplate="%{y} × %{x}<br>Q3 = %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(420, 14 * j), margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        **_BASE_LAYOUT,
    )
    return fig


def fig_attr_prior(model_result) -> go.Figure:
    """属性模式先验 π(α) 分布，附均匀先验参考线。"""
    prior = np.asarray(model_result.attr_prior, dtype=float)
    s = prior.size
    fig = go.Figure(
        go.Bar(
            x=list(range(s)), y=prior, marker_color="#1D9E75",
            hovertemplate="模式 %{x}<br>π(α) = %{y:.5f}<extra></extra>",
        )
    )
    fig.add_hline(y=1.0 / s, line_dash="dash", line_color="#D85A30",
                  annotation_text=f"均匀先验 {1.0/s:.4f}")
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="属性模式索引（二进制编码）", yaxis_title="π(α)",
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


# ====================================================================== #
# 群体 / 班级报告
# ====================================================================== #
_MASTERY_COLORS = {"确定未掌握": "#E11D48", "模棱两可": "#D97706", "确定掌握": "#0D9488"}


def _ratio_color(rate: float, threshold: float = 0.85) -> str:
    """按掌握率分档取色：低=玫红（薄弱），中=琥珀，高=青绿。"""
    if rate < 0.6:
        return "#E11D48"
    if rate < threshold:
        return "#D97706"
    return "#0D9488"


def fig_attr_mastery_bar(report, threshold: float = 0.85) -> go.Figure:
    """各属性掌握率横向条形（最薄弱的排在最上方）。"""
    rates = np.asarray(report.attr_mastery_rate, dtype=float)
    probs = np.asarray(report.attr_mean_prob, dtype=float)
    names = list(report.attr_names)
    order = np.argsort(rates)                       # 升序 -> 画在纵轴上方为最弱
    y = [names[i] for i in order]
    colors = [_ratio_color(float(rates[i]), threshold) for i in order]

    fig = go.Figure(
        go.Bar(
            x=[float(rates[i]) for i in order], y=y, orientation="h",
            marker_color=colors,
            customdata=[[float(probs[i])] for i in order],
            hovertemplate=("%{y}<br>掌握率 %{x:.1%}"
                           "<br>平均掌握概率 %{customdata[0]:.3f}<extra></extra>"),
            text=[f"{rates[i]:.0%}" for i in order],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"目标 {threshold:.0%}")
    fig.update_layout(
        height=max(230, 46 * len(y) + 90),
        margin=dict(l=10, r=40, t=24, b=10),
        xaxis=dict(range=[0, 1.06], tickformat=".0%", title="掌握率"),
        yaxis=dict(title="", showgrid=False),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def fig_mastery_certainty(report, low: float = 0.2, high: float = 0.8) -> go.Figure:
    """确定性分档堆叠条：确定未掌握 / 模棱两可 / 确定掌握。

    模棱两可（琥珀）段越长，说明该属性越需要继续加题才能判清楚。
    """
    probs = np.asarray(report.probabilities, dtype=float)      # (n, K)
    names = list(report.attr_names)
    order = np.argsort(np.asarray(report.attr_mastery_rate, dtype=float))
    y = [names[i] for i in order]

    seg = {"确定未掌握": [], "模棱两可": [], "确定掌握": []}
    for i in order:
        p = probs[:, i]
        n = max(len(p), 1)
        seg["确定未掌握"].append(float((p < low).sum()) / n)
        seg["模棱两可"].append(float(((p >= low) & (p <= high)).sum()) / n)
        seg["确定掌握"].append(float((p > high).sum()) / n)

    fig = go.Figure()
    for name in ("确定未掌握", "模棱两可", "确定掌握"):
        fig.add_trace(
            go.Bar(
                x=seg[name], y=y, orientation="h", name=name,
                marker_color=_MASTERY_COLORS[name],
                hovertemplate=f"{name}<br>%{{y}}: %{{x:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=max(230, 42 * len(y) + 110),
        margin=dict(l=10, r=20, t=24, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", title="人数占比"),
        yaxis=dict(title="", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **_BASE_LAYOUT,
    )
    return fig


def fig_group_patterns(report, top_n: int = 10) -> go.Figure:
    """最高频知识状态（属性模式）频次。"""
    idxs = list(report.most_common_patterns(top_n))
    freq = np.asarray(report.pattern_freq, dtype=float)
    n_attrs = int(report.n_attrs)
    labels = [format(int(i), f"0{n_attrs}b") for i in idxs]
    vals = [float(freq[i]) for i in idxs]

    # 与群体掌握率差异最大的位置用强调色标出
    rates = np.asarray(report.attr_mastery_rate, dtype=float)
    colors = []
    for i in idxs:
        bits = np.array([int(c) for c in format(int(i), f"0{n_attrs}b")], dtype=float)
        colors.append("#4F46E5" if np.abs(bits - rates).mean() > 0.25 else "#94A3B8")

    fig = go.Figure(
        go.Bar(
            x=labels, y=vals, marker_color=colors,
            hovertemplate="模式 %{x}<br>占比 %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=24, b=10),
        xaxis_title="属性模式（1 = 掌握）", yaxis_title="被试占比",
        yaxis=dict(tickformat=".0%"),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def fig_group_radar(report, max_groups: int = 6) -> go.Figure:
    """属性掌握率雷达图：总体 + 各组（超过 ``max_groups`` 组只画前几组）。"""
    names = list(report.attr_names)
    if not names:
        return go.Figure(layout=go.Layout(height=340))

    fig = go.Figure()
    overall = np.asarray(report.attr_mastery_rate, dtype=float).tolist()
    fig.add_trace(
        go.Scatterpolar(
            r=overall + overall[:1], theta=names + names[:1],
            name="总体", fill="toself", line=dict(color="#4F46E5", width=2.4),
            fillcolor="rgba(79,70,229,0.14)",
            hovertemplate="%{theta}<br>掌握率 %{r:.1%}<extra>总体</extra>",
        )
    )

    grouped = report.group_attr_mastery() or {}
    for pos, (gname, rates) in enumerate(grouped.items()):
        if pos >= max_groups:
            break
        vals = np.asarray(rates, dtype=float).tolist()
        color = PALETTE[(pos + 1) % len(PALETTE)]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + vals[:1], theta=names + names[:1],
                name=str(gname), line=dict(color=color, width=1.8, dash="dot"),
                hovertemplate=f"%{{theta}}<br>掌握率 %{{r:.1%}}<extra>{gname}</extra>",
            )
        )

    fig.update_layout(
        height=390, margin=dict(l=30, r=30, t=30, b=20),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                            gridcolor=_GRID, showline=False, tickfont=dict(size=10)),
            angularaxis=dict(gridcolor=_GRID),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, x=0),
        **_BASE_LAYOUT,
    )
    return fig


def fig_group_compare(report) -> go.Figure:
    """分组 × 属性掌握率分组柱状图。"""
    grouped = report.group_attr_mastery() or {}
    names = list(report.attr_names)
    fig = go.Figure()
    for pos, (gname, rates) in enumerate(grouped.items()):
        fig.add_trace(
            go.Bar(
                x=names, y=[float(v) for v in rates], name=str(gname),
                marker_color=PALETTE[pos % len(PALETTE)],
                hovertemplate=f"%{{x}}<br>掌握率 %{{y:.1%}}<extra>{gname}</extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=340, margin=dict(l=10, r=10, t=24, b=10),
        xaxis_title="属性", yaxis=dict(tickformat=".0%", title="掌握率", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **_BASE_LAYOUT,
    )
    return fig


# ====================================================================== #
# 多级评分（顺序 G-DINA）
# ====================================================================== #
def fig_poly_category_probs(pq, model, item_idx: int = 0,
                            patterns: Optional[Sequence[int]] = None) -> go.Figure:
    """某题在各知识状态下取各档位的概率（分组柱状）。"""
    cats = model.category_probs()                  # (2^K, J, Cmax)
    n_cat = int(pq.n_categories[item_idx])
    sub = cats[:, item_idx, :]                     # (2^K, Cmax)

    if patterns is None:
        k = int(pq.n_attrs)
        # 取"全未掌握 / 全掌握 / 中间状态"三个代表性知识状态
        patterns = sorted({0, 2 ** k - 1, int((2 ** k - 1) // 2)})
    patterns = list(patterns)

    fig = go.Figure()
    for c in range(n_cat):
        fig.add_trace(
            go.Bar(
                x=[format(int(p), f"0{pq.n_attrs}b") for p in patterns],
                y=[float(sub[p, c]) for p in patterns],
                name=f"{c} 分",
                marker_color=PALETTE[c % len(PALETTE)],
                hovertemplate=f"{c} 分<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=340, margin=dict(l=10, r=10, t=24, b=10),
        xaxis_title="属性模式（1 = 掌握）", yaxis=dict(title="概率", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **_BASE_LAYOUT,
    )
    return fig


def fig_poly_step_recovery(est, truth, pq) -> go.Figure:
    """顺序步骤 slip 参数的真值 vs 估计散点（对角线为完美恢复）。"""
    est_p = np.asarray(est.step_probs(), dtype=float)      # (2^K, S)
    true_p = np.asarray(truth.step_probs(), dtype=float)
    x = true_p.ravel()
    y = est_p.ravel()
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")

    lim = [float(min(x.min(), y.min())) - 0.03, float(max(x.max(), y.max())) + 0.03]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=7, color="#4F46E5", opacity=0.55,
                        line=dict(width=0.6, color="#FFFFFF")),
            hovertemplate="真值 %{x:.3f}<br>估计 %{y:.3f}<extra></extra>",
            name="步骤答对概率",
        )
    )
    fig.add_trace(
        go.Scatter(x=lim, y=lim, mode="lines", name="完美恢复",
                   line=dict(color="#94A3B8", dash="dash", width=1.6))
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        title=f"步骤参数恢复（r = {corr:.3f}）",
        xaxis=dict(title="真实 P(答对)", range=lim),
        yaxis=dict(title="估计 P(答对)", range=lim),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **_BASE_LAYOUT,
    )
    return fig


# ====================================================================== #
# 知识地图（属性层级 DAG）
# ====================================================================== #
def _map_layout(hierarchy) -> Dict[int, tuple]:
    """按层级给节点排坐标：层号决定纵坐标，同层内横向均分。"""
    levels = np.asarray(hierarchy.levels, dtype=int)
    pos: Dict[int, tuple] = {}
    for lv in np.unique(levels):
        nodes = np.where(levels == lv)[0]
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = 0.5 if n == 1 else i / (n - 1)
            pos[int(node)] = (float(x), -float(lv))
    return pos


# 知识地图节点的三档配色，与 marker.colorscale 保持一致
_NODE_STOPS = ((0.0, (225, 29, 72)), (0.5, (217, 119, 6)), (1.0, (13, 148, 136)))
_NODE_INK_DARK = "#0B1220"


def _mix_node_colour(t: float) -> tuple:
    """按掌握率拿到节点实际填充色的 RGB（复刻 colorscale 的线性插值）。"""
    t = float(np.clip(t, 0.0, 1.0)) if np.isfinite(t) else 0.0
    for (t0, c0), (t1, c1) in zip(_NODE_STOPS, _NODE_STOPS[1:]):
        if t <= t1:
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(int(round(a + (b - a) * w)) for a, b in zip(c0, c1))
    return _NODE_STOPS[-1][1]


def _rel_luminance(rgb: tuple) -> float:
    def _lin(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (_lin(int(v)) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _readable_on(rgb: tuple) -> str:
    """在白字与近黑字之间挑对比度更高的那个。

    三档底色里琥珀（#D97706）与青绿（#0D9488）配白字只有 3.2:1 / 3.8:1，
    小字号下偏弱；玫红（#E11D48）反过来白字更好。所以逐点判断，不写死。
    """
    lum = _rel_luminance(rgb)
    contrast_white = 1.05 / (lum + 0.05)
    contrast_dark = (lum + 0.05) / (_rel_luminance((11, 18, 32)) + 0.05)
    return _NODE_INK_DARK if contrast_dark >= contrast_white else "#FFFFFF"



def _short_label(text: object, limit: int = 6) -> str:
    """属性名过长时截断加省略号：节点下方只有约 46px 的可读宽度。"""
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def fig_attribute_map(
    hierarchy,
    mastery: Optional[Sequence[float]] = None,
    attr_names: Optional[Sequence[str]] = None,
    height: Optional[int] = None,
):
    """知识地图：把属性层级画成有向图，节点按掌握率染色。

    横轴无含义，纵轴是层级（先备属性在上）。箭头 ``A → B`` 表示
    "掌握 B 必须先掌握 A"。节点颜色从玫红（掌握率低）过渡到青绿（高），
    一眼就能看出"卡在哪一环"——这是纯表格看不出来的东西。
    """
    k = int(hierarchy.n_attrs)
    names = [
        _short_label(n)
        for n in (attr_names or hierarchy.attr_names or [f"A{i + 1}" for i in range(k)])
    ]
    pos = _map_layout(hierarchy)
    edges = list(hierarchy.edges)

    xs = [pos[i][0] for i in range(k)]
    ys = [pos[i][1] for i in range(k)]

    if mastery is None:
        colors = ["#4F46E5"] * k
        inner = [""] * k
        hover_extra = ""
        ink_colors = ["#FFFFFF"] * k
    else:
        m = np.asarray(mastery, dtype=float).ravel()
        colors = list(m)
        # 圆里只放两位数的百分比（必然放得下），属性名放到圆下方，
        # 否则中文属性名会被 46px 的圆裁掉。
        inner = [f"{v:.0%}" if np.isfinite(v) else "—" for v in m]
        hover_extra = "<br>掌握率 %{text}"
        # 文字颜色跟着底色走：琥珀/青绿用近黑字，玫红用白字
        ink_colors = [_readable_on(_mix_node_colour(v)) for v in m]

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=inner,
        textposition="middle center",
        textfont=dict(size=11.5, color=ink_colors, family=FONT_FAMILY),
        customdata=names,
        marker=dict(
            size=46,
            color=colors,
            colorscale=[[0.0, "#E11D48"], [0.5, "#D97706"], [1.0, "#0D9488"]],
            cmin=0.0, cmax=1.0,
            showscale=False,
            line=dict(width=1.6, color="rgba(255,255,255,.85)"),
            opacity=0.97,
        ),
        hovertemplate=("<b>%{customdata}</b>" + hover_extra + "<extra></extra>"
                       if mastery is not None
                       else "<b>%{customdata}</b><extra></extra>"),
        showlegend=False,
        cliponaxis=False,
    )

    label_ink = "#C9C9E2" if _DARK_MODE else "#3A3A56"
    name_trace = go.Scatter(
        x=xs, y=[y - 0.42 for y in ys],
        mode="text",
        text=names,
        textposition="middle center",
        textfont=dict(size=11, color=label_ink, family=FONT_FAMILY),
        hoverinfo="skip",
        showlegend=False,
        cliponaxis=False,
    )

    fig = go.Figure([node_trace, name_trace])

    # 边：用 annotation 画带箭头的连线，standoff 让箭头不压住节点
    for a, b in edges:
        xa, ya = pos[int(a)]
        xb, yb = pos[int(b)]
        fig.add_annotation(
            x=xb, y=yb, ax=xa, ay=ya,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.15, arrowwidth=1.6,
            arrowcolor="#9CA3AF" if not _DARK_MODE else "#5A5A80",
            standoff=28, startstandoff=28,
        )

    pad_x = 0.16
    levels = int(max(ys)) - int(min(ys)) + 1
    fig.update_layout(
        height=height or max(260, 130 + 88 * levels),
        margin=dict(l=18, r=18, t=16, b=34),
        xaxis=dict(visible=False, range=[-pad_x, 1 + pad_x]),
        yaxis=dict(visible=False,
                   range=[float(min(ys)) - 0.95, float(max(ys)) + 0.62]),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


# ====================================================================== #
# 质量仪表（环形）
# ====================================================================== #
def _quality_colour(v: float) -> str:
    """质量指标的三档配色：优（青绿）／可（琥珀）／需改进（玫红）。"""
    if v >= 0.85:
        return "#0D9488"
    if v >= 0.65:
        return "#D97706"
    return "#E11D48"


def fig_quality_gauge(
    value: float,
    title: str = "",
    target: Optional[float] = None,
    height: int = 190,
    thresholds: Optional[tuple] = None,
):
    """半环仪表：把一个 0~1 的质量指标画成单值盘。

    用于"一页式仪表盘"——比数字更直观地看出离目标还差多少。
    ``target`` 给出时在盘上画一条目标刻度线（``go.Indicator`` 的
    ``threshold`` 原生支持，无需手算极坐标箭头）。

    参数
    ----
    value : float
        指标值，超出 [0, 1] 会被裁剪；``NaN`` 视为 0。
    title : str
        盘下方标题。
    target : float, optional
        目标线位置，如 0.90 的模式判准率目标。
    height : int
        图高（px）。
    thresholds : tuple of (float, str), optional
        刻度配色断点，默认 ``((0.65, 琥珀), (0.85, 青绿))``。
    """
    v = float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0
    col = _quality_colour(v)
    dark = _DARK_MODE
    ink = "#ECECF8" if dark else "#1A1A33"
    sub = "#9A9AB8" if dark else "#8A8AA0"
    track = "rgba(140,140,170,0.30)" if dark else "rgba(140,140,170,0.16)"

    gauge = dict(
        shape="angular",
        axis=dict(
            range=[0, 100],
            # 半环本身已经足够表达"走了多少"，刻度数字在小尺寸下只会互相挤压，
            # 故隐藏刻度标签，仅保留两端锚点。
            showticklabels=False,
            ticks="",
            tickwidth=0,
        ),
        bar=dict(color=col, thickness=0.42),
        bgcolor=track,
        borderwidth=0,
    )
    if target is not None and 0.0 <= float(target) <= 1.0:
        gauge["threshold"] = dict(
            line=dict(color="#64748B" if not dark else "#B9B9D4", width=2.6),
            thickness=0.92,
            value=float(target) * 100.0,
        )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v * 100.0,
            number=dict(
                font=dict(size=24, color=col, family=FONT_FAMILY),
                suffix="%",
                valueformat=".1f",
            ),
            gauge=gauge,
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    if title:
        fig.add_annotation(
            text=title, x=0.5, y=0.02, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11.5, color=sub),
        )
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=6, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family=FONT_FAMILY, color=ink),
    )
    return fig
