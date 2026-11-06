"""CD-CAT 选题与仿真系统 —— Streamlit 交互式界面。

启动方式::

    streamlit run demo/app.py                    # 浅色（默认）
    streamlit run demo/app.py --theme.base dark  # 深色

包含九大功能页：
1. 质量看板   —— 一屏汇总诊断精度、知识地图、逐题体检与一键报告
2. 在线施测   —— 模拟一名考生逐题作答，实时展示后验收敛
3. 蒙特卡洛仿真 —— 批量仿真并评估 PMR/AMR、题量、曝光
4. 准则对比   —— 同一批被试下对比各选题准则
5. 题库诊断   —— Q 矩阵校验、属性覆盖、关联图
6. 群体报告   —— 班级/群体掌握率、薄弱知识点、分组对比 + Q 矩阵自动修正
7. 参数标定   —— EM 算法从作答数据估计题目参数
8. 多级评分   —— 顺序 G-DINA：多级题库标定、档位概率、多级自适应施测
9. 系统说明

界面主题由 ``--theme.base`` 或 ``.streamlit/config.toml`` 决定（见 ``_detect_dark``），
图表模板与页面配色会一起切换。
"""

from __future__ import annotations

import html as _html
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 保证可导入 cdcat 包
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdcat import (  # noqa: E402
    QMatrix,
    CDCATSimulator,
    Evaluator,
    FixedLength,
    PatternConfidence,
    CompositeRule,
    StabilityRule,
    MaxExposureLimit,
    RandomesqueExposure,
    StratifiedExposure,
    SympsonHetter,
    NoExposureControl,
    analyse_identifiability,
    analyse_items,
    build_examinee_report,
    build_group_report,
    create_model,
    run_cdm_analysis,
    validate_q_matrix,
)
from cdcat.utils import pattern_to_str  # noqa: E402
from demo.cdm_charts import (  # noqa: E402
    fig_ll_convergence,
    fig_param_recovery,
    fig_item_discrimination as fig_cdm_discrimination,
    fig_fit_comparison,
    fig_mastery_proportion,
    fig_attr_reliability,
    fig_pattern_distribution,
    fig_item_fit,
    fig_q3_heatmap,
    fig_attr_prior,
    fig_attr_mastery_bar,
    fig_mastery_certainty,
    fig_group_patterns,
    fig_group_radar,
    fig_group_compare,
    fig_poly_category_probs,
    fig_poly_step_recovery,
    fig_attribute_map,
    fig_quality_gauge,
    set_theme,
)
from demo.bank_utils import (  # noqa: E402
    build_preset_bank,
    list_presets,
    bank_from_json,
)

st.set_page_config(page_title="CD-CAT 选题与仿真系统", page_icon="◆", layout="wide")

# ---------------------------------------------------------------------- #
# 主题检测：跟随应用的 theme.base（命令行 --theme.base dark 或 config.toml）
# ---------------------------------------------------------------------- #
def _detect_dark() -> bool:
    """读取应用当前生效的明暗主题。

    判定顺序（实测结论，勿随意调换）：

    1. ``st.get_option("theme.base")`` —— 应用侧的权威来源，同时反映
       ``--theme.base dark`` 命令行参数与 ``.streamlit/config.toml``。实测
       传参时返回 ``"dark"``、不传时返回 ``None``（默认即浅色）。
    2. ``st.context.theme`` —— 兜底。它读的是**浏览器**上报的主题，与本应用
       配置无关：实测无论是否传 ``--theme.base dark`` 都返回 ``light``，
       所以只能当作 ``theme.base = "auto"`` 时的参考。

    取不到时保守返回浅色。
    """
    try:
        base = st.get_option("theme.base")
        if isinstance(base, str) and base.strip().lower() in {"light", "dark"}:
            return base.strip().lower() == "dark"
    except Exception:
        pass

    try:
        theme = st.context.theme
        kind = theme.get("type") if hasattr(theme, "get") else getattr(theme, "type", None)
        return str(kind).strip().lower() == "dark"
    except Exception:
        return False


_IS_DARK = _detect_dark()
# 图表主题与页面主题保持一致，避免"深底浅图"的割裂感
set_theme(_IS_DARK)

# 热图配色：浅色底在深色页面上会像一块发光的白板，所以两套各给一份。
# 深色版把三个端点都压成深色调，文字改用浅色，才不会出现"白底白字"。
if _IS_DARK:
    _CMAP_RATE = [[0, "#4A1F28"], [0.5, "#3B2F1B"], [1, "#0F3B35"]]
    _CMAP_CORR = [[0, "#4A2030"], [0.5, "#161628"], [1, "#232A66"]]
    _CMAP_Q = [[0, "#14141F"], [1, "#0D9488"]]
    _HEAT_TEXT = "#ECECF9"
else:
    _CMAP_RATE = [[0, "#FEECEF"], [0.5, "#FEF4E6"], [1, "#E6F7F3"]]
    _CMAP_CORR = [[0, "#F6C9D3"], [0.5, "#FBFBFE"], [1, "#B9BEF5"]]
    _CMAP_Q = [[0, "#F2F2F9"], [1, "#0D9488"]]
    _HEAT_TEXT = "#2A2A44"

# ---------------------------------------------------------------------- #
# 全局样式
# ---------------------------------------------------------------------- #
#: 浅色设计令牌
_CSS_TOKENS_LIGHT = """
    :root{
      --cdc-canvas:#F6F7FB;
      --cdc-surface:#FFFFFF;
      --cdc-surface-2:#F2F2F8;
      --cdc-surface-3:#FAFAFF;
      --cdc-line:#E8E8F2;
      --cdc-line-soft:#F1F1F8;
      --cdc-line-strong:#DBDBEE;
      --cdc-hover:#F1F1F9;
      --cdc-ink:#14142B;
      --cdc-ink2:#52526B;
      --cdc-ink3:#8A8AA0;
      --cdc-brand:#4F46E5;
      --cdc-brand-d:#4338CA;
      --cdc-brand-l:#EEF0FF;
      --cdc-brand-bd:#DFE1FF;
      --cdc-teal:#0D9488;
      --cdc-teal-l:#E6F7F3;
      --cdc-teal-ink:#0A5F55;
      --cdc-teal-bd:#7FD9C8;
      --cdc-amber:#D97706;
      --cdc-amber-l:#FEF4E6;
      --cdc-amber-ink:#8A5300;
      --cdc-rose:#E11D48;
      --cdc-rose-l:#FEECEF;
      --cdc-rose-ink:#9F1239;
      --cdc-rose-bd:#F0A0B4;
      --cdc-track:#EDEDF6;
      --cdc-scroll:#D6D6E6;
      --cdc-scroll-h:#BCBCD4;
      --cdc-glow-a:#EEF0FF;
      --cdc-glow-a0:rgba(238,240,255,0);
      --cdc-glow-b:#E9F7F3;
      --cdc-glow-b0:rgba(233,247,243,0);
      --cdc-glow-c:rgba(99,102,241,.15);
      --cdc-sidebar-a:#FFFFFF;
      --cdc-sidebar-b:#FAFAFE;
      --cdc-hero-a:#FFFFFF;
      --cdc-hero-b:#F7F7FF;
      --cdc-hero-c:#F0FBF8;
      --cdc-note-bg:#FAFAFF;
      --cdc-note-good:#F5FCFA;
      --cdc-note-warn:#FFFBF5;
      --cdc-note-bad:#FFF7F9;
      --cdc-grad-mid:#8B5CF6;
      --cdc-sh1:0 1px 2px rgba(20,20,43,.05);
      --cdc-sh2:0 2px 6px rgba(20,20,43,.05), 0 12px 28px rgba(20,20,43,.06);
      --cdc-r1:10px; --cdc-r2:14px; --cdc-r3:18px;
    }
"""

#: 深色设计令牌（同一套变量名，只换取值，样式表无需改一处）
_CSS_TOKENS_DARK = """
    :root{
      --cdc-canvas:#0B0B14;
      --cdc-surface:#15152B;
      --cdc-surface-2:#1D1D36;
      --cdc-surface-3:#12122A;
      --cdc-line:#282842;
      --cdc-line-soft:#202038;
      --cdc-line-strong:#34345A;
      --cdc-hover:#1E1E38;
      --cdc-ink:#ECECF9;
      --cdc-ink2:#B6B6D2;
      --cdc-ink3:#8C8CB0;
      --cdc-brand:#8F8BF7;
      --cdc-brand-d:#B6B2FF;
      --cdc-brand-l:rgba(99,102,241,.20);
      --cdc-brand-bd:rgba(143,139,247,.42);
      --cdc-teal:#2DD4BF;
      --cdc-teal-l:rgba(13,148,136,.22);
      --cdc-teal-ink:#6FE9D8;
      --cdc-teal-bd:rgba(45,212,191,.45);
      --cdc-amber:#FBBF24;
      --cdc-amber-l:rgba(217,119,6,.22);
      --cdc-amber-ink:#FCD34D;
      --cdc-rose:#FB7185;
      --cdc-rose-l:rgba(225,29,72,.22);
      --cdc-rose-ink:#FDA4AF;
      --cdc-rose-bd:rgba(251,113,133,.45);
      --cdc-track:#23233C;
      --cdc-scroll:#34345A;
      --cdc-scroll-h:#46466F;
      --cdc-glow-a:#1B1740;
      --cdc-glow-a0:rgba(27,23,64,0);
      --cdc-glow-b:#0B3630;
      --cdc-glow-b0:rgba(11,54,48,0);
      --cdc-glow-c:rgba(139,92,246,.22);
      --cdc-sidebar-a:#0F0F1E;
      --cdc-sidebar-b:#0A0A14;
      --cdc-hero-a:#15152B;
      --cdc-hero-b:#191936;
      --cdc-hero-c:#0F2A28;
      --cdc-note-bg:#15152B;
      --cdc-note-good:#0E2422;
      --cdc-note-warn:#241D10;
      --cdc-note-bad:#26121A;
      --cdc-grad-mid:#A78BFA;
      --cdc-sh1:0 1px 2px rgba(0,0,0,.40);
      --cdc-sh2:0 2px 8px rgba(0,0,0,.44), 0 14px 32px rgba(0,0,0,.42);
      --cdc-r1:10px; --cdc-r2:14px; --cdc-r3:18px;
    }
"""

_CSS_BASE = """

    /* ================= 基础排版 ================= */
    html, body, .stApp, [class*="css"], button, input, select, textarea{
      font-family:"Inter","SF Pro Text","PingFang SC","Hiragino Sans GB",
                  "Microsoft YaHei",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif !important;
      -webkit-font-smoothing:antialiased;
      text-rendering:optimizeLegibility;
    }
    .stApp{
      background:
        radial-gradient(1250px 520px at 10% -10%, var(--cdc-glow-a) 0%, var(--cdc-glow-a0) 62%),
        radial-gradient(900px 430px at 92% 0%, var(--cdc-glow-b) 0%, var(--cdc-glow-b0) 58%),
        var(--cdc-canvas) !important;
    }
    .block-container{ padding-top:1.9rem; padding-bottom:3.6rem; max-width:1420px; }
    header[data-testid="stHeader"]{ background:transparent; }

    h1,h2,h3,h4{ color:var(--cdc-ink); letter-spacing:-.015em; }
    h1{ font-size:1.72rem !important; font-weight:700 !important; }
    h2{ font-size:1.23rem !important; font-weight:650 !important; margin-top:1.7rem !important; }
    h3{ font-size:1.02rem !important; font-weight:650 !important; margin-top:1.2rem !important; }
    p, li{ color:var(--cdc-ink2); }
    a{ color:var(--cdc-brand); text-decoration:none; }
    a:hover{ text-decoration:underline; }
    hr{ border:none; border-top:1px solid var(--cdc-line); margin:1.4rem 0; }

    /* 滚动条 */
    ::-webkit-scrollbar{ width:9px; height:9px; }
    ::-webkit-scrollbar-track{ background:transparent; }
    ::-webkit-scrollbar-thumb{
      background:var(--cdc-scroll); border-radius:6px;
      border:2px solid transparent; background-clip:content-box;
    }
    ::-webkit-scrollbar-thumb:hover{ background:var(--cdc-scroll-h); background-clip:content-box; }

    /* ================= 指标卡 ================= */
    div[data-testid="stMetric"]{
      position:relative; overflow:hidden;
      background:var(--cdc-surface);
      border:1px solid var(--cdc-line);
      border-radius:var(--cdc-r2);
      padding:16px 18px 14px 18px;
      box-shadow:var(--cdc-sh1);
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    div[data-testid="stMetric"]::before{
      content:""; position:absolute; top:0; left:0; height:3px; width:100%;
      background:linear-gradient(90deg,var(--cdc-brand) 0%,var(--cdc-grad-mid) 52%,var(--cdc-teal) 100%);
    }
    div[data-testid="stMetric"]:hover{
      transform:translateY(-2px);
      box-shadow:var(--cdc-sh2);
      border-color:var(--cdc-line-strong);
    }
    div[data-testid="stMetricValue"]{
      font-size:1.58rem !important; font-weight:700 !important;
      letter-spacing:-.02em; color:var(--cdc-ink) !important;
      font-variant-numeric:tabular-nums;
    }
    div[data-testid="stMetricLabel"] p{
      font-size:.795rem !important; font-weight:500 !important;
      color:var(--cdc-ink3) !important; letter-spacing:.01em;
    }
    div[data-testid="stMetricDelta"]{ font-size:.78rem !important; font-weight:600 !important; }

    /* ================= 按钮 ================= */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button{
      border-radius:10px !important;
      font-weight:600 !important; letter-spacing:.005em;
      transition:transform .16s ease, box-shadow .16s ease, filter .16s ease,
                 border-color .16s ease, color .16s ease;
    }
    /* 让按钮内层文字（markdown 段落）继承按钮自身颜色，
       否则会被全局 p 颜色规则覆盖，导致主按钮白字变深色 */
    div[data-testid="stButton"] > button *,
    div[data-testid="stDownloadButton"] > button *,
    div[data-testid="stFormSubmitButton"] > button *{ color:inherit !important; }
    button[data-testid="stBaseButton-primary"],
    div[data-testid="stButton"] > button[kind="primary"]{
      background:linear-gradient(135deg,#5B54EC 0%,#4338CA 100%) !important;
      border:none !important; color:#FFFFFF !important;
      box-shadow:0 2px 8px rgba(79,70,229,.26), 0 1px 2px rgba(20,20,43,.07);
    }
    button[data-testid="stBaseButton-primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:hover{
      transform:translateY(-1px); filter:saturate(1.08);
      box-shadow:0 9px 24px rgba(79,70,229,.34);
    }
    button[data-testid="stBaseButton-secondary"],
    div[data-testid="stButton"] > button[kind="secondary"]{
      background:var(--cdc-surface) !important;
      border:1px solid var(--cdc-line) !important;
      color:var(--cdc-ink2) !important;
      box-shadow:var(--cdc-sh1);
    }
    button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stButton"] > button[kind="secondary"]:hover{
      border-color:var(--cdc-line-strong) !important; color:var(--cdc-brand) !important;
      transform:translateY(-1px); box-shadow:var(--cdc-sh2);
    }

    /* ================= 标签页 ================= */
    div[data-testid="stTabs"] div[data-baseweb="tab-list"]{
      gap:4px; border-bottom:1px solid var(--cdc-line);
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]{
      border-radius:9px 9px 0 0 !important;
      padding:8px 15px !important;
      font-weight:600 !important; color:var(--cdc-ink3) !important;
      transition:color .15s ease, background .15s ease;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover{
      background:var(--cdc-hover) !important; color:var(--cdc-ink) !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]{
      color:var(--cdc-brand-d) !important; background:var(--cdc-brand-l) !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"]{
      background:var(--cdc-brand) !important; height:2px !important; border-radius:2px;
    }

    /* ================= 折叠面板 ================= */
    details[data-testid="stExpander"]{
      border:1px solid var(--cdc-line) !important;
      border-radius:var(--cdc-r2) !important;
      background:var(--cdc-surface);
      box-shadow:var(--cdc-sh1);
      overflow:hidden;
      transition:box-shadow .18s ease, border-color .18s ease;
    }
    details[data-testid="stExpander"]:hover{
      box-shadow:var(--cdc-sh2); border-color:var(--cdc-line-strong) !important;
    }
    details[data-testid="stExpander"] summary{ font-weight:600; color:var(--cdc-ink); }
    div[data-testid="stExpanderDetails"]{
      border-top:1px solid var(--cdc-line-soft);
      padding:14px 16px 16px 16px;
    }

    /* ================= 提示 / 表格 ================= */
    div[data-testid="stAlert"]{
      border-radius:var(--cdc-r1); border:1px solid var(--cdc-line);
      box-shadow:var(--cdc-sh1);
    }
    div[data-testid="stDataFrame"]{
      border:1px solid var(--cdc-line); border-radius:var(--cdc-r1);
      overflow:hidden; box-shadow:var(--cdc-sh1);
    }

    /* ================= 表单控件 ================= */
    div[data-testid="stWidgetLabel"] p{
      font-size:.845rem !important; font-weight:600 !important;
      color:var(--cdc-ink2) !important;
    }
    div[data-baseweb="input"], div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"], div[data-baseweb="base-input"]{
      border-radius:10px !important;
      border-color:var(--cdc-line) !important;
      transition:border-color .15s ease, box-shadow .15s ease;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"]:focus-within{
      border-color:var(--cdc-brand) !important;
      box-shadow:0 0 0 3px rgba(79,70,229,.12) !important;
    }

    /* ================= 侧边栏 ================= */
    section[data-testid="stSidebar"]{
      background:linear-gradient(180deg,var(--cdc-sidebar-a) 0%,var(--cdc-sidebar-b) 100%);
      border-right:1px solid var(--cdc-line);
      min-width:312px;
    }
    div[data-testid="stSidebarContent"]{ padding-left:1rem; padding-right:1rem; }
    section[data-testid="stSidebar"] hr{ margin:.95rem 0; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3{
      font-size:1rem !important; color:var(--cdc-ink) !important;
      margin-bottom:.25rem !important; letter-spacing:-.01em;
    }
    section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p{
      font-size:.78rem !important; color:var(--cdc-ink3) !important;
    }

    /* 品牌区 */
    .cdc-brandbox{
      position:relative; overflow:hidden;
      background:linear-gradient(135deg,#312E81 0%,#4338CA 50%,#6D28D9 100%);
      border-radius:var(--cdc-r2);
      padding:16px 18px 15px 18px;
      margin:0 0 14px 0;
      box-shadow:0 12px 28px rgba(67,56,202,.26);
    }
    .cdc-brandbox::after{
      content:""; position:absolute; width:160px; height:160px; right:-52px; top:-62px;
      background:radial-gradient(circle,rgba(255,255,255,.36),rgba(255,255,255,0) 68%);
      border-radius:50%; pointer-events:none;
    }
    .cdc-brandbox .t{
      color:#fff; font-size:1.03rem; font-weight:700;
      letter-spacing:-.012em; position:relative;
    }
    .cdc-brandbox .s{
      color:rgba(255,255,255,.8); font-size:.755rem;
      margin-top:4px; line-height:1.5; position:relative;
    }
    .cdc-brandbox .b{ margin-top:10px; position:relative; }
    .cdc-brandbox .b span{
      display:inline-block; font-size:.675rem; padding:2px 8px;
      border-radius:999px; margin-right:5px;
      background:rgba(255,255,255,.17); color:#EFEDFF;
      border:1px solid rgba(255,255,255,.22);
    }

    /* 导航（radio 改造成卡片式）
       注：Streamlit 1.65 的 radio 基于 react-aria —— 分组下是 wrapper div 包 label，
       且选中项带 data-selected="true"（无需 :has）。选择器按真实 DOM 编写。 */
    /* 元素容器默认是 fit-content，需强制撑满侧边栏，否则底色只有一小块 */
    .st-key-nav,
    .st-key-nav .stRadio,
    .st-key-nav div[data-testid="stRadioGroup"]{ width:100% !important; }
    .st-key-nav div[data-testid="stRadioGroup"]{ gap:5px; }
    .st-key-nav div[data-testid="stRadioGroup"] > div{ width:100%; }
    .st-key-nav label[data-testid="stRadioOption"]{
      display:flex; align-items:center; box-sizing:border-box;
      width:100%; margin:0; padding:8px 12px;
      border:1px solid transparent; border-radius:10px;
      cursor:pointer;
      transition:background .16s ease, box-shadow .16s ease, border-color .16s ease;
    }
    .st-key-nav label[data-testid="stRadioOption"] p{
      font-size:.875rem !important; font-weight:550 !important;
      color:var(--cdc-ink2) !important; transition:color .16s ease;
    }
    .st-key-nav label[data-testid="stRadioOption"]:hover{ background:var(--cdc-hover); }
    .st-key-nav label[data-testid="stRadioOption"]:hover p{ color:var(--cdc-ink) !important; }

    /* 选中态 */
    .st-key-nav label[data-testid="stRadioOption"][data-selected="true"]{
      background:var(--cdc-brand-l);
      border-color:var(--cdc-brand-bd);
      box-shadow:inset 3px 0 0 var(--cdc-brand), 0 1px 3px rgba(79,70,229,.10);
    }
    .st-key-nav label[data-testid="stRadioOption"][data-selected="true"] p{
      color:var(--cdc-brand-d) !important; font-weight:700 !important;
    }
    /* 隐藏单选框圆点（结构：label > 内容div > 圆点div + markdown）。
       若 DOM 结构不同则自动保留圆点，不影响可用性。 */
    .st-key-nav label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]){
      display:none !important;
    }
    /* 键盘可访问性：聚焦时给出清晰轮廓 */
    .st-key-nav label[data-testid="stRadioOption"]:focus-within{
      outline:2px solid var(--cdc-brand); outline-offset:1px;
    }

    /* 题库来源：与侧边栏其他表单一一致的轻量卡片样式 */
    .st-key-bank_source div[role="radiogroup"]{ gap:2px; }

    /* ================= Hero 区 ================= */
    .cdc-hero{
      position:relative; overflow:hidden;
      background:linear-gradient(120deg,var(--cdc-hero-a) 0%,var(--cdc-hero-b) 46%,var(--cdc-hero-c) 100%);
      border:1px solid var(--cdc-line);
      border-radius:var(--cdc-r3);
      padding:26px 30px 24px 30px;
      margin:0 0 20px 0;
      box-shadow:var(--cdc-sh2);
    }
    .cdc-hero::before{
      content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
      background:linear-gradient(180deg,var(--cdc-brand) 0%,var(--cdc-grad-mid) 52%,var(--cdc-teal) 100%);
    }
    .cdc-hero::after{
      content:""; position:absolute; width:300px; height:300px; right:-100px; top:-140px;
      background:radial-gradient(circle,rgba(99,102,241,.15),rgba(99,102,241,0) 66%);
      border-radius:50%; pointer-events:none;
    }
    .cdc-hero .kicker{
      display:inline-block; font-size:.685rem; font-weight:700;
      letter-spacing:.15em; text-transform:uppercase;
      color:var(--cdc-brand); background:var(--cdc-brand-l);
      border-radius:999px; padding:3px 10px; margin-bottom:10px;
      position:relative;
    }
    .cdc-hero h1{ margin:0 0 7px 0 !important; font-size:1.6rem !important; position:relative; }
    .cdc-hero p{
      margin:0; color:var(--cdc-ink2); font-size:.925rem;
      line-height:1.64; max-width:80ch; position:relative;
    }
    .cdc-hero .tags{ margin-top:13px; position:relative; }

    /* ================= 区块小标题 ================= */
    .cdc-sec{ display:flex; align-items:center; gap:10px; margin:26px 0 12px 0; }
    .cdc-sec .bar{
      width:3px; height:17px; border-radius:2px; flex:none;
      background:linear-gradient(180deg,var(--cdc-brand),var(--cdc-grad-mid));
    }
    .cdc-sec .tx{
      font-size:1rem; font-weight:700; color:var(--cdc-ink); letter-spacing:-.01em;
    }
    .cdc-sec .hint{
      margin-left:auto; font-size:.775rem; color:var(--cdc-ink3); font-weight:500;
    }
    /* markdown 三级标题：与 .cdc-sec 视觉统一（左侧品牌渐变竖条） */
    section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] h3{
      display:flex; align-items:center; gap:10px;
      margin:26px 0 12px 0; padding:0;
      font-size:1rem; font-weight:700; letter-spacing:-.01em;
      color:var(--cdc-ink);
    }
    section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] h3::before{
      content:""; flex:none; width:3px; height:17px; border-radius:2px;
      background:linear-gradient(180deg,var(--cdc-brand),var(--cdc-grad-mid));
    }

    /* ================= 文本 / 标签 ================= */
    .cdc-caption{ color:var(--cdc-ink3); font-size:.855rem; line-height:1.6; }

    .cdc-tag{
      display:inline-block; padding:3px 11px; border-radius:999px;
      background:var(--cdc-brand-l); color:var(--cdc-brand-d);
      font-size:.755rem; font-weight:600; margin-right:6px;
    }
    .cdc-tag-warn{ background:var(--cdc-amber-l); color:var(--cdc-amber-ink); }
    .cdc-tag-ok  { background:var(--cdc-teal-l);  color:var(--cdc-teal-ink); }

    /* 属性标签 */
    .cdc-skill-yes, .cdc-skill-no, .cdc-skill-err{
      display:inline-block; padding:3px 12px; border-radius:999px;
      font-size:.79rem; font-weight:550; margin:2px;
    }
    .cdc-skill-yes{
      background:var(--cdc-teal-l); color:var(--cdc-teal-ink); border:1px solid var(--cdc-teal-bd);
    }
    .cdc-skill-no{
      background:var(--cdc-surface-2); color:var(--cdc-ink2); border:1px solid var(--cdc-line);
    }
    .cdc-skill-err{
      background:var(--cdc-rose-l); color:var(--cdc-rose-ink); border:1px dashed var(--cdc-rose-bd);
    }

    /* 判定条 */
    .cdc-bar-wrap{ margin:7px 0; }
    .cdc-bar-label{
      display:flex; justify-content:space-between;
      font-size:.845rem; margin-bottom:4px; color:var(--cdc-ink2);
    }
    .cdc-bar-label span:last-child{ font-weight:650; color:var(--cdc-ink); }
    .cdc-bar-track{
      background:var(--cdc-track); border-radius:999px; height:8px; overflow:hidden;
    }
    .cdc-bar-fill{
      height:100%; border-radius:999px;
      box-shadow:inset 0 -1px 0 rgba(0,0,0,.06);
      transition:width .4s cubic-bezier(.4,0,.2,1);
    }

    /* 结论框 */
    .cdc-note{
      border-left:3px solid var(--cdc-brand); background:var(--cdc-note-bg);
      padding:12px 15px; border-radius:0 10px 10px 0;
      font-size:.885rem; line-height:1.62; color:var(--cdc-ink2);
    }
    .cdc-note-good{ border-left-color:var(--cdc-teal); background:var(--cdc-note-good); }
    .cdc-note-warn{ border-left-color:var(--cdc-amber); background:var(--cdc-note-warn); }
    .cdc-note-bad { border-left-color:var(--cdc-rose); background:var(--cdc-note-bad); }

    /* 功能卡片 */
    .cdc-feat{
      height:100%;
      background:var(--cdc-surface);
      border:1px solid var(--cdc-line);
      border-radius:var(--cdc-r2);
      padding:18px 20px 18px 20px;
      box-shadow:var(--cdc-sh1);
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .cdc-feat:hover{
      transform:translateY(-3px);
      box-shadow:var(--cdc-sh2);
      border-color:var(--cdc-line-strong);
    }
    .cdc-feat .ic{
      display:inline-flex; align-items:center; justify-content:center;
      width:30px; height:30px; border-radius:9px; margin-bottom:11px;
      font-size:.85rem; font-weight:700; color:#fff;
      background:linear-gradient(135deg,#5B54EC,#4338CA);
      box-shadow:0 3px 10px rgba(79,70,229,.28);
    }
    .cdc-feat .h{ font-size:.95rem; font-weight:700; color:var(--cdc-ink); margin-bottom:6px; }
    .cdc-feat .d{ font-size:.845rem; line-height:1.62; color:var(--cdc-ink2); }

    /* ================= 数据表（条件着色） ================= */
    .cdc-tbl-wrap{
      overflow:auto; max-height:430px;
      border:1px solid var(--cdc-line); border-radius:var(--cdc-r1);
      background:var(--cdc-surface); box-shadow:var(--cdc-sh1);
    }
    table.cdc-tbl{
      width:100%; border-collapse:separate; border-spacing:0;
      font-size:.83rem; color:var(--cdc-ink2);
    }
    table.cdc-tbl thead th{
      position:sticky; top:0; z-index:2;
      background:var(--cdc-surface-2); color:var(--cdc-ink3);
      font-size:.755rem; font-weight:700; letter-spacing:.02em;
      text-align:left; white-space:nowrap;
      padding:9px 13px; border-bottom:1px solid var(--cdc-line);
    }
    table.cdc-tbl tbody td{
      padding:7px 13px; border-bottom:1px solid var(--cdc-line-soft);
      font-variant-numeric:tabular-nums;
    }
    table.cdc-tbl tbody tr:last-child td{ border-bottom:none; }
    table.cdc-tbl tbody tr:hover td{ background:var(--cdc-hover); }
    table.cdc-tbl td.num{ text-align:right; }
    table.cdc-tbl td.mono{
      font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      font-size:.775rem; color:var(--cdc-ink);
    }
    table.cdc-tbl td.strong{ color:var(--cdc-ink); font-weight:600; }

    /* 热力单元格：底色由语义决定（好=青绿 / 差=玫红），透明度表强弱 */
    .cdc-heat{
      display:inline-block; min-width:52px; text-align:right;
      padding:1px 7px; border-radius:5px; font-variant-numeric:tabular-nums;
      color:var(--cdc-ink); font-weight:600;
    }
    .cdc-pill{
      display:inline-block; padding:2px 9px; border-radius:999px;
      font-size:.735rem; font-weight:650; white-space:nowrap;
    }
    .cdc-pill-ok{ background:var(--cdc-teal-l); color:var(--cdc-teal-ink); }
    .cdc-pill-warn{ background:var(--cdc-amber-l); color:var(--cdc-amber-ink); }
    .cdc-pill-bad{ background:var(--cdc-rose-l); color:var(--cdc-rose-ink); }
    .cdc-pill-mute{ background:var(--cdc-surface-2); color:var(--cdc-ink3); }

    /* ================= 小色谱条 ================= */
    .cdc-scale{
      display:flex; align-items:center; gap:9px;
      font-size:.735rem; color:var(--cdc-ink3); margin:2px 0 9px 0;
    }
    .cdc-scale .ramp{
      flex:1; max-width:190px; height:7px; border-radius:999px;
      background:linear-gradient(90deg,
        rgba(225,29,72,.55) 0%, rgba(217,119,6,.45) 46%, rgba(13,148,136,.55) 100%);
    }
"""

#: 深色模式下的少量"抵消式"补丁：
#: Streamlit 自身控件（代码块、下拉浮层、滑块）由主题接管，
#: 这里只处理被本文件写死的少数浅色残留。
_CSS_DARK_PATCH = """
    <style>
    /* 折叠面板箭头、下拉浮层阴影在深色下需要提亮 */
    div[data-baseweb="popover"] li:hover{ background:var(--cdc-hover) !important; }
    div[data-testid="stDataFrame"]{ border-color:var(--cdc-line); }
    /* 主按钮在深色下降低发光强度，避免刺眼 */
    button[data-testid="stBaseButton-primary"]{
      box-shadow:0 2px 10px rgba(79,70,229,.34) !important;
    }
    /* 内联代码块 */
    code{
      background:var(--cdc-surface-2) !important; color:var(--cdc-brand-d) !important;
    }
    </style>
"""

st.markdown(
    "<style>"
    + (_CSS_TOKENS_DARK if _IS_DARK else _CSS_TOKENS_LIGHT)
    + _CSS_BASE
    + "</style>",
    unsafe_allow_html=True,
)
if _IS_DARK:
    st.markdown(_CSS_DARK_PATCH, unsafe_allow_html=True)

CRITERIA = ["PWKL", "MPWKL", "KL", "GDI", "SHE", "CDI", "MIV", "NPSHE", "ERHD", "AMI"]
MODELS = ["DINA", "DINO", "G-DINA", "RRUM", "ACDM", "LLM"]
ESTIMATORS = ["MAP", "EAP", "MLE"]
EXPOSURE_METHODS = ["无控制", "曝光硬上限", "随机化选题", "分层曝光", "Sympson-Hetter"]


def _esc(text: object) -> str:
    """HTML 转义，避免属性名等用户输入破坏页面结构。"""
    return _html.escape(str(text), quote=False)


def build_exposure_control(name: str, n_items: int, max_rate: float = 0.2,
                           top_n: int = 5, penalty: float = 0.5):
    """按 UI 选择构造曝光控制器。"""
    if name == "曝光硬上限":
        return MaxExposureLimit(n_items, max_rate=max_rate, warmup=int(n_items * 0.3))
    if name == "随机化选题":
        return RandomesqueExposure(n_items, top_n=int(top_n))
    if name == "分层曝光":
        return StratifiedExposure(n_items, penalty=penalty, warmup=int(n_items * 0.3))
    if name == "Sympson-Hetter":
        return SympsonHetter(n_items, target_rate=max_rate)
    return NoExposureControl(n_items)


# ---------------------------------------------------------------------- #
# 会话状态
# ---------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "bank": None,
        "admin_state": None,
        "sim_results": None,
        "compare_results": None,
        "calib_result": None,
        "item_analysis": None,
        "bank_fingerprint": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------------- #
# 侧边栏：题库与模型配置
# ---------------------------------------------------------------------- #
def sidebar_bank_selector() -> dict:
    """侧边栏题库选择与模型配置，返回配置字典。"""
    st.sidebar.markdown(
        """
        <div class="cdc-brandbox">
            <div class="t">CD-CAT 选题与仿真系统</div>
            <div class="s">认知诊断计算机自适应测验<br>以更少的题目，精准识别知识结构</div>
            <div class="b"><span>DINA</span><span>G-DINA</span><span>PWKL</span><span>Q3</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("1. 选择题库")
    source = st.sidebar.radio("题库来源", ["内置题库", "上传 CSV", "导入 JSON"], key="bank_source")

    bank = None
    if source == "内置题库":
        preset = st.sidebar.selectbox("内置题库", list_presets())
        bank = build_preset_bank(preset)
    elif source == "上传 CSV":
        up = st.sidebar.file_uploader("Q 矩阵 CSV（行为题目，列为属性）", type=["csv"])
        if up is not None:
            from demo.bank_utils import load_qmatrix_from_upload

            try:
                q = load_qmatrix_from_upload(up)
                rng = np.random.default_rng(0)
                bank = {
                    "q": q,
                    "slip": rng.uniform(0.05, 0.25, q.n_items),
                    "guess": rng.uniform(0.05, 0.25, q.n_items),
                    "name": up.name,
                    "description": "上传的 Q 矩阵（题目参数为随机生成）",
                    "attrs": q.attr_names,
                }
                st.sidebar.success(f"已读取 {q.n_items} x {q.n_attrs} 的 Q 矩阵")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"读取失败：{exc}")
    else:
        txt = st.sidebar.text_area("粘贴题库 JSON", height=120)
        if txt.strip():
            try:
                bank = bank_from_json(txt)
                st.sidebar.success("题库已导入")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"解析失败：{exc}")

    if bank is None:
        st.sidebar.info("请选择或导入一个题库")
        return {}

    # 题库切换时清空依赖题库的缓存结果
    fingerprint = (bank["name"], bank["q"].n_items, bank["q"].n_attrs)
    if st.session_state.get("bank_fingerprint") != fingerprint:
        st.session_state["bank_fingerprint"] = fingerprint
        st.session_state["item_analysis"] = None
        st.session_state["admin_state"] = None
        st.session_state["sim_results"] = None
        st.session_state["compare_results"] = None

    st.session_state["bank"] = bank

    st.sidebar.header("2. 题目参数")
    mode = st.sidebar.radio(
        "参数来源",
        ["使用题库预设参数", "手动统一设定", "由 EM 标定"],
        index=0,
    )
    if mode == "手动统一设定":
        slip_v = st.sidebar.slider("失误概率 slip", 0.01, 0.45, 0.15, 0.01)
        guess_v = st.sidebar.slider("猜测概率 guess", 0.01, 0.45, 0.15, 0.01)
        bank["slip"] = np.full(bank["q"].n_items, slip_v)
        bank["guess"] = np.full(bank["q"].n_items, guess_v)
        st.sidebar.caption("统一参数用于快速演示，真实应用请用标定结果")
    elif mode == "由 EM 标定":
        st.sidebar.caption("请先到「参数标定」页完成标定")

    if mode == "由 EM 标定" and "calibrated_params" in st.session_state:
        cp = st.session_state["calibrated_params"]
        if len(cp["slip"]) == bank["q"].n_items:
            bank["slip"] = cp["slip"]
            bank["guess"] = cp["guess"]
            st.sidebar.success("已应用 EM 标定参数")

    st.sidebar.header("3. 响应模型")
    model = st.sidebar.selectbox("模型", MODELS, index=0)
    if model in ("G-DINA", "RRUM", "ACDM", "LLM"):
        st.sidebar.caption(
            "该模型的原始参数更多，此处由 DINA 的 slip/guess 投影生成"
        )

    # ---- 当前配置速览 ----
    st.sidebar.divider()
    st.sidebar.caption("当前配置")
    st.sidebar.markdown(
        f'<span class="cdc-tag">{bank["q"].n_items} 题</span>'
        f'<span class="cdc-tag">{bank["q"].n_attrs} 属性</span>'
        f'<span class="cdc-tag">{bank["q"].n_patterns} 模式</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        f"平均 slip={np.mean(bank['slip']):.3f}　平均 guess={np.mean(bank['guess']):.3f}"
    )

    return {"bank": bank, "model": model}


# ---------------------------------------------------------------------- #
# 工具：绘图
# ---------------------------------------------------------------------- #
def plot_posterior_bar(posterior: np.ndarray, q: QMatrix, top_n: int = 12) -> go.Figure:
    """后验分布条形图（只显示概率最高的 top_n 个模式）。"""
    order = np.argsort(-posterior)[:top_n]
    patterns = q.attribute_patterns()
    labels = [pattern_to_str(patterns[i]) for i in order]
    values = posterior[order]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color="#7C3AED"),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=max(260, 26 * len(labels)),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="后验概率",
        yaxis=dict(autorange="reversed", title="属性掌握模式"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def plot_marginal_convergence(marginal_hist: np.ndarray, attrs: list) -> go.Figure:
    """属性边际概率随题量的变化曲线。"""
    fig = go.Figure()
    palette = ["#4F46E5", "#0D9488", "#E11D48", "#2563EB", "#D97706", "#DB2777"]
    for k in range(marginal_hist.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(marginal_hist))),
                y=marginal_hist[:, k],
                mode="lines+markers",
                name=attrs[k] if k < len(attrs) else f"A{k+1}",
                line=dict(color=palette[k % len(palette)], width=2),
                marker=dict(size=5),
            )
        )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#B4B4C7")
    fig.add_hline(y=0.8, line_dash="dot", line_color="#14B8A6")
    fig.add_hline(y=0.2, line_dash="dot", line_color="#14B8A6")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="已作答题数",
        yaxis_title="P(掌握该属性)",
        yaxis=dict(range=[-0.02, 1.02]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(size=12),
    )
    return fig


def plot_exposure(usage: np.ndarray, item_names: list) -> go.Figure:
    """题目曝光分布。"""
    fig = go.Figure(
        go.Bar(
            x=list(range(len(usage))),
            y=usage,
            marker=dict(color="#2563EB"),
            hovertext=item_names if item_names else None,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="题目编号",
        yaxis_title="曝光次数",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def plot_convergence(curve: dict) -> go.Figure:
    """PMR / AMR 收敛曲线。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=curve["n_items"], y=curve["pmr"], mode="lines+markers",
                   name="模式判准率 PMR", line=dict(color="#4F46E5", width=2.5))
    )
    fig.add_trace(
        go.Scatter(x=curve["n_items"], y=curve["amr"], mode="lines+markers",
                   name="属性判准率 AMR", line=dict(color="#0D9488", width=2.5))
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="题量",
        yaxis_title="判准率",
        yaxis=dict(range=[0, 1.02]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(size=12),
    )
    return fig


def plot_item_discrimination(analysis, q: QMatrix, top_n: int = 25) -> go.Figure:
    """题目区分度排序条形图（最强 top_n + 最弱 top_n）。"""
    disc = analysis.discrimination
    order = np.argsort(-disc)
    show = list(order[:top_n])
    if len(order) > 2 * top_n:
        show += list(order[-top_n:])
    show = sorted(set(show), key=lambda i: -disc[i])
    names = [q.item_names[i] for i in show]
    vals = [float(disc[i]) for i in show]
    colors = ["#0D9488" if v >= 0.5 else "#E11D48" if v < 0.3 else "#D97706" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals, y=names, orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            hovertext=[f"考查属性: {'+'.join(q.attr_names[k] for k in q.required_attrs(i))}" for i in show],
        )
    )
    fig.update_layout(
        height=max(340, 20 * len(show)),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="区分度 D", range=[0, 1.05]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    return fig


def plot_item_information(analysis, q: QMatrix, top_n: int = 30) -> go.Figure:
    """题目信息量条形图。"""
    info = analysis.information
    order = np.argsort(-info)[:top_n]
    names = [q.item_names[i] for i in order]
    vals = [float(info[i]) for i in order]
    fig = go.Figure(
        go.Bar(
            x=names, y=vals,
            marker=dict(color="#2563EB"),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        )
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="题目（按信息量降序）", tickangle=-45),
        yaxis=dict(title="平均信息量"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    return fig


def plot_pattern_heatmap(result, q: QMatrix, attrs: list) -> go.Figure:
    """真实模式 × 判准率 热力图。"""
    by_pat = result.pmr_by_pattern()
    patterns = q.attribute_patterns()
    idxs = sorted(by_pat.keys())
    labels = [pattern_to_str(patterns[i]) for i in idxs]
    pmr = [by_pat[i]["pmr"] for i in idxs]
    amr = [by_pat[i]["amr"] for i in idxs]

    fig = go.Figure(
        go.Heatmap(
            z=[pmr, amr],
            x=labels,
            y=["PMR", "AMR"],
            colorscale=_CMAP_RATE,
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in pmr], [f"{v:.2f}" for v in amr]],
            texttemplate="%{text}",
            textfont=dict(size=11, color=_HEAT_TEXT),
            colorbar=dict(title="判准率", thickness=12),
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="真实掌握模式", tickangle=-45, showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_identifiability(report) -> go.Figure:
    """可辨识性：2^K 模式中被唯一确定 vs 不可区分的占比。"""
    total = report.n_patterns
    distinct = report.n_distinct_response_vectors
    ambig = total - distinct
    fig = go.Figure(
        go.Pie(
            labels=["可唯一辨识模式", "不可区分模式"],
            values=[distinct, ambig],
            hole=0.55,
            marker=dict(colors=["#0D9488", "#E11D48"]),
            textinfo="label+value+percent",
            textfont=dict(size=12),
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12),
    )
    return fig


def plot_ceiling_gauge(pack: dict) -> go.Figure:
    """实际 PMR 与理论上限的对比条。"""
    ceiling = pack["ceiling"]
    pmr = pack["pmr"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["理论上限", "实际达成"], y=[ceiling, pmr],
        marker=dict(color=["#B4B4C7", "#4F46E5"]),
        text=[f"{ceiling:.3f}", f"{pmr:.3f}"], textposition="outside",
        width=0.5,
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 1.08], title="PMR"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), showlegend=False,
    )
    return fig


def render_bar(label: str, value: float, color: str | None = None) -> str:
    """生成一条 HTML 进度条。"""
    pct = max(0.0, min(1.0, float(value)))
    if color is None:
        color = "#0D9488" if pct >= 0.8 else "#D97706" if pct >= 0.5 else "#E11D48"
    return (
        f'<div class="cdc-bar-wrap">'
        f'<div class="cdc-bar-label"><span>{label}</span><span>{pct:.0%}</span></div>'
        f'<div class="cdc-bar-track"><div class="cdc-bar-fill" '
        f'style="width:{pct*100:.1f}%;background:{color};"></div></div></div>'
    )


def section(title: str, hint: str = "") -> None:
    """区块小标题：左侧渐变竖条 + 标题 + 可选右侧提示。"""
    hint_html = f'<span class="hint">{_esc(hint)}</span>' if hint else ""
    st.markdown(
        f'<div class="cdc-sec"><span class="bar"></span>'
        f'<span class="tx">{_esc(title)}</span>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, desc: str, kicker: str = "", tags: list[str] | None = None) -> None:
    """页面顶部的 Hero 说明区（支持眉标与徽章）。"""
    kick = f'<div class="kicker">{_esc(kicker)}</div>' if kicker else ""
    tag_html = ""
    if tags:
        tag_html = '<div class="tags">' + "".join(
            f'<span class="cdc-tag">{_esc(str(t))}</span>' for t in tags
        ) + "</div>"
    st.markdown(
        f'<div class="cdc-hero">{kick}<h1>{_esc(title)}</h1>'
        f'<p>{_esc(desc)}</p>{tag_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------- #
# 条件着色表格
#
# 用 ``st.dataframe`` 做不了"按数值上色"而不牺牲深浅色适配：Styler 的背景色
# 是写死的浅色渐变，切到深色主题后文字会看不清。这里改用纯 HTML 表格 + 半透明
# 叠加色 —— 底色由语义决定（好=青绿 / 中=琥珀 / 差=玫红），透明度固定，
# 于是叠在浅色或深色卡片上都能读。
# ---------------------------------------------------------------------- #
_HEAT_STOPS = ((225, 29, 72), (217, 119, 6), (13, 148, 136))  # 差 → 中 → 好


def _heat_colour(t: float, alpha: float = 0.30) -> str:
    """把 ``t∈[0,1]`` 映射成三段插值的半透明底色。"""
    t = float(min(max(t, 0.0), 1.0))
    if t >= 0.5:
        (r0, g0, b0), (r1, g1, b1) = _HEAT_STOPS[1], _HEAT_STOPS[2]
        u = (t - 0.5) / 0.5
    else:
        (r0, g0, b0), (r1, g1, b1) = _HEAT_STOPS[0], _HEAT_STOPS[1]
        u = t / 0.5
    rgb = f"{round(r0 + (r1 - r0) * u)},{round(g0 + (g1 - g0) * u)},{round(b0 + (b1 - b0) * u)}"
    return f"rgba({rgb},{alpha:.2f})"


def heat_cell(
    value,
    lo: float = 0.0,
    hi: float = 1.0,
    digits: int = 3,
    invert: bool = False,
    suffix: str = "",
    alpha: float = 0.30,
) -> str:
    """生成一个带热力底色的数值单元格。

    ``invert=True`` 用于"越小越好"的指标（如 slip、SRMSR、|Q3|）。
    缺失值渲染成灰色占位符，不参与着色。
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '<span class="cdc-pill cdc-pill-mute">—</span>'
    if not np.isfinite(v):
        return '<span class="cdc-pill cdc-pill-mute">—</span>'
    span = (hi - lo) or 1.0
    t = min(max((v - lo) / span, 0.0), 1.0)
    if invert:
        t = 1.0 - t
    return (
        f'<span class="cdc-heat" style="background:{_heat_colour(t, alpha)}">'
        f"{v:.{digits}f}{suffix}</span>"
    )


def pill(text: str, kind: str = "mute") -> str:
    """语义徽章：``ok`` / ``warn`` / ``bad`` / ``mute``。"""
    return f'<span class="cdc-pill cdc-pill-{kind}">{_esc(text)}</span>'


def cdc_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_height: int = 430,
    align: list[str] | None = None,
) -> str:
    """把已格式化的单元格拼成一张主题安全的表格（含粘性表头）。

    ``align`` 逐列指定 ``num`` / ``mono`` / ``strong`` / ``""`` 之一，
    统一套在 ``<td class="...">`` 上，避免每行手写 class。
    """
    cls = align or [""] * len(headers)
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{cls[i] or ""}">{c}</td>' for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    return (
        f'<div class="cdc-tbl-wrap" style="max-height:{int(max_height)}px">'
        f"<table class=\"cdc-tbl\"><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def render_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_height: int = 430,
    align: list[str] | None = None,
    caption: str = "",
) -> None:
    """渲染表格（可选左侧图例说明）。"""
    if caption:
        st.caption(caption)
    st.markdown(
        cdc_table(headers, rows, max_height=max_height, align=align),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------- #
# 页面 1：在线施测
# ---------------------------------------------------------------------- #
def page_administer(bank: dict, model_name: str) -> None:
    hero("在线自适应施测", "模拟一名考生逐题作答。每一题都由系统根据当前属性后验动态挑选，"
         "并在右侧实时展示诊断结果如何收敛。",
         kicker="Adaptive Administration", tags=["逐题动态选题", "后验实时收敛", "个人诊断报告"])

    q: QMatrix = bank["q"]

    col1, col2, col3, col4 = st.columns(4)
    selector = col1.selectbox("选题准则", CRITERIA, index=0)
    estimator = col2.selectbox("估计方法", ESTIMATORS, index=0)
    max_items = col3.number_input("最大题量", 5, q.n_items, min(40, q.n_items), 1)
    threshold = col4.slider("终止置信阈值", 0.5, 0.99, 0.80, 0.01)

    mode = st.radio(
        "真实模式设定方式",
        ["随机生成", "手动指定"],
        horizontal=True,
    )
    if mode == "手动指定":
        attrs = bank["attrs"]
        cols = st.columns(min(len(attrs), 6))
        chosen = []
        for i, a in enumerate(attrs):
            chosen.append(cols[i % len(cols)].checkbox(a, value=(i % 2 == 0)))
        true_pattern = np.array([1 if c else 0 for c in chosen], dtype=np.int8)
    else:
        true_pattern = None

    seed = st.number_input("随机种子", 0, 99999, 42, 1)

    if st.button("开始新的施测", type="primary", width="stretch"):
        rng = np.random.default_rng(int(seed))
        if true_pattern is None:
            tp = q.attribute_patterns()[rng.integers(0, q.n_patterns)]
        else:
            tp = true_pattern

        model_params = {"slip": bank["slip"], "guess": bank["guess"]}
        term = CompositeRule(
            [PatternConfidence(threshold=float(threshold))],
            how="any",
            min_items=3,
            max_items=int(max_items),
        )
        sim = CDCATSimulator(
            q,
            model=model_name if model_name in ("DINA", "DINO") else "DINA",
            model_params=model_params,
            selector=selector,
            estimator=estimator,
            termination=term,
            min_items=3,
            max_items=int(max_items),
        )
        record = sim.administer(tp, rng=rng)
        st.session_state["admin_state"] = {
            "record": record,
            "true": tp,
            "sim": sim,
            "selector": selector,
            "estimator": estimator,
            "attrs": bank["attrs"],
        }

    state = st.session_state.get("admin_state")
    if not state:
        st.info("点击上方按钮开始一次施测。")
        return

    rec = state["record"]
    tp = state["true"]
    attrs = state["attrs"]

    st.divider()
    # ---- 指标卡片 ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("作答题目数", rec.n_answered)
    c2.metric("是否完全判对", "是" if rec.pattern_correct else "否")
    c3.metric("属性判准率", f"{rec.attribute_accuracy:.0%}")
    c4.metric("最终后验置信", f"{rec.max_posterior:.3f}")

    colL, colR = st.columns([1, 1])

    with colL:
        section("诊断结果")
        est = rec.est_pattern
        chips_true = "".join(
            f'<span class="cdc-skill-{"yes" if tp[i] else "no"}">{a} {"✓" if tp[i] else "✗"}</span>'
            for i, a in enumerate(attrs)
        )
        st.markdown(f"**真实掌握情况**：{chips_true}", unsafe_allow_html=True)

        chips_est = ""
        for i, a in enumerate(attrs):
            ok = est[i] == tp[i]
            cls = "cdc-skill-yes" if est[i] else "cdc-skill-no"
            mark = "" if ok else " ← 判错"
            chips_est += f'<span class="{cls}">{a} {"✓" if est[i] else "✗"}{mark}</span>'
        st.markdown(f"**系统诊断结果**：{chips_est}", unsafe_allow_html=True)

        st.markdown(
            f"真实模式 `{pattern_to_str(tp)}` 　"
            f"估计模式 `{pattern_to_str(est)}`"
        )
        st.caption(f"终止原因：{rec.stop_reason}")

        section("作答序列")
        df = pd.DataFrame(
            {
                "次序": list(range(1, len(rec.administered) + 1)),
                "题号": [q.item_names[i] for i in rec.administered],
                "考查属性": [
                    "+".join(attrs[k] for k in q.required_attrs(i)) for i in rec.administered
                ],
                "作答": ["正确" if r == 1 else "错误" for r in rec.responses],
            }
        )
        st.dataframe(df, width="stretch", hide_index=True, height=260)

    with colR:
        section("属性掌握概率收敛过程")
        st.plotly_chart(
            plot_marginal_convergence(rec.marginal_history, attrs),
            width="stretch",
        )

        section("最终属性后验分布")
        sim: CDCATSimulator = state["sim"]
        final_res = sim.estimator.estimate(
            np.array(rec.responses, dtype=float),
            np.array(rec.administered, dtype=int),
        )
        st.plotly_chart(
            plot_posterior_bar(final_res.posterior, q),
            width="stretch",
        )
        st.caption(
            f"最大后验概率 {final_res.max_posterior:.4f}；"
            f"属性边际概率 " + ", ".join(
                f"{attrs[i]}={final_res.marginal[i]:.3f}" for i in range(len(attrs))
            )
        )

    # ---- 个人诊断报告 ----
    st.divider()
    section("个人诊断报告")
    st.caption("基于最终属性掌握概率生成的结构化报告，可导出用于反馈给学生或存档。")

    report = build_examinee_report(
        rec, attrs, q.item_names, q, include_truth=True
    )
    marg = report.marginal
    conf = [report.attr_confidence(i) for i in range(len(attrs))]

    bars = "".join(
        render_bar(
            f"{attrs[i]}　P(掌握)={marg[i]:.3f}　置信={conf[i]:.2f}",
            conf[i],
            color="#0D9488" if marg[i] >= 0.5 else "#E11D48",
        )
        for i in range(len(attrs))
    )
    st.markdown(bars, unsafe_allow_html=True)

    low = report.low_confidence_attrs()
    if low:
        low_names = ", ".join(attrs[i] for i in low)
        st.markdown(
            f'<div class="cdc-note cdc-note-warn">以下属性诊断置信度偏低，'
            f'建议加测或人工复核：<b>{low_names}</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cdc-note cdc-note-good">所有属性诊断置信度均达到阈值，'
            '诊断结果可直接采信。</div>',
            unsafe_allow_html=True,
        )

    with st.expander("查看完整文本报告"):
        st.code(report.to_text(), language="text")

    e1, e2, e3 = st.columns(3)
    e1.metric("答题正确数", f"{report.correct_count} / {rec.n_answered}")
    e2.metric("掌握属性数", f"{len(report.mastered)} / {len(attrs)}")
    e3.metric("模式判定", "正确" if report.pattern_correct else "错误")

    r1, r2 = st.columns(2)
    with r1:
        st.download_button(
            "下载报告 (TXT)",
            report.to_text().encode("utf-8-sig"),
            file_name=f"cdcat_examinee_{report.examinee_id}.txt",
            mime="text/plain",
            width="stretch",
        )
    with r2:
        st.download_button(
            "下载报告 (JSON)",
            json.dumps(report.to_dict()).encode("utf-8"),
            file_name=f"cdcat_examinee_{report.examinee_id}.json",
            mime="application/json",
            width="stretch",
        )


# ---------------------------------------------------------------------- #
# 页面 2：蒙特卡洛仿真
# ---------------------------------------------------------------------- #
def page_simulation(bank: dict, model_name: str) -> None:
    hero("蒙特卡洛仿真", "批量模拟大量考生，统计系统整体的诊断准确率、用题效率与题目曝光控制，"
         "并与 Q 矩阵的结构上限对比。",
         kicker="Monte Carlo Simulation", tags=["PMR / AMR", "曝光均匀性", "n_jobs 并行"])

    q: QMatrix = bank["q"]
    c1, c2, c3 = st.columns(3)
    selector = c1.selectbox("选题准则", CRITERIA, index=0, key="sim_sel")
    estimator = c2.selectbox("估计方法", ESTIMATORS, index=0, key="sim_est")
    threshold = c3.slider("终止置信阈值", 0.5, 0.99, 0.80, 0.01, key="sim_thr")

    c4, c5, c6 = st.columns(3)
    n_examinees = c4.number_input("每次重复被试数", 20, 5000, 300, 20)
    n_rep = c5.number_input("重复次数", 1, 50, 5, 1)
    max_items = c6.number_input("最大题量", 5, q.n_items, min(40, q.n_items), 1, key="sim_max")
    seed = st.number_input("随机种子", 0, 99999, 0, 1, key="sim_seed")

    with st.expander("高级设置：终止规则与曝光控制"):
        use_fixed = st.checkbox("改用固定题量（研究收敛曲线）")
        fixed_n = st.number_input("固定题量", 1, q.n_items, min(20, q.n_items), 1, disabled=not use_fixed)
        use_stability = st.checkbox("追加估计稳定性规则", value=False)
        n_stable = st.number_input("连续稳定题数", 2, 20, 5, 1, disabled=not use_stability)

        st.markdown("**题目曝光控制**")
        st.caption("控制高频题目的曝光，兼顾题库安全性与利用率。")
        exp_method = st.selectbox("曝光控制策略", EXPOSURE_METHODS, index=0)
        exp_rate = st.slider("目标最大曝光率", 0.05, 0.6, 0.20, 0.05,
                             disabled=(exp_method != "曝光硬上限" and exp_method != "Sympson-Hetter"))
        exp_topn = st.slider("随机候选池大小", 2, 15, 5, 1, disabled=exp_method != "随机化选题")
        exp_penalty = st.slider("曝光惩罚强度", 0.0, 2.0, 0.5, 0.1, disabled=exp_method != "分层曝光")
        use_sh_calib = st.checkbox(
            "自动标定 Sympson-Hetter K 参数（较慢）",
            value=False, disabled=exp_method != "Sympson-Hetter",
        )

    if st.button("开始仿真", type="primary", width="stretch"):
        model_params = {"slip": bank["slip"], "guess": bank["guess"]}
        eff_model = model_name if model_name in ("DINA", "DINO") else "DINA"
        if use_fixed:
            term = FixedLength(int(fixed_n))
        else:
            rules = [PatternConfidence(threshold=float(threshold))]
            if use_stability:
                rules.append(StabilityRule(n_stable=int(n_stable)))
            term = CompositeRule(rules, how="any", min_items=3, max_items=int(max_items))

        ctrl = build_exposure_control(
            exp_method, q.n_items, exp_rate, exp_topn, exp_penalty
        )
        sim = CDCATSimulator(
            q,
            model=eff_model,
            model_params=model_params,
            selector=selector,
            estimator=estimator,
            termination=term,
            min_items=3,
            max_items=int(max_items),
            exposure_control=ctrl,
        )

        if exp_method == "Sympson-Hetter" and use_sh_calib:
            with st.spinner("Sympson-Hetter 参数标定中…"):
                ctrl.calibrate(
                    sim,
                    n_examinees=min(int(n_examinees), 200),
                    n_replications=2,
                    n_iterations=6,
                    seed=int(seed),
                )

        with st.spinner("仿真进行中…"):
            result = sim.run(
                n_examinees=int(n_examinees),
                n_replications=int(n_rep),
                seed=int(seed),
                progress=False,
            )
        st.session_state["sim_results"] = {
            "result": result, "bank": bank, "selector": selector,
            "estimator": estimator, "model": eff_model, "sim": sim,
            "exposure": exp_method,
        }

    pack = st.session_state.get("sim_results")
    if not pack:
        st.info("点击上方按钮开始仿真。")
        return

    result = pack["result"]
    st.divider()
    ev = Evaluator()
    report = ev.evaluate(result, label=pack["selector"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("模式判准率 PMR", f"{report.pmr:.3f}")
    m2.metric("属性判准率 AMR", f"{report.amr:.3f}")
    m3.metric("平均用题量", f"{report.avg_items:.2f}")
    m4.metric("题库利用率", f"{report.test_utilization:.3f}")
    m5.metric("曝光率标准差", f"{report.exposure_sd:.4f}")

    st.caption(
        f"PMR 95% 置信区间 [{report.pmr_ci[0]:.3f}, {report.pmr_ci[1]:.3f}]　·　"
        f"AMR 95% 置信区间 [{report.amr_ci[0]:.3f}, {report.amr_ci[1]:.3f}]"
    )

    # ---- 与结构上限的对比 ----
    ceiling = result.pmr_vs_ceiling()
    section("诊断精度 vs. 结构上限")
    st.caption("理论上限由 Q 矩阵的结构可辨识性决定 —— 有些模式在数学上就无法区分。"
               "实际 PMR 越接近上限，说明算法越接近该题库的最优表现。")

    g1, g2, g3, g4 = st.columns([1, 1, 1, 1.4])
    with g1:
        st.plotly_chart(plot_ceiling_gauge(ceiling), width="stretch")
    with g2:
        st.metric("理论上限 PMR", f"{ceiling['ceiling']:.3f}")
        st.metric("不可辨识模式数", ceiling["n_unidentifiable"])
    with g3:
        st.metric("实际 PMR", f"{ceiling['pmr']:.3f}")
        st.metric("算法效率", f"{ceiling['efficiency']:.1%}")
    with g4:
        verdict = ceiling["verdict"]
        cls = "cdc-note-good" if ceiling["efficiency"] >= 0.9 else "cdc-note-warn"
        st.markdown(
            f'<div class="cdc-note {cls}" style="margin-top:8px;">'
            f'<b>结论</b><br>{verdict}</div>',
            unsafe_allow_html=True,
        )

    section("各掌握模式的判准率")
    st.caption("深绿=判准率高，浅红=判准率低。若某些模式系统性偏低，"
               "通常对应 Q 矩阵中难以区分的模式。")
    st.plotly_chart(
        plot_pattern_heatmap(result, q, bank["attrs"]),
        width="stretch",
    )

    colL, colR = st.columns(2)
    with colL:
        section("各属性判准率")
        fig = go.Figure(
            go.Bar(
                x=[bank["attrs"][i] if i < len(bank["attrs"]) else f"A{i+1}"
                   for i in range(len(report.attribute_accuracy))],
                y=report.attribute_accuracy,
                marker=dict(color="#0D9488"),
                text=[f"{v:.3f}" for v in report.attribute_accuracy],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 1.05], title="判准率"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12),
        )
        st.plotly_chart(fig, width="stretch")

    with colR:
        section("终止原因分布")
        dist = result.stop_reason_distribution()
        fig = go.Figure(
            go.Pie(
                labels=list(dist.keys()),
                values=list(dist.values()),
                hole=0.45,
                marker=dict(colors=["#4F46E5", "#0D9488", "#E11D48", "#2563EB"]),
            )
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(size=12))
        st.plotly_chart(fig, width="stretch")

    section("题目曝光分布")
    st.plotly_chart(
        plot_exposure(result.item_usage(), q.item_names),
        width="stretch",
    )
    chi = ev.exposure_chi_square(result)
    st.caption(
        f"曝光均匀性 χ² 检验：χ²={chi['statistic']:.2f}, df={chi['df']}, p={chi['p_value']:.4f} → "
        + ("曝光分布均匀" if chi["p_value"] > 0.05 else "存在题目曝光不均（可能个别题目过载）")
    )

    if use_fixed or True:
        section("判准率随题量收敛曲线")
        if st.button("计算收敛曲线（额外仿真）"):
            sim_cc: CDCATSimulator = pack.get("sim") or CDCATSimulator(
                q, model=pack["model"],
                model_params={"slip": bank["slip"], "guess": bank["guess"]},
                selector=pack["selector"], estimator=pack["estimator"],
            )
            with st.spinner("计算中…"):
                curve = sim_cc.convergence_curve(
                    n_examinees=min(int(n_examinees), 300),
                    max_items=min(30, q.n_items),
                    seed=int(seed),
                )
            st.plotly_chart(plot_convergence(curve), width="stretch")

    with st.expander("导出结果"):
        st.download_button(
            "下载被试明细 CSV",
            result.to_dataframe().to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cdcat_records_{pack['selector']}.csv",
            mime="text/csv",
        )
        st.download_button(
            "下载评估报告 JSON",
            json.dumps(result.to_dict()).encode("utf-8"),
            file_name=f"cdcat_report_{pack['selector']}.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------- #
# 页面 3：准则对比
# ---------------------------------------------------------------------- #
def page_compare(bank: dict, model_name: str) -> None:
    hero("选题准则横向对比", "在同一批被试、同一随机流下对比各选题准则，"
         "结果可直接用于选型决策。",
         kicker="Item Selection Criteria", tags=["PWKL / MPWKL", "GDI / SHE / CDI", "MIV 无参数"])

    q: QMatrix = bank["q"]
    c1, c2, c3, c4 = st.columns(4)
    selected = c1.multiselect("对比准则", CRITERIA, default=["PWKL", "MPWKL", "GDI", "SHE"])
    estimator = c2.selectbox("估计方法", ESTIMATORS, index=0, key="cmp_est")
    n_examinees = c3.number_input("被试数", 50, 3000, 200, 50, key="cmp_n")
    n_rep = c4.number_input("重复次数", 1, 30, 3, 1, key="cmp_rep")

    c5, c6 = st.columns(2)
    threshold = c5.slider("终止置信阈值", 0.5, 0.99, 0.80, 0.01, key="cmp_thr")
    fixed = c6.checkbox("使用固定题量对比（消除终止差异）", value=True)
    fixed_n = c6.number_input("固定题量", 1, q.n_items, min(20, q.n_items), 1, disabled=not fixed)
    seed = st.number_input("随机种子", 0, 99999, 0, 1, key="cmp_seed")

    if st.button("开始对比", type="primary", width="stretch", key="cmp_run"):
        if not selected:
            st.warning("请至少选择一个准则。")
            return
        eff_model = model_name if model_name in ("DINA", "DINO") else "DINA"
        model_params = {"slip": bank["slip"], "guess": bank["guess"]}
        term = CompositeRule(
            [PatternConfidence(threshold=float(threshold))],
            how="any", min_items=3, max_items=min(40, q.n_items),
        )
        sim = CDCATSimulator(
            q, model=eff_model, model_params=model_params,
            selector=selected[0], estimator=estimator, termination=term,
        )
        with st.spinner("对比仿真中…"):
            results = sim.compare_criteria(
                selected,
                n_examinees=int(n_examinees),
                n_replications=int(n_rep),
                seed=int(seed),
                fixed_length=int(fixed_n) if fixed else None,
            )
        st.session_state["compare_results"] = {"results": results, "bank": bank}

    pack = st.session_state.get("compare_results")
    if not pack:
        st.info("点击上方按钮开始对比。")
        return

    results = pack["results"]
    ev = Evaluator()
    df = ev.compare(results)

    st.divider()
    section("对比总表")
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            # 统一小数位：题库利用率原本是 1.0，渲染出来只有一个 "1"，与其他列对不齐
            "PMR": st.column_config.NumberColumn(format="%.4f"),
            "AMR": st.column_config.NumberColumn(format="%.4f"),
            "平均题量": st.column_config.NumberColumn(format="%.1f"),
            "题量标准差": st.column_config.NumberColumn(format="%.1f"),
            "题库利用率": st.column_config.NumberColumn(format="%.3f"),
            "曝光率标准差": st.column_config.NumberColumn(format="%.4f"),
            "最大曝光率": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # ---- 题量波动足够大时才画「效率-精度」散点 ----
    # 固定题量模式下每个准则的平均题量都相同，散点会全部叠在一条竖线上、
    # 标签糊成一团（6 个准则时已经看不清）。这种情况下"效率"维度不存在，
    # 换成带 95% 置信区间的精度排序图，正好回答"差距是不是噪声"。
    n_items_each = [float(res.avg_items()) for res in results.values()]
    degenerate = bool(n_items_each) and (max(n_items_each) - min(n_items_each)) < 1e-9

    # 10 色：准则共 10 个（6 个参数化 + 4 个非参数），少一个就会有两个准则撞色
    palette = ["#4F46E5", "#0D9488", "#E11D48", "#2563EB", "#D97706", "#DB2777",
               "#7C3AED", "#0891B2", "#65A30D", "#B45309"]

    if degenerate:
        section("精度排序（95% 置信区间）")
        st.caption(
            f"固定题量下各方案用题量相同（{n_items_each[0]:.0f} 题），没有效率维度可权衡。"
            "看区间：两个方案的区间重叠，就不能断言谁更强。"
        )
        reps = sorted(
            (ev.evaluate(res, label=name) for name, res in results.items()),
            key=lambda rep: rep.pmr,
        )
        fig = go.Figure()
        for i, rep in enumerate(reps):
            fig.add_trace(
                go.Scatter(
                    x=[rep.pmr], y=[rep.label],
                    mode="markers",
                    error_x=dict(
                        type="data", symmetric=False, thickness=1.6, width=6,
                        array=[max(rep.pmr_ci[1] - rep.pmr, 0.0)],
                        arrayminus=[max(rep.pmr - rep.pmr_ci[0], 0.0)],
                        color=palette[i % len(palette)],
                    ),
                    marker=dict(size=13, color=palette[i % len(palette)],
                                line=dict(width=1, color="#FFFFFF")),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{rep.label}</b><br>PMR={rep.pmr:.4f}<br>"
                        f"95%CI [{rep.pmr_ci[0]:.4f}, {rep.pmr_ci[1]:.4f}]<extra></extra>"
                    ),
                )
            )
        lo = min(rep.pmr_ci[0] for rep in reps)
        hi = max(rep.pmr_ci[1] for rep in reps)
        pad = max((hi - lo) * 0.25, 0.005)
        fig.update_layout(
            height=max(260, 52 * len(reps) + 90),
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="模式判准率 PMR", range=[lo - pad, hi + pad]),
            yaxis=dict(title="", automargin=True),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        section("效率-精度权衡")
        st.caption("理想方案位于左上角：用更少的题获得更高的判准率。")
        fig = go.Figure()
        for i, (name, res) in enumerate(results.items()):
            fig.add_trace(
                go.Scatter(
                    x=[res.avg_items()],
                    y=[res.pmr()],
                    mode="markers+text",
                    name=name,
                    text=[name],
                    # 题量接近时标签会互相压住，上下交错摆放可显著缓解
                    textposition="top center" if i % 2 == 0 else "bottom center",
                    marker=dict(size=16, color=palette[i % len(palette)],
                                line=dict(width=1, color="#FFFFFF")),
                )
            )
        fig.update_layout(
            height=400, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="平均用题量", yaxis_title="模式判准率 PMR",
            yaxis=dict(range=[0, 1.05]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=dict(size=12),
        )
        st.plotly_chart(fig, width="stretch")

    # ---- 分组柱状图 ----
    names = list(results.keys())
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=names, y=[results[n].pmr() for n in names],
                          name="PMR", marker_color="#4F46E5"))
    fig2.add_trace(go.Bar(x=names, y=[results[n].amr() for n in names],
                          name="AMR", marker_color="#0D9488"))
    fig2.update_layout(
        barmode="group", height=360, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(range=[0, 1.05], title="判准率"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(size=12),
    )
    st.plotly_chart(fig2, width="stretch")

    st.download_button(
        "下载对比表 CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cdcat_criteria_comparison.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------- #
# 页面 4：题库诊断
# ---------------------------------------------------------------------- #
def page_qmatrix(bank: dict) -> None:
    hero("题库与 Q 矩阵诊断", "Q 矩阵是认知诊断的基石。设计缺陷会直接压低下限判准率，"
         "务必先诊断再施测。",
         kicker="Q-Matrix Diagnostics", tags=["结构可辨识性", "PMR 上限", "冗余检测"])

    q: QMatrix = bank["q"]
    diag = q.diagnose()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("题目数 J", q.n_items)
    c2.metric("属性数 K", q.n_attrs)
    c3.metric("模式总数 2^K", q.n_patterns)
    c4.metric("关联图连通", "是" if diag.is_connected else "否")

    issues = q.validate()
    if issues:
        for it in issues:
            st.error(it)
    else:
        st.success("基础校验通过：每个属性均被考查，每道题均考查至少一个属性。")

    section("属性考查次数")
    fig = go.Figure(
        go.Bar(
            x=bank["attrs"] if len(bank["attrs"]) == q.n_attrs else [f"A{i+1}" for i in range(q.n_attrs)],
            y=diag.attr_exposure,
            marker=dict(color="#2563EB"),
            text=diag.attr_exposure,
            textposition="outside",
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="被考查次数", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12))
    st.plotly_chart(fig, width="stretch")

    colL, colR = st.columns([1, 1])
    with colL:
        section("属性关联图连通性")
        if diag.is_connected:
            st.success("属性关联图连通：任意两个属性通过共同题目间接相连，可联合诊断。")
        else:
            st.warning(
                f"属性关联图不连通，存在 {len(diag.connected_components)} 个分块："
                f"{[[bank['attrs'][i] if i < len(bank['attrs']) else f'A{i+1}' for i in comp] for comp in diag.connected_components]}"
            )
            st.caption("不同分块的属性之间缺乏共同题目，无法互相约束，建议增加跨属性题目。")

        if diag.duplicate_rows:
            section("重复 Q 向量")
            st.caption(f"发现 {len(diag.duplicate_rows)} 组完全相同的 Q 向量，"
                       "这些题目提供的诊断信息完全等价，会浪费题库信息量。")
            sample = list(diag.duplicate_rows.items())[:5]
            rows = []
            for key, idxs in sample:
                rows.append({"Q 向量": pattern_to_str(np.array(key)),
                             "题目数": len(idxs),
                             "题号": ", ".join(q.item_names[i] for i in idxs[:8])})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with colR:
        section("属性共现相关矩阵")
        fig = go.Figure(
            go.Heatmap(
                z=diag.attr_correlations,
                x=[bank["attrs"][i] for i in range(q.n_attrs)],
                y=[bank["attrs"][i] for i in range(q.n_attrs)],
                colorscale=_CMAP_CORR,
                zmin=-1, zmax=1,
                text=np.round(diag.attr_correlations, 2),
                texttemplate="%{text}",
                textfont=dict(size=10.5, color=_HEAT_TEXT),
                colorbar=dict(thickness=12, outlinewidth=0),
            )
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    section("Q 矩阵明细")
    st.caption("深绿格子表示该题考查该属性。")
    mat = q.matrix
    fig = go.Figure(
        go.Heatmap(
            z=mat,
            x=[bank["attrs"][i] for i in range(q.n_attrs)],
            y=q.item_names,
            colorscale=_CMAP_Q,
            showscale=False,
            zmin=0, zmax=1,
        )
    )
    n_show = min(q.n_items, 60)
    fig.update_layout(
        height=max(300, 14 * n_show), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    st.download_button(
        "下载 Q 矩阵 CSV",
        q.to_dataframe().to_csv().encode("utf-8-sig"),
        file_name="qmatrix.csv",
        mime="text/csv",
    )

    # ================================================================== #
    # 结构可辨识性分析
    # ================================================================== #
    st.divider()
    st.header("结构可辨识性分析")
    st.caption("核心结论：即使算法完美、题量无限，Q 矩阵结构本身也可能让某些掌握模式"
               "产生完全相同的理想作答向量，从而在数学上无法区分。"
               "这里给出该题库的 PMR 理论上限。")

    rep = analyse_identifiability(q, ideal_mode="conjunctive")

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("模式总数 2^K", rep.n_patterns)
    i2.metric("可唯一辨识模式", rep.n_distinct_response_vectors)
    i3.metric("不可辨识模式", rep.n_unidentifiable_patterns)
    i4.metric("PMR 理论上限", f"{rep.expected_pmr_ceiling:.3f}")

    colI, colT = st.columns([1, 1.3])
    with colI:
        st.plotly_chart(plot_identifiability(rep), width="stretch")
    with colT:
        if rep.is_complete:
            st.markdown(
                '<div class="cdc-note cdc-note-good">'
                '<b>Q 矩阵完备</b><br>所有 2^K 个掌握模式都产生唯一的理想作答向量，'
                '理论 PMR 上限为 1.0，诊断能力不受结构限制。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="cdc-note cdc-note-warn">'
                f'<b>存在结构缺陷</b><br>有 {rep.n_unidentifiable_patterns} 个模式'
                f'无法被唯一区分，理论 PMR 上限仅 {rep.expected_pmr_ceiling:.3f}。'
                f'无论用多好的算法，PMR 都无法突破该上限。</div>',
                unsafe_allow_html=True,
            )

        if rep.equivalence_classes:
            st.markdown("**不可区分的模式等价组**")
            rows = []
            for grp in rep.equivalence_classes[:10]:
                rows.append({
                    "等价组": " ≡ ".join(pattern_to_str(q.attribute_patterns()[i]) for i in grp),
                    "模式数": len(grp),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if rep.warnings:
        with st.expander(f"详细诊断建议（{len(rep.warnings)} 条）", expanded=True):
            for w in rep.warnings:
                st.markdown(f"- {w}")
    else:
        st.success("未发现结构问题：所有属性均可独立测量，属性对之间无冗余。")

    # 属性级问题清单
    a1, a2, a3 = st.columns(3)
    with a1:
        if rep.uncovered_attrs:
            st.warning("无任何题目考查的属性："
                       + ", ".join(bank["attrs"][i] if i < len(bank["attrs"]) else f"A{i+1}"
                                   for i in rep.uncovered_attrs))
        else:
            st.success("所有属性均有题目覆盖")
    with a2:
        if rep.singleton_attrs:
            st.info("有独立测量题目的属性："
                    + ", ".join(bank["attrs"][i] if i < len(bank["attrs"]) else f"A{i+1}"
                                for i in rep.singleton_attrs))
        else:
            st.warning("没有任何属性拥有独立测量题目（即不存在只考查该属性的题）")
    with a3:
        if rep.interchangeable_attr_pairs:
            pairs = ", ".join(
                f"{bank['attrs'][a] if a < len(bank['attrs']) else f'A{a+1}'}"
                f"↔{bank['attrs'][b] if b < len(bank['attrs']) else f'A{b+1}'}"
                for a, b in rep.interchangeable_attr_pairs
            )
            st.warning(f"可互换（无法区分）的属性对：{pairs}")
        else:
            st.success("不存在无法区分的属性对")

    # ================================================================== #
    # 题目信息量分析
    # ================================================================== #
    st.divider()
    st.header("题目信息量与冗余分析")
    st.caption("从响应模型出发，评估每道题对诊断的贡献：区分度、信息量，"
               "并检出 Q 向量完全相同的冗余题目。")

    j1, j2 = st.columns([1, 3])
    n_sim = j1.number_input("蒙特卡洛样本量", 500, 20000, 4000, 500, key="ia_nsim")
    run_ia = j2.button("开始题目分析", type="primary", key="ia_run")

    if run_ia or st.session_state.get("item_analysis") is not None:
        if run_ia:
            with st.spinner("正在评估题目区分度与信息量…"):
                model_obj = create_model(
                    "DINA", q,
                    slip=np.full(q.n_items, bank["slip"]),
                    guess=np.full(q.n_items, bank["guess"]),
                )
                ia = analyse_items(
                    model_obj, n_simulate=int(n_sim), seed=2024
                )
                st.session_state["item_analysis"] = ia

        ia = st.session_state.get("item_analysis")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("平均区分度", f"{ia.discrimination.mean():.3f}")
        k2.metric("最强题区分度", f"{ia.discrimination.max():.3f}")
        k3.metric("最弱题区分度", f"{ia.discrimination.min():.3f}")
        k4.metric("结构冗余题组", len(ia.structural_duplicates))

        colA, colB = st.columns(2)
        with colA:
            section("题目区分度排序")
            st.caption("区分度 D 越高，说明该题越能拉开掌握与未掌握考生的作答差异。")
            st.plotly_chart(
                plot_item_discrimination(ia, q), width="stretch"
            )
        with colB:
            section("题目信息量")
            st.caption("信息量综合反映题目对降低属性后验不确定性的贡献。")
            st.plotly_chart(
                plot_item_information(ia, q), width="stretch"
            )

        if ia.weakest_items:
            st.markdown("**区分度最低的题目（建议优先审查或替换）**")
            wrows = []
            for i in ia.weakest_items:
                wrows.append({
                    "题号": q.item_names[i],
                    "Q 向量": "+".join(q.attr_names[k] for k in q.required_attrs(i)) or "—",
                    "区分度": f"{ia.discrimination[i]:.3f}",
                    "信息量": f"{ia.information[i]:.3f}",
                })
            st.dataframe(pd.DataFrame(wrows), width="stretch", hide_index=True)

        if ia.structural_duplicates:
            st.markdown("**结构冗余题组（Q 向量完全相同）**")
            st.caption("这些题目考查完全相同的属性组合，在结构上提供等价的诊断信息。"
                       "保留其中 1–2 道即可，多余题目只增加施测成本。")
            drows = []
            for grp in ia.structural_duplicates:
                drows.append({
                    "Q 向量": "+".join(q.attr_names[k] for k in q.required_attrs(grp[0])) or "—",
                    "题目数": len(grp),
                    "题号": ", ".join(q.item_names[i] for i in grp[:10]),
                })
            st.dataframe(pd.DataFrame(drows), width="stretch", hide_index=True)
        else:
            st.success("未发现 Q 向量完全相同的冗余题目，题库结构良好。")

        ia_rows = [
            {
                "题号": q.item_names[i],
                "Q 向量": "+".join(q.attr_names[k] for k in q.required_attrs(i)) or "—",
                "区分度 D": round(float(ia.discrimination[i]), 4),
                "信息量 I": round(float(ia.information[i]), 4),
                "区分度排名": int(ia.discrimination_rank[i]),
                "信息量排名": int(ia.information_rank[i]),
            }
            for i in range(q.n_items)
        ]
        st.download_button(
            "下载题目分析 CSV",
            pd.DataFrame(ia_rows).to_csv(index=False).encode("utf-8-sig"),
            file_name="cdcat_item_analysis.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------- #
# 页面 5：参数标定
# ---------------------------------------------------------------------- #
def page_calibration(bank: dict) -> None:
    hero("CDM 参数标定与诊断分析", "输入作答矩阵，用 MMLE/EM 估计题目参数（DINA / DINO / G-DINA / RRUM / ACDM / LLM），"
         "输出被试知识状态、项目参数与诊断分类结果，并给出模型拟合指标与可视化。",
         kicker="Model Calibration", tags=["MMLE / EM", "经验贝叶斯 π(α)", "χ²/G² 绝对拟合", "Yen Q3"])

    q: QMatrix = bank["q"]
    attrs = bank["attrs"]

    # ---------------- 数据来源 ---------------- #
    section("1. 数据来源")
    src = st.radio("作答数据", ["合成数据演示", "上传 CSV"], horizontal=True, key="cal_src")

    responses = None
    truth_pack = None
    if src == "合成数据演示":
        c1, c2, c3, c4 = st.columns(4)
        n_ex = c1.number_input("考生数", 100, 20000, 2000, 100, key="cal_n")
        gen_model = c2.selectbox("真实生成模型", MODELS, index=0, key="cal_gen")
        seed = c3.number_input("随机种子", 0, 99999, 0, 1, key="cal_seed")
        c4.caption("合成数据已知真值，可评估参数恢复精度")

        if st.button("生成合成数据", type="primary", width="stretch", key="cal_gen_btn"):
            rng = np.random.default_rng(int(seed))
            gen = _make_truth_model(gen_model, q, bank)
            alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, int(n_ex))]
            responses = np.array([gen.simulate(a, rng=rng) for a in alphas], dtype=float)
            st.session_state["cal_responses"] = responses
            st.session_state["cal_truth"] = {
                "model": gen_model, "alphas": alphas,
                "slip": bank["slip"], "guess": bank["guess"],
            }
    else:
        up = st.file_uploader("作答矩阵 CSV（行=考生，列=题目，值为 0/1）", type=["csv"], key="cal_up")
        if up is not None:
            df = pd.read_csv(up)
            responses = df.to_numpy(dtype=float)
            if responses.shape[1] != q.n_items:
                st.error(f"作答矩阵列数 {responses.shape[1]} 与题库题目数 {q.n_items} 不一致，"
                         "请先在左侧切换到对应题库。")
                return
            st.success(f"已读取作答矩阵：{responses.shape[0]} 名考生 x {responses.shape[1]} 道题")
            st.session_state["cal_responses"] = responses
            st.session_state["cal_truth"] = None

    responses = st.session_state.get("cal_responses")
    truth_pack = st.session_state.get("cal_truth")
    if responses is None:
        st.info("请先生成合成数据或上传作答矩阵。")
        return

    st.caption(f"当前作答矩阵：{responses.shape[0]} 名考生 × {responses.shape[1]} 道题")

    # ---------------- 标定配置 ---------------- #
    section("2. 标定配置")
    c1, c2 = st.columns([3, 1])
    with c1:
        to_fit = st.multiselect(
            "要标定的模型（可多选，将自动做拟合比较）",
            MODELS, default=["DINA", "G-DINA"], key="cal_models",
        )
    with c2:
        max_iter = st.number_input("最大迭代", 50, 2000, 300, 50, key="cal_maxit")

    if not to_fit:
        st.warning("请至少选择一个模型。")
        return

    if st.button("开始标定与分析", type="primary", width="stretch", key="cal_run"):
        with st.spinner("EM 迭代与拟合指标计算中…"):
            analysis = run_cdm_analysis(
                q, responses,
                models=to_fit,
                max_iter=int(max_iter),
                tol=1e-7,
                attr_names=attrs,
                true_slip=truth_pack["slip"] if truth_pack else None,
                true_guess=truth_pack["guess"] if truth_pack else None,
                true_patterns=truth_pack["alphas"] if truth_pack else None,
                nonparametric=("gdd", "hamming"),
            )
        st.session_state["cdm_analysis"] = analysis
        # 兼容旧逻辑：把最优模型参数写入 session，供侧边栏「由 EM 标定」使用
        best = analysis.models[analysis.best_model]
        if best.slip is not None:
            st.session_state["calibrated_params"] = {
                "slip": best.slip, "guess": best.guess
            }

    analysis = st.session_state.get("cdm_analysis")
    if analysis is None:
        st.info("点击上方按钮开始标定。")
        return

    # ---------------- 3. 模型拟合指标 ---------------- #
    st.divider()
    section("3. 模型拟合指标与择优")
    st.caption("AIC / BIC / AICc 越小越好。BIC 对参数个数惩罚更重，通常作为首选。")

    best = analysis.best_model
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最优模型 (BIC)", best)
    m2.metric("最优模型 BIC", f"{analysis.fits[best].bic:,.1f}")
    m3.metric("自由参数数", analysis.fits[best].n_params)
    m4.metric("边际对数似然", f"{analysis.fits[best].log_likelihood:,.1f}")

    st.dataframe(
        analysis.fit_table().round(4),
        width="stretch", hide_index=True,
    )

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**EM 收敛曲线**")
        st.plotly_chart(fig_ll_convergence(analysis.models), width="stretch")
    with colR:
        st.markdown("**拟合指标对比**")
        st.plotly_chart(fig_fit_comparison(analysis.fits), width="stretch")

    if analysis.lrt is not None:
        lrt = analysis.lrt
        msg = (f"似然比检验（最简模型 vs 最优模型）："
               f"LR = {lrt['lr_statistic']:.2f}，df = {lrt['df']}，"
               f"p = {lrt['p_value']:.4g}")
        if lrt["significant"]:
            st.info(msg + " —— 模型间差异显著，选择更复杂的模型有统计学依据。")
        else:
            st.info(msg + " —— 模型间无显著差异，建议选择更简洁的模型（奥卡姆剃刀）。")

    # ---------------- 3.5 非参数基线 ---------------- #
    if analysis.nonparametric:
        st.markdown("**非参数基线（不估计题目参数）**")
        st.caption(
            "距离判别只用 Q 矩阵决定的理想反应模式，0 个自由参数、无需标定。"
            "它的价值是当对照组：若参数模型的 PMR 追不平它，说明标定没有带来"
            "实际收益，问题多半在 Q 矩阵或模型设定上；反过来，参数模型的领先幅度"
            "就是标定的真实增益。"
        )
        st.dataframe(
            analysis.nonparametric_table(),
            width="stretch", hide_index=True,
            column_config={
                # 保持 4 位小数：round() 后 0.97 会丢掉尾零，统计表看着不齐
                "PMR": st.column_config.NumberColumn(format="%.4f"),
                "AMR": st.column_config.NumberColumn(format="%.4f"),
                "平均判别把握": st.column_config.NumberColumn(format="%.4f"),
                "PMR 差（非参数−参数）": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        if getattr(analysis.fits[best], "pmr", None) is None:
            st.caption(
                "当前没有真实属性模式（上传数据），只能看到判别把握；"
                "用「合成数据演示」可同时得到 PMR / AMR。"
            )

    # ---------------- 4. 项目参数 ---------------- #
    st.divider()
    section("4. 项目参数与区分度")

    pt = analysis.param_table()
    if not pt.empty:
        st.dataframe(
            pt[pt["模型"] == best].reset_index(drop=True),
            width="stretch", hide_index=True, height=300,
        )

    best_res = analysis.models[best]
    disc = 1.0 - best_res.slip - best_res.guess
    st.plotly_chart(
        fig_cdm_discrimination(disc, q.item_names, top_n=30),
        width="stretch",
    )
    low_disc = np.where(disc < 0.3)[0]
    if len(low_disc):
        st.warning(
            f"发现 {len(low_disc)} 道区分度低于 0.3 的题目，建议修订或淘汰："
            + ", ".join(q.item_names[i] for i in low_disc[:12])
        )
    else:
        st.success("所有题目区分度均达到 0.3 以上。")

    # G-DINA / RRUM / ACDM / LLM 的完整参数展示
    with st.expander("查看完整模型参数（G-DINA / RRUM / ACDM / LLM 的专属参数）"):
        sel = st.selectbox("模型", list(analysis.models), key="cal_param_sel")
        res_sel = analysis.models[sel]
        if res_sel.beta is not None:
            st.markdown("**G-DINA 的 λ 参数（beta = baseline，gamma = 各交互项）**")
            rows = []
            for i in range(q.n_items):
                rows.append({
                    "题号": q.item_names[i],
                    "考查属性": "+".join(attrs[k] for k in q.required_attrs(i)) or "—",
                    "beta": round(float(res_sel.beta[i]), 4),
                    "gamma": ", ".join(f"{v:.3f}" for v in res_sel.gamma[i]) or "—",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        elif res_sel.pi_star is not None:
            st.markdown(r"**RRUM 的 $\pi^*$ 与 $r^*$ 参数**")
            rows = []
            for i in range(q.n_items):
                rows.append({
                    "题号": q.item_names[i],
                    "考查属性": "+".join(attrs[k] for k in q.required_attrs(i)) or "—",
                    "π*": round(float(res_sel.pi_star[i]), 4),
                    "r*": ", ".join(f"{v:.3f}" for v in res_sel.r_star[i]),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        elif res_sel.delta0 is not None:
            st.markdown("**ACDM 的 δ 参数（δ0 = 基线概率，δk = 各属性掌握后的加性增量）**")
            rows = []
            for i in range(q.n_items):
                rows.append({
                    "题号": q.item_names[i],
                    "考查属性": "+".join(attrs[k] for k in q.required_attrs(i)) or "—",
                    "δ0": round(float(res_sel.delta0[i]), 4),
                    "δk": ", ".join(f"{v:.3f}" for v in res_sel.delta[i]) or "—",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        elif res_sel.llm_beta0 is not None:
            st.markdown("**LLM 的 β 参数（β0 = 截距，βk = 各属性在 logit 尺度上的增量）**")
            rows = []
            for i in range(q.n_items):
                rows.append({
                    "题号": q.item_names[i],
                    "考查属性": "+".join(attrs[k] for k in q.required_attrs(i)) or "—",
                    "β0": round(float(res_sel.llm_beta0[i]), 4),
                    "βk": ", ".join(f"{v:.3f}" for v in res_sel.llm_beta[i]) or "—",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("该模型的参数已在上方「项目参数与区分度」表中展示（slip / guess）。")

    # ---------------- 5. 参数恢复评价 ---------------- #
    if analysis.recovery:
        st.divider()
        section("5. 参数恢复精度")
        st.caption("合成数据已知真值，可据此评估标定算法的参数恢复能力。")

        rrows = []
        for k, v in analysis.recovery.items():
            rrows.append({
                "参数": k, "相关系数": round(v["correlation"], 4),
                "MAE": round(v["mae"], 4), "RMSE": round(v["rmse"], 4),
                "偏差": round(v["bias"], 4),
            })
        st.dataframe(pd.DataFrame(rrows), width="stretch", hide_index=True)

        ts, tg = truth_pack["slip"], truth_pack["guess"]
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"**{best} · slip 恢复**")
            st.plotly_chart(
                fig_param_recovery(best_res.slip, ts, "slip"),
                width="stretch",
            )
        with rc2:
            st.markdown(f"**{best} · guess 恢复**")
            st.plotly_chart(
                fig_param_recovery(best_res.guess, tg, "guess"),
                width="stretch",
            )

    # ---------------- 6. 被试知识状态诊断 ---------------- #
    st.divider()
    section("6. 被试知识状态诊断结果")

    diag = analysis.diagnosis[best]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("诊断被试数", diag.n_examinees)
    d2.metric("平均分类信度", f"{diag.mean_confidence():.4f}")
    d3.metric("出现过的知识状态数", len(diag.pattern_distribution()))
    if truth_pack is not None:
        f = analysis.fits[best]
        d4.metric("模式判准率 PMR", f"{f.pmr:.4f}" if f.pmr is not None else "—")
    else:
        d4.metric("属性数", diag.n_attrs)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**各属性掌握比例**")
        st.plotly_chart(
            fig_mastery_proportion(diag.mastery_proportion(), attrs),
            width="stretch",
        )
    with dc2:
        st.markdown("**属性级分类信度**")
        st.plotly_chart(
            fig_attr_reliability(diag.attr_reliability, attrs),
            width="stretch",
        )

    st.markdown("**知识状态频次分布**")
    st.plotly_chart(fig_pattern_distribution(diag), width="stretch")

    with st.expander("查看被试诊断明细（前 200 行）"):
        st.dataframe(
            diag.to_dataframe().head(200),
            width="stretch", hide_index=True,
        )

    low_rel = diag.low_reliability_attrs(threshold=0.6)
    n_low = sum(len(x) for x in low_rel)
    if n_low:
        st.warning(
            f"共有 {n_low} 个「被试×属性」组合的分类信度低于 0.6，"
            "建议对这些考生加测或人工复核。"
        )

    # ---------------- 7. 绝对拟合与局部依赖 ---------------- #
    if analysis.absolute_fit is not None:
        af = analysis.absolute_fit
        st.divider()
        section("7. 绝对拟合与局部依赖诊断")
        st.caption(
            "AIC/BIC 只能回答「哪个模型更好」；这一节回答「这个模型本身够不够好」。"
            "p 值会被大样本放大，请优先看「最大残差」这一效应量。"
        )

        m1, m2, m3, m4 = st.columns(4)
        bad = int((af.item_fit.p_value < 0.05).sum())
        m1.metric("不合格题目数", f"{bad} / {af.item_fit.chi_square.size}")
        m2.metric("最大残差（均值）", f"{af.item_fit.max_abs_residual.mean():.4f}")
        m3.metric("Q3 超标题目对", f"{af.local_dependence.n_flagged}")
        m4.metric("平均 Q3", f"{af.local_dependence.mean_q3:.4f}")

        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.plotly_chart(fig_item_fit(af.item_fit), width="stretch")
        with c2:
            st.plotly_chart(fig_q3_heatmap(af.local_dependence),
                            width="stretch")

        with st.expander("查看题目级拟合明细与 Q3 题目对", expanded=bad > 0):
            st.markdown("**题目级绝对拟合（按 p 值升序）**")
            st.dataframe(
                af.item_fit.to_dataframe().sort_values("p值").reset_index(drop=True),
                width="stretch", hide_index=True, height=280,
            )
            flagged = af.local_dependence.flagged_pairs(top_n=30)
            if not flagged.empty:
                st.markdown(
                    f"**Q3 超过 {af.local_dependence.threshold} 的题目对"
                    "（可能漏标了共同属性）**"
                )
                st.dataframe(flagged, width="stretch",
                             hide_index=True, height=240)
            else:
                st.success("未检出超过阈值的局部依赖题目对，Q 矩阵结构合理。")

        prior_best = analysis.models[best].attr_prior
        if prior_best is not None:
            with st.expander("查看属性模式先验 π(α)（经验贝叶斯估计）"):
                st.plotly_chart(fig_attr_prior(analysis.models[best]),
                                width="stretch")
                st.caption(
                    "虚线为均匀先验 1/2^K。柱子高低即各知识状态在总体中的"
                    "估计占比；在 M 步与题目参数一起被估出来。"
                )

    # ---------------- 8. 导出 ---------------- #
    st.divider()
    section("8. 导出结果")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "下载拟合指标 CSV",
            analysis.fit_table().to_csv(index=False).encode("utf-8-sig"),
            file_name="cdm_fit_indices.csv", mime="text/csv",
            width="stretch",
        )
    with e2:
        st.download_button(
            "下载被试诊断 CSV",
            diag.to_dataframe().to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cdm_diagnosis_{best}.csv", mime="text/csv",
            width="stretch",
        )
    with e3:
        st.download_button(
            "下载题目参数 CSV",
            analysis.param_table().to_csv(index=False).encode("utf-8-sig"),
            file_name="cdm_item_params.csv", mime="text/csv",
            width="stretch",
        )

    e4, e5 = st.columns(2)
    if analysis.absolute_fit is not None:
        with e4:
            st.download_button(
                "下载题目绝对拟合 CSV",
                analysis.absolute_fit.item_fit.to_dataframe()
                .to_csv(index=False).encode("utf-8-sig"),
                file_name="cdm_item_fit.csv", mime="text/csv",
                width="stretch",
            )
        with e5:
            st.download_button(
                "下载局部依赖 Q3 超标对 CSV",
                analysis.absolute_fit.local_dependence.flagged_pairs()
                .to_csv(index=False).encode("utf-8-sig"),
                file_name="cdm_q3_pairs.csv", mime="text/csv",
                width="stretch",
            )

    st.download_button(
        "下载完整分析报告 (TXT)",
        analysis.summary().encode("utf-8-sig"),
        file_name="cdm_analysis_report.txt", mime="text/plain",
        width="stretch",
    )

    if st.button("将最优模型参数应用到题库", width="stretch"):
        if best_res.slip is not None:
            st.session_state["bank"]["slip"] = best_res.slip
            st.session_state["bank"]["guess"] = best_res.guess
            st.success(
                f"已应用 {best} 的等价 DINA 参数。"
                "可在侧边栏「参数来源」中选择「由 EM 标定」确认使用。"
            )
        else:
            st.warning("该模型未产出等价的 slip / guess 参数。")


def _make_truth_model(model_name: str, q: QMatrix, bank: dict):
    """按名称构造"真实"生成模型，用于合成作答数据。"""
    from cdcat import ACDM as _ACDM, DINA as _DINA, DINO as _DINO, LLM as _LLM

    slip, guess = bank["slip"], bank["guess"]
    if model_name == "DINO":
        return _DINO(q, slip=slip, guess=guess)
    if model_name == "G-DINA":
        from cdcat import GDINA as _GDINA
        return _GDINA.from_dina(q, slip, guess)
    if model_name == "RRUM":
        from cdcat import RRUM as _RRUM
        r_star = [np.full(max(int(q.matrix[i].sum()), 1), 0.75) for i in range(q.n_items)]
        return _RRUM(q, pi_star=1.0 - slip, r_star=r_star)
    if model_name == "ACDM":
        return _ACDM.from_dina(q, slip, guess)
    if model_name == "LLM":
        return _LLM.from_dina(q, slip, guess)
    return _DINA(q, slip=slip, guess=guess)



# ---------------------------------------------------------------------- #
# 页面：群体报告 & Q 矩阵修正
# ---------------------------------------------------------------------- #
def _load_uploaded_responses(up, n_items: int):
    """把上传的作答 CSV 落到临时文件后交给 ``cdcat.io.read_responses`` 解析。

    复用库内同一套嗅探逻辑（分隔符 / 表头 / 缺失标记），
    避免界面里再写一遍解析代码导致两边行为漂移。
    """
    import tempfile

    from cdcat.io import read_responses

    raw = up.getvalue()
    with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as fh:
        fh.write(raw)
        tmp = fh.name
    try:
        mat, ids = read_responses(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if mat.shape[1] != n_items:
        raise ValueError(
            f"作答矩阵有 {mat.shape[1]} 列，与当前题库的 {n_items} 题不一致"
        )
    return mat, ids


def page_group(bank: dict, model_name: str) -> None:
    hero(
        "群体 / 班级诊断报告",
        "把个人诊断汇总到班级或群体层面：谁掌握了什么、哪个知识点最薄弱、"
        "哪些知识状态最需要继续加题。",
        kicker="Group Diagnostic Report",
        tags=["掌握率排行", "薄弱知识点", "确定性分档", "分组对比", "Q 矩阵修正"],
    )

    q: QMatrix = bank["q"]
    model = _make_truth_model(model_name, q, bank)

    # ---------------- 数据来源 ----------------
    section("1. 数据来源", "合成数据可立刻看到完整流程；上传 CSV 用于真实作答")
    src = st.radio("作答数据", ["合成数据演示", "上传作答矩阵 CSV"],
                   horizontal=True, key="grp_src")

    responses = None
    ids: list[str] | None = None
    truth = None

    if src == "合成数据演示":
        c1, c2, c3 = st.columns(3)
        n_ex = c1.number_input("被试数", min_value=30, max_value=5000,
                               value=300, step=30, key="grp_n")
        miss = c3.slider("人为缺失比例", 0.0, 0.30, 0.0, 0.01, key="grp_miss",
                         help="模拟漏答：缺失值按 NaN 处理，不插补")
        c2.caption("真实知识状态从均匀分布抽样；人数越多，掌握率估计越稳。")
        if st.button("生成合成数据", key="grp_gen", type="primary"):
            rng = np.random.default_rng(2024)
            pats = q.attribute_patterns()
            idx = rng.integers(0, q.n_patterns, size=int(n_ex))
            truth = pats[idx]
            resp = np.array([model.simulate(a, rng=rng) for a in truth], dtype=float)
            if miss > 0:
                resp = np.where(rng.random(resp.shape) < miss, np.nan, resp)
            st.session_state["grp_resp"] = resp
            st.session_state["grp_ids"] = [f"S{i + 1:04d}" for i in range(int(n_ex))]
            st.session_state["grp_truth"] = truth
            st.session_state.pop("grp_report", None)
    else:
        up = st.file_uploader("作答矩阵 CSV（行=被试，列=题目，缺失值留空）",
                              type=["csv", "tsv", "txt"], key="grp_file")
        if up is not None:
            try:
                mat, eids = _load_uploaded_responses(up, q.n_items)
                st.session_state["grp_resp"] = mat
                st.session_state["grp_ids"] = eids or [f"E{i + 1}" for i in range(mat.shape[0])]
                st.session_state["grp_truth"] = None
                st.session_state.pop("grp_report", None)
                st.success(f"已读取 {mat.shape[0]} 名被试 × {mat.shape[1]} 题；"
                           f"缺失率 {np.isnan(mat).mean():.2%}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"读取失败：{exc}")

    responses = st.session_state.get("grp_resp")
    ids = st.session_state.get("grp_ids")
    truth = st.session_state.get("grp_truth")

    if responses is None:
        st.info("请先生成合成数据，或上传一份作答矩阵 CSV。")
        return

    # ---------------- 选项 ----------------
    section("2. 分析选项", "属性层级会收紧估计空间；分组表用于班级横向对比")
    o1, o2, o3, o4 = st.columns([1, 1, 1, 2])
    method = o1.selectbox("估计方法", ["MAP", "EAP"], key="grp_method")
    missing = o2.selectbox(
        "缺失作答", ["skip", "wrong", "ignore"], key="grp_missing",
        help="skip=跳过该题 / wrong=按答错 / ignore=有缺失就报错",
    )
    thr = o3.slider("掌握判定阈值", 0.5, 0.95, 0.85, 0.05, key="grp_thr")

    with o4:
        hierarchy = _hierarchy_selector(q, key="grp_hier")
        if hierarchy is not None and not hierarchy.is_empty:
            st.caption(f"合法模式 {hierarchy.n_legal_patterns} / {q.n_patterns}")

    gcol1, gcol2 = st.columns([1, 2])
    with gcol1:
        gup = st.file_uploader("分组表 CSV（可选，两列：被试ID,组名）",
                               type=["csv", "tsv", "txt"], key="grp_gfile")
    with gcol2:
        st.caption("按 ID 匹配（与行序无关）；也可只给一列组名，此时按作答矩阵行序对应。")
        if gup is not None:
            st.caption(f"已选择：{gup.name}")

    # 注意：这里用 session_state 承载结果，而不是靠 st.button 的返回值。
    # 否则用户一旦点了页面上其它按钮（如 Q 矩阵校验），按钮返回 False，
    # 整份报告就会从页面上消失。
    if st.button("生成群体报告", key="grp_run", type="primary"):
        try:
            group_labels = _parse_group_file(gup, ids)
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner("正在逐人估计知识状态 ……"):
            try:
                report = build_group_report(
                    None,
                    attr_names=q.attr_names,
                    q_matrix=q,
                    item_names=q.item_names,
                    model=model,
                    responses=responses,
                    examinee_ids=ids,
                    group_labels=group_labels,
                    method=method,
                    label="群体诊断",
                    missing=missing,
                    include_truth=truth is not None,
                    hierarchy=hierarchy,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"分析失败：{exc}")
                return
        st.session_state["grp_report"] = report
        st.session_state["grp_labels"] = group_labels
        st.session_state["grp_truth"] = truth

    report = st.session_state.get("grp_report")
    if report is None:
        st.info("点击上方「生成群体报告」开始分析。")
        return
    group_labels = st.session_state.get("grp_labels")
    truth = st.session_state.get("grp_truth")

    # ---------------- 总览 ----------------
    section("3. 总体概览")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("有效被试数", f"{report.n_examinees}")
    m2.metric("人均掌握属性数", f"{report.mean_mastered:.2f}",
              delta=f"/ {q.n_attrs}", delta_color="off")
    m3.metric("完全掌握人数", f"{report.n_full_mastery}",
              delta=f"{report.full_mastery_rate:.1%}", delta_color="off")
    m4.metric("零掌握人数", f"{report.n_zero_mastery}")

    weak = report.weakest_attrs(3)
    st.warning(
        "**最需要补强的知识点**：" + "、".join(
            f"{q.attr_names[i]}（掌握率 {report.attr_mastery_rate[i]:.1%}）" for i in weak
        )
    )

    section("4. 属性掌握率", "最薄弱的排在最上方；虚线为目标线")
    st.plotly_chart(fig_attr_mastery_bar(report, threshold=float(thr)),
                    width="stretch")

    section("5. 诊断确定性", "琥珀段越长，说明该属性越需要继续加题才能判清楚")
    st.plotly_chart(fig_mastery_certainty(report), width="stretch")
    _sure = np.asarray(report.probabilities)
    _amb = float(((_sure >= 0.2) & (_sure <= 0.8)).mean())
    _c1, _c2 = st.columns([1, 2])
    _c1.metric("待定判定占比", f"{_amb:.1%}")
    _c2.caption("该指标汇总所有「被试 × 属性」单元中掌握概率落在 0.2~0.8 之间的比例。"
                "偏高说明当前题库对部分属性区分力不足，或有效作答量偏少。")

    col_l, col_r = st.columns(2)
    with col_l:
        section("6. 典型知识状态", "出现频次最高的属性模式")
        st.plotly_chart(fig_group_patterns(report, top_n=10), width="stretch")
    with col_r:
        section("7. 属性掌握雷达", "总体 vs 各组")
        st.plotly_chart(fig_group_radar(report), width="stretch")

    df = report.attr_dataframe()
    with st.expander("各属性明细表", expanded=False):
        st.dataframe(df, width="stretch")
    with st.expander("典型属性模式频次", expanded=False):
        st.dataframe(report.pattern_dataframe(12), width="stretch")

    # ---------------- 分组 ----------------
    if group_labels is not None:
        section("8. 分组对比", f"共 {len(set(group_labels))} 组")
        st.plotly_chart(fig_group_compare(report), width="stretch")
        gdf = report.group_dataframe()
        if not gdf.empty:
            st.dataframe(gdf, width="stretch")

    # ---------------- 与真值对照 ----------------
    if truth is not None:
        section("9. 与真实知识状态对照", "仅合成数据可用")
        mm = np.asarray(report.mastery_matrix)
        amr = float((mm == np.asarray(truth)).mean())
        pmr = float((np.asarray(report.pattern_idx) ==
                     (np.asarray(truth) * (2 ** np.arange(q.n_attrs - 1, -1, -1))).sum(axis=1)
                     ).mean())
        c1, c2 = st.columns(2)
        c1.metric("属性判准率 AMR", f"{amr:.4f}")
        c2.metric("模式判准率 PMR", f"{pmr:.4f}")

    # ---------------- 个体 vs 群体 ----------------
    section("10. 个体 vs 群体常模", "查看某位考生在群体中的位置（百分位越高越强）")
    pick = st.selectbox("选择被试", list(range(len(ids))),
                        format_func=lambda i: f"{ids[i]}", key="grp_pick")
    if st.button("查看该生对比", key="grp_cmp"):
        info = report.compare_examinee(int(pick))
        st.json(info)

    # ---------------- 导出 ----------------
    section("11. 导出", "报告文本 / JSON / 属性表")
    e1, e2, e3 = st.columns(3)
    e1.download_button("报告 (TXT)", report.to_text().encode("utf-8"),
                       file_name="group_report.txt", key="grp_dl_txt")
    e2.download_button("结果 (JSON)", json.dumps(report.to_dict(), ensure_ascii=False,
                                                indent=2).encode("utf-8"),
                       file_name="group_report.json", key="grp_dl_json")
    e3.download_button("属性表 (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="attribute_mastery.csv", key="grp_dl_csv")

    # ---------------- Q 矩阵修正 ----------------
    st.divider()
    _q_fix_block(q, responses, model, ids)


def _parse_group_file(up, ids) -> list | None:
    """解析上传的分组表，返回与 ``ids`` 等长的组名列表。

    两种格式：

    - 两列 ``被试ID,组名`` —— 按 ID 匹配（与行序无关）
    - 单列 ``组名`` —— 顺序须与作答矩阵一致

    任一被试在分组表中缺失时抛 ``ValueError``（不静默填默认值）。
    """
    if up is None:
        return None
    import csv as _csv
    import io as _io

    raw = up.getvalue().decode("utf-8-sig")
    rows = [r for r in _csv.reader(_io.StringIO(raw)) if any(str(c).strip() for c in r)]
    if not rows:
        raise ValueError("分组表为空")

    header_like = (
        len(rows[0]) >= 2
        and str(rows[0])[0].strip().lower() in
        ("sid", "id", "examinee", "examinee_id", "被试", "学号", "编号")
    )
    if header_like:
        rows = rows[1:]
    if not rows:
        raise ValueError("分组表只有表头，没有数据行")

    if len(rows[0]) >= 2:
        mapping = {str(r[0]).strip(): str(r[1]).strip() for r in rows}
        lack = [i for i in ids if str(i) not in mapping]
        if lack:
            raise ValueError(
                f"分组表缺少 {len(lack)} 名被试的组别（例如 {lack[:3]}）"
            )
        return [mapping[str(i)] for i in ids]

    labels = [str(r[0]).strip() for r in rows]
    if len(labels) != len(ids):
        raise ValueError(
            f"分组表 {len(labels)} 行，作答矩阵 {len(ids)} 行，无法一一对应"
        )
    return labels


def _hierarchy_selector(q: QMatrix, key: str):
    """属性层级选择器（返回 AttributeHierarchy 或 None）。"""
    from cdcat import AttributeHierarchy

    k = q.n_attrs
    label = st.selectbox(
        "属性层级", ["无（属性相互独立）", "线型（逐级先备）",
                     "汇聚型（末端汇聚）", "发散型（多点发散）"],
        key=f"{key}_kind",
        help="承认属性间的先备关系可把合法模式从 2^K 压缩到少数几个，从而抬高判准率上限。",
    )
    try:
        if label.startswith("线型"):
            return AttributeHierarchy.linear(k, q.attr_names)
        if label.startswith("汇聚"):
            return AttributeHierarchy.convergent(k, q.attr_names)
        if label.startswith("发散"):
            return AttributeHierarchy.divergent(k, q.attr_names)
    except Exception:  # noqa: BLE001 - 层级不适用（如 K 太小）时退回无层级
        st.caption("该题库属性数不适用此层级，已退回无层级。")
    return None


def _q_fix_block(q: QMatrix, responses: np.ndarray, model, ids) -> None:
    """Q 矩阵自动修正（Qval / PVAF）区块。"""
    section("Q 矩阵自动修正（Qval / PVAF）",
            "用作答数据反向检验专家的属性标注，给出加/删属性建议")
    st.caption(
        "原理：对每题穷举属性子集，取最能解释该题得分的那个。"
        "PVAF 达标的候选中优先选**属性最少**者（奥卡姆）。"
        "PVAF 偏低说明该题的属性标注几乎解释不了作答差异，最该人工复核。"
    )
    c1, c2 = st.columns([1, 2])
    pvaf_thr = c1.slider("PVAF 阈值", 0.60, 0.99, 0.95, 0.01, key="qf_thr")
    c2.caption("阈值越高越严格：0.95 通常只报出问题明显的题目；"
               "调低到 0.8 可以查看更多潜在改进项。")

    if st.button("运行 Q 矩阵校验", key="qf_run"):
        with st.spinner("正在逐题扫描候选属性子集 ……"):
            try:
                rep = validate_q_matrix(q, responses, model, pvaf_threshold=float(pvaf_thr))
            except Exception as exc:  # noqa: BLE001
                st.error(f"校验失败：{exc}")
                return
        st.session_state["qf_rep"] = rep

    rep = st.session_state.get("qf_rep")
    if rep is None:
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("疑似需修正题目", f"{rep.n_suspect} / {rep.n_items}")
    m2.metric("建议加属性", f"{rep.n_additions}")
    m3.metric("建议删属性", f"{rep.n_removals}")
    m4.metric("全库平均 PVAF", f"{rep.mean_pvaf:.4f}")

    if rep.n_suspect == 0:
        st.success("未发现需要修正的 Q 行 —— 现有标注与作答数据一致。")
    else:
        st.dataframe(rep.to_dataframe().head(20), width="stretch")
        with st.expander("逐条建议", expanded=False):
            for s in rep.suggestions[:30]:
                st.markdown(
                    f"- **{s.item_name}**：`{s.current_attrs_str()}` → "
                    f"`{s.suggested_attrs_str()}` ｜ PVAF "
                    f"{s.pvaf_current:.3f} → {s.pvaf_suggested:.3f} "
                    f"（+{s.improvement:.3f}）"
                )
        imp_thr = st.slider("只采纳改善量 ≥ 该值的建议", 0.0, 0.5, 0.05, 0.01, key="qf_imp")
        fixed = rep.apply(q, min_improvement=float(imp_thr))
        st.caption(
            f"应用后共修改 {int((np.asarray(q.matrix) != np.asarray(fixed.matrix)).any(axis=1).sum())} 题。"
            "新 Q 矩阵可在下方下载，导入后重新做结构可辨识性分析即可看到上限变化。"
        )
        st.download_button(
            "下载修正后的 Q 矩阵 (CSV)",
            _qmatrix_csv(fixed).encode("utf-8-sig"),
            file_name="q_matrix_fixed.csv", key="qf_dl",
        )


def _qmatrix_csv(q: QMatrix) -> str:
    """把 Q 矩阵序列化为 CSV 文本。"""
    df = pd.DataFrame(
        np.asarray(q.matrix, dtype=int),
        columns=list(q.attr_names),
        index=list(q.item_names) if q.item_names else None,
    )
    df.index.name = "item"
    return df.to_csv()


# ---------------------------------------------------------------------- #
# 页面：多级评分（顺序 G-DINA）
# ---------------------------------------------------------------------- #
def page_polytomous() -> None:
    hero(
        "多级评分 CDM（顺序 G-DINA）",
        "把 0/1/2/3 分的过程性评分建模为一串顺序步骤，"
        "在二分框架之外扩展可诊断的题型。",
        kicker="Polytomous Cognitive Diagnosis",
        tags=["顺序 G-DINA", "条件 EM", "档位概率", "参数恢复", "多级自适应"],
    )
    from cdcat.polytomous import (
        build_polytomous_bank,
        calibrate_polytomous,
        create_poly_selector,
    )

    section("1. 构造多级题库", "每题 2~4 档；属性即步骤")
    c1, c2, c3, c4 = st.columns(4)
    n_items = c1.number_input("题目数", 8, 60, 20, 2, key="poly_j")
    n_attrs = c2.number_input("属性数", 2, 6, 4, 1, key="poly_k")
    n_ex = c3.number_input("标定被试数", 200, 6000, 1500, 100, key="poly_n")
    seed = c4.number_input("随机种子", 0, 9999, 7, 1, key="poly_seed")

    st.caption("档位数默认在 2~4 之间循环；每题考查 1~2 个属性。"
               "顺序 G-DINA 假设「只有走到第 m−1 步的人才可能走到第 m 步」。")

    if st.button("生成题库并标定", key="poly_run", type="primary"):
        from cdcat.utils import enumerate_attribute_patterns

        with st.spinner("正在模拟作答并跑条件 EM ……"):
            bank = build_polytomous_bank(
                n_items=int(n_items), n_attrs=int(n_attrs), seed=int(seed)
            )
            pq, true_model = bank["pq"], bank["model"]
            rng = np.random.default_rng(int(seed) + 1)
            pats = enumerate_attribute_patterns(pq.n_attrs)
            idx = rng.integers(0, len(pats), size=int(n_ex))
            truth = pats[idx]
            resp = np.array([true_model.simulate(a, rng=rng) for a in truth], dtype=float)
            est, res = calibrate_polytomous(pq, resp, max_iter=300, tol=1e-6)
        st.session_state["poly_pack"] = {
            "pq": pq, "true": true_model, "est": est, "res": res,
            "resp": resp, "truth": truth,
        }

    pack = st.session_state.get("poly_pack")
    if pack is None:
        st.info("点击上方按钮开始。标定使用专用的**条件 EM** —— "
                "第 m 步的参数只用「已经走到该步」的被试，"
                "而不是把多级题朴素展开成二分题（那样会系统性低估参数）。")
        return

    pq, true_model, est, res = pack["pq"], pack["true"], pack["est"], pack["res"]

    section("2. 结构与拟合")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("题目数", f"{pq.n_items}")
    m2.metric("步骤数", f"{pq.n_steps}")
    m3.metric("平均档位", f"{pq.n_categories.mean():.2f}")
    m4.metric("迭代", f"{res.n_iterations}")
    m5.metric("BIC", f"{res.bic:,.0f}")

    section("3. 参数恢复", "步骤答对概率的真值 vs 估计")
    st.plotly_chart(fig_poly_step_recovery(est, true_model, pq), width="stretch")

    tp = np.asarray(true_model.step_probs(), dtype=float).ravel()
    ep = np.asarray(est.step_probs(), dtype=float).ravel()
    r = float(np.corrcoef(tp, ep)[0, 1])
    mae = float(np.abs(tp - ep).mean())
    k1, k2 = st.columns(2)
    k1.metric("步骤参数相关系数", f"{r:.4f}")
    k2.metric("步骤参数 MAE", f"{mae:.4f}")
    st.caption("恢复精度随被试数单调改善 —— 这正是 MMLE 的渐近性质。"
               "若相关系数长期停在 0.2 以下且不随样本量改善，说明模型参数化写错了。")

    section("4. 档位概率", "同一道题在不同知识状态下取各档位的概率")
    pick = st.selectbox("选择题目", list(range(pq.n_items)),
                        format_func=lambda i: f"第 {i + 1} 题（{int(pq.n_categories[i])} 档）",
                        key="poly_item")
    st.plotly_chart(fig_poly_category_probs(pq, est, item_idx=int(pick)),
                    width="stretch")
    st.caption("应看到「掌握得越多 → 越可能拿高分」的一阶随机占优关系："
               "高分段概率随知识状态单调上升，低分段概率单调下降。")

    with st.expander("各题步骤构成", expanded=False):
        rows = []
        for j in range(pq.n_items):
            s0, s1 = pq.step_slice(j)
            attrs = ["+".join(pq.attr_names[k] for k in np.where(pq.steps[m] > 0)[0])
                     for m in range(s0, s1)]
            rows.append({"题目": j + 1, "档位": int(pq.n_categories[j]),
                         "步骤（每步考查属性）": " → ".join(attrs)})
        st.dataframe(pd.DataFrame(rows), width="stretch")

    section("5. 多级自适应施测", "把多级模型直接接入既有的 CD-CAT 仿真器")
    c1, c2, c3 = st.columns(3)
    crit = c1.selectbox("选题准则", ["PolyPWKL", "PolyKL", "PolyGDI"], key="poly_crit")
    n_adm = c2.number_input("仿真被试数", 30, 1000, 200, 10, key="poly_nadm")
    reps = c3.number_input("重复次数", 1, 10, 2, 1, key="poly_rep")

    if st.button("开始仿真", key="poly_sim_btn"):
        with st.spinner("正在跑自适应施测 ……"):
            try:
                q_union = pq.union_q_matrix()
                sim = CDCATSimulator(
                    q_union, model=est,
                    selector=create_poly_selector(crit.replace("Poly", ""), est),
                    estimator="MAP",
                    termination=PatternConfidence(threshold=0.8),
                    min_items=5, max_items=min(pq.n_items, 30),
                )
                out = sim.run(n_examinees=int(n_adm), n_replications=int(reps), seed=0)
                st.session_state["poly_sim"] = out
            except Exception as exc:  # noqa: BLE001
                st.error(f"仿真失败：{exc}")
                return

    out = st.session_state.get("poly_sim")
    if out is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("PMR", f"{out.pmr():.4f}")
        m2.metric("AMR", f"{out.amr():.4f}")
        m3.metric("平均用题量", f"{out.avg_items():.2f}")
        st.caption("多级评分提供的额外信息通常让后验收敛得更快；"
                   "平均用题量往往低于同等的二分测验。")


# ---------------------------------------------------------------------- #
# 页面：一页式质量看板
# ---------------------------------------------------------------------- #
def _dashboard_run(
    bank: dict,
    model_name: str,
    fit_model: str,
    n_ex: int,
    cat_len: int,
    seed: int,
    hierarchy,
    do_ppmc: bool,
) -> dict:
    """跑一遍"体检"流水线，返回看板所需的全部中间结果。

    两条线并行产出，对应两类完全不同的质量问题：

    1. **自适应施测线** —— 用 CAT 引擎按当前题库实测一遍，回答"这个题库
       加上这套选题算法，到底能把人分得多准、平均用几道题"。
    2. **全题库标定线** —— 用完整作答矩阵做 MMLE/EM，回答"题目参数是否
       可恢复、模型与数据是否拟合、有没有局部依赖"。

    两条线共用同一个真值模型与同一个随机种子，因此数字之间可以互相印证。
    """
    from cdcat import posterior_predictive_check

    q: QMatrix = bank["q"]

    # ---------------- 线 1：自适应施测 ---------------- #
    rng = np.random.default_rng(int(seed))
    gen = _make_truth_model(model_name, q, bank)
    engine = model_name if model_name in ("DINA", "DINO") else "DINA"
    n_cat = int(min(cat_len, q.n_items))
    sim = CDCATSimulator(
        q,
        model=engine,
        model_params={"slip": bank["slip"], "guess": bank["guess"]},
        selector="PWKL",
        estimator="MAP",
        termination=FixedLength(n_cat),
        min_items=n_cat,
        max_items=n_cat,
        exposure_control=NoExposureControl(q.n_items),
    )
    sim_res = sim.run(n_examinees=int(n_ex), seed=int(seed), progress=False)
    cat_report = Evaluator().evaluate(sim_res, label="PWKL")

    # ---------------- 线 2：全题库标定 ---------------- #
    pool = q.attribute_patterns()
    if hierarchy is not None and not hierarchy.is_empty:
        pool = pool[hierarchy.legal_mask()]
    alphas = pool[rng.integers(0, len(pool), int(n_ex))]
    responses = np.array([gen.simulate(a, rng=rng) for a in alphas], dtype=float)

    analysis = run_cdm_analysis(
        q,
        responses,
        models=[fit_model],
        true_patterns=alphas,
        true_slip=bank["slip"],
        true_guess=bank["guess"],
        absolute_fit=True,
    )
    best = analysis.best_model or fit_model
    diag = analysis.diagnosis[best]
    fit = analysis.fits[best]

    af = analysis.absolute_fit
    item_fit = af.item_fit.to_dataframe() if af is not None else None
    ld = af.local_dependence if af is not None else None

    ident = analyse_identifiability(q)

    ppmc = None
    if do_ppmc:
        ppmc = posterior_predictive_check(
            analysis.model(best),
            responses,
            prior=analysis.models[best].attr_prior,
            n_replicates=200,
            seed=int(seed),
        )

    return {
        "q": q,
        "gen_model": model_name,
        "fit_model": best,
        "cat": cat_report,
        "cat_len": n_cat,
        "analysis": analysis,
        "diag": diag,
        "fit": fit,
        "responses": responses,
        "alphas": alphas,
        "item_fit": item_fit,
        "ld": ld,
        "ident": ident,
        "ppmc": ppmc,
        "n_ex": int(n_ex),
        "hierarchy": hierarchy,
        "seed": int(seed),
    }


def _dashboard_item_rows(pack: dict) -> list[list[str]]:
    """把题目级体检结果拼成表格行（含热力着色与判定徽章）。"""
    q: QMatrix = pack["q"]
    af_rows = pack["item_fit"]
    ld = pack["ld"]
    res = pack["analysis"].models[pack["fit_model"]]

    p_by_item, resid_by_item = {}, {}
    if af_rows is not None:
        for _, r in af_rows.iterrows():
            p_by_item[str(r["题号"])] = float(r["p值"])
            resid_by_item[str(r["题号"])] = float(r["最大残差"])

    q3_max = {}
    if ld is not None:
        mat = np.array(ld.q3, dtype=float, copy=True)
        np.fill_diagonal(mat, np.nan)
        for j in range(mat.shape[0]):
            q3_max[j] = float(np.nanmax(np.abs(mat[j]))) if mat.shape[1] > 1 else 0.0

    rows = []
    for j in range(q.n_items):
        name = q.item_names[j]
        slip = float(res.slip[j]) if res.slip is not None else float("nan")
        guess = float(res.guess[j]) if res.guess is not None else float("nan")
        disc = 1.0 - slip - guess
        p_val = p_by_item.get(name, float("nan"))
        q3 = q3_max.get(j, float("nan"))

        flags = []
        if np.isfinite(p_val) and p_val < 0.05:
            flags.append("拟合偏差")
        if np.isfinite(q3) and ld is not None and q3 > ld.threshold:
            flags.append("局部依赖")
        if np.isfinite(disc) and disc < 0.30:
            flags.append("区分度低")
        if not flags:
            verdict = pill("良好", "ok")
        elif len(flags) == 1:
            verdict = pill(flags[0], "warn")
        else:
            verdict = pill("＋".join(flags[:2]), "bad")

        rows.append([
            f'<span class="mono">{_esc(name)}</span>',
            _esc("+".join(q.attr_names[k] for k in q.required_attrs(j)) or "—"),
            heat_cell(slip, 0.0, 0.45, 3, invert=True),
            heat_cell(guess, 0.0, 0.45, 3, invert=True),
            heat_cell(disc, 0.0, 0.95, 3),
            heat_cell(p_val, 0.0, 1.0, 3),
            heat_cell(q3, 0.0, 0.35, 3, invert=True),
            verdict,
        ])
    return rows


def page_dashboard(bank: dict, model_name: str) -> None:
    hero(
        "一页式质量看板",
        "一次运行，把「这套题库 + 这套算法」的体检结论压到一屏："
        "上排是自适应施测的诊断精度，中间是知识地图，下排是逐题质量与拟合诊断。",
        kicker="Quality Dashboard",
        tags=["KPI 仪表", "知识地图", "逐题体检", "一键报告"],
    )

    q: QMatrix = bank["q"]

    # ---------------- 1. 体检设置 ---------------- #
    section("1. 体检设置", "同一随机种子下结果可复现")
    c1, c2, c3, c4 = st.columns(4)
    n_ex = c1.number_input("体检样本量", 100, 5000, 600, 100, key="dash_n")
    cat_len = c2.number_input(
        "自适应题量", 5, q.n_items, min(max(2 * q.n_attrs + 6, 10), q.n_items), 1,
        key="dash_len",
        help="每人只答这么多题；用题量越少，越能体现自适应选题的价值。",
    )
    fit_model = c3.selectbox(
        "标定模型", MODELS, index=MODELS.index(model_name) if model_name in MODELS else 0,
        key="dash_fit_model",
        help=f"数据按 {model_name} 生成。故意选错模型可以观察失配如何体现在拟合诊断上。",
    )
    seed = c4.number_input("随机种子", 0, 99999, 2024, 1, key="dash_seed")

    c5, c6 = st.columns([1.35, 1])
    with c5:
        hierarchy = _hierarchy_selector(q, key="dash_hier")
        if hierarchy is not None and not hierarchy.is_empty:
            st.caption(f"合法模式 {hierarchy.n_legal_patterns} / {q.n_patterns}")
    with c6:
        st.caption("层级约束既压缩合法模式空间，也直接抬高判准率上限。")
        do_ppmc = st.checkbox(
            "追加后验预测检验（PPMC）", value=False, key="dash_ppmc",
            help="用复制数据重估一遍再比统计量，比单看 χ² 更能识别模型误设；约多花 5–20 秒。",
        )

    if st.button("运行体检", type="primary", width="stretch", key="dash_run"):
        with st.spinner("正在施测、标定与诊断 ……"):
            try:
                st.session_state["dash_pack"] = _dashboard_run(
                    bank, model_name, fit_model, int(n_ex), int(cat_len),
                    int(seed), hierarchy, bool(do_ppmc),
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state["dash_pack"] = None
                st.error(f"体检失败：{exc}")

    pack = st.session_state.get("dash_pack")
    if not pack:
        st.info("点击上方「运行体检」生成看板。默认参数下约需 3–8 秒。")
        return

    cat = pack["cat"]
    diag = pack["diag"]
    fit = pack["fit"]
    ident = pack["ident"]
    ld = pack["ld"]
    item_fit = pack["item_fit"]

    # 题目级拟合合格率
    if item_fit is not None and len(item_fit):
        pass_rate = float((item_fit["p值"].to_numpy(dtype=float) >= 0.05).mean())
    else:
        pass_rate = float("nan")
    n_flagged = int(ld.n_flagged) if ld is not None else 0
    n_pairs = max(q.n_items * (q.n_items - 1) // 2, 1)

    # ---------------- 2. KPI 仪表 ---------------- #
    st.divider()
    section("2. 关键指标", f"样本量 {pack['n_ex']} 人 · 每人 {pack['cat_len']} 题")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(
            fig_quality_gauge(cat.pmr, "模式判准率 PMR", target=0.90),
            width="stretch",
        )
        st.caption(f"95% CI [{cat.pmr_ci[0]:.3f}, {cat.pmr_ci[1]:.3f}]")
    with g2:
        st.plotly_chart(
            fig_quality_gauge(cat.amr, "属性判准率 AMR", target=0.90),
            width="stretch",
        )
        st.caption(f"题库利用率 {cat.test_utilization:.1%}")
    with g3:
        st.plotly_chart(
            fig_quality_gauge(pass_rate, "题目拟合合格率", target=0.95),
            width="stretch",
        )
        st.caption("χ² 拟合 p ≥ .05 的题目占比")
    with g4:
        st.plotly_chart(
            fig_quality_gauge(diag.mean_confidence(), "平均分类信度", target=0.85),
            width="stretch",
        )
        st.caption("MAP 后验概率均值")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("平均用题量", f"{cat.avg_items:.1f}", f"共 {q.n_items} 题")
    m2.metric("全题库标定 PMR", f"{fit.pmr:.3f}" if fit.pmr is not None else "—")
    m3.metric("结构上限 PMR", f"{ident.expected_pmr_ceiling:.3f}")
    m4.metric("Q3 超标题目对", f"{n_flagged}", f"共 {n_pairs} 对")
    m5.metric("标定模型", pack["fit_model"])

    # ---------------- 3. 结论 ---------------- #
    issues = []
    if cat.pmr < 0.90:
        issues.append(f"自适应 PMR {cat.pmr:.3f} 低于 0.90")
    if np.isfinite(pass_rate) and pass_rate < 0.90:
        issues.append(f"{int(round((1 - pass_rate) * q.n_items))} 道题存在拟合偏差")
    if n_flagged:
        issues.append(f"{n_flagged} 对题目存在局部依赖")
    if ident.n_unidentifiable_patterns:
        issues.append(f"{ident.n_unidentifiable_patterns} 个模式结构上不可辨识")

    if not issues:
        cls, head = "cdc-note-good", "体检通过"
        body = ("诊断精度、题目拟合与题库结构三项均在阈值以内，"
                "该题库可用于正式施测。")
    elif len(issues) <= 2:
        cls, head = "cdc-note-warn", "需要关注"
        body = "；".join(issues) + "。建议先处理上面这些问题再上线。"
    else:
        cls, head = "cdc-note-bad", "建议整改"
        body = "；".join(issues) + "。当前状态下诊断结论的可靠性有限。"

    st.markdown(
        f'<div class="cdc-note {cls}" style="margin-top:14px;">'
        f"<b>{head}</b> —— {_esc(body)}</div>",
        unsafe_allow_html=True,
    )

    # ---------------- 4. 知识地图 ---------------- #
    st.divider()
    section("4. 知识地图", "节点颜色 = 掌握率；箭头 = 先备关系")
    from cdcat import AttributeHierarchy

    h_map = pack["hierarchy"]
    if h_map is None or h_map.is_empty:
        h_map = AttributeHierarchy.none(q.n_attrs, attr_names=q.attr_names)
        map_note = "当前未启用层级约束，属性视为彼此独立。"
    else:
        map_note = f"当前层级含 {h_map.n_legal_patterns} 个合法模式。"

    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(
            fig_attribute_map(
                h_map, diag.mastery_proportion(), q.attr_names,
                height=340 if h_map.is_empty else None,
            ),
            width="stretch",
        )
        st.caption(map_note + " 掌握率取全题库标定后的 MAP 知识状态。")
    with right:
        st.plotly_chart(
            fig_mastery_proportion(diag.mastery_proportion(), q.attr_names),
            width="stretch",
        )
        mastery = diag.mastery_proportion()
        weakest = int(np.argmin(mastery))
        st.caption(
            f"最薄弱属性：**{q.attr_names[weakest]}**（掌握率 {mastery[weakest]:.1%}），"
            "教学上应优先补强。"
        )

    # ---------------- 5. 逐题体检 ---------------- #
    st.divider()
    section("5. 逐题体检", "底色越绿越好；判定综合拟合、局部依赖与区分度")
    st.markdown(
        '<div class="cdc-scale"><span>差</span><span class="ramp"></span>'
        "<span>好</span><span style='margin-left:6px;'>（slip / guess / |Q3| 已反向：越低越绿）</span></div>",
        unsafe_allow_html=True,
    )
    render_table(
        ["题号", "考查属性", "slip", "guess", "区分度", "拟合 p", "|Q3|max", "判定"],
        _dashboard_item_rows(pack),
        align=["", "", "num", "num", "num", "num", "num", ""],
        max_height=352,   # ≈ 表头 + 8 整行，避免滚动容器底部切半行
    )
    st.caption(
        "区分度 = 1 − slip − guess；拟合 p 来自题目级 χ²/G² 检验；"
        "|Q3|max 是该题与其他题的最大 Yen Q3 绝对值。"
    )

    # ---------------- 6. 后验预测检验 ---------------- #
    ppmc = pack.get("ppmc")
    if ppmc is not None:
        section("6. 后验预测检验（PPMC）", "PPP 接近 0.5 表示复制数据与真实数据同分布")
        df = ppmc.to_dataframe()
        rows = []
        for _, r in df.iterrows():
            ppp = float(r["PPP"])
            kind = "bad" if r["判读"] == "失配" else "ok"
            rows.append([
                f'<span class="mono">{_esc(str(r["统计量"]))}</span>',
                f'{float(r["观测值"]):.4f}',
                f'{float(r["复制均值"]):.4f}',
                f'{float(r["复制 SD"]):.4f}',
                heat_cell(1.0 - abs(ppp - 0.5) * 2, 0.0, 1.0, 3),
                f"{ppp:.3f}",
                pill(str(r["判读"]), kind),
            ])
        render_table(
            ["统计量", "观测值", "复制均值", "复制 SD", "近似度", "PPP", "判读"],
            rows,
            align=["", "num", "num", "num", "num", "num", ""],
            max_height=260,
        )
        st.caption(ppmc.verdict())

    # ---------------- 7. 导出 ---------------- #
    st.divider()
    section("7. 导出", "报告为单文件 HTML，图表内联，可离线归档")
    e1, e2, e3 = st.columns([1, 1, 1])
    with e1:
        if st.button("生成 HTML 报告", width="stretch", key="dash_rep_btn"):
            try:
                from cdcat import build_cdm_report

                st.session_state["dash_report"] = build_cdm_report(
                    pack["analysis"],
                    pack["responses"],
                    title=f"{bank.get('name', '题库')} · 质量体检报告",
                    subtitle=f"{pack['n_ex']} 人 · 每人 {pack['cat_len']} 题 · "
                             f"{pack['gen_model']} 生成 / {pack['fit_model']} 标定",
                    author="CD-CAT 选题与仿真系统",
                ).render()
            except Exception as exc:  # noqa: BLE001
                st.error(f"报告生成失败：{exc}")

        html_report = st.session_state.get("dash_report")
        if html_report:
            st.download_button(
                "下载 HTML 报告",
                html_report.encode("utf-8"),
                file_name="cdcat_quality_report.html",
                mime="text/html",
                width="stretch",
            )
        else:
            st.caption("点击左侧按钮生成。")

    with e2:
        if item_fit is not None:
            st.download_button(
                "下载逐题体检 CSV",
                item_fit.to_csv(index=False).encode("utf-8-sig"),
                file_name="cdcat_item_check.csv",
                mime="text/csv",
                width="stretch",
            )
    with e3:
        if ld is not None:
            st.download_button(
                "下载 Q3 超标对 CSV",
                ld.flagged_pairs().to_csv(index=False).encode("utf-8-sig"),
                file_name="cdcat_q3_pairs.csv",
                mime="text/csv",
                width="stretch",
            )


# 主入口
# ---------------------------------------------------------------------- #
def main() -> None:
    cfg = sidebar_bank_selector()

    st.sidebar.divider()
    st.sidebar.header("功能导航")
    pages = ["质量看板", "在线施测", "蒙特卡洛仿真", "准则对比", "题库诊断",
             "群体报告", "参数标定", "多级评分", "系统说明"]
    page = st.sidebar.radio("选择页面", pages, label_visibility="collapsed", key="nav")
    st.sidebar.caption(
        f"界面主题：{'深色' if _IS_DARK else '浅色'}　·　"
        "深色启动：streamlit run demo/app.py --theme.base dark"
    )

    if not cfg:
        st.markdown(
            """
            <div class="cdc-hero">
                <div class="kicker">Cognitive Diagnostic CAT</div>
                <h1>CD-CAT 选题与仿真系统</h1>
                <p>认知诊断计算机自适应测验 —— 以更少的题目，精准识别每一位考生的知识结构。</p>
                <div class="tags">
                    <span class="cdc-tag">4 种响应模型</span>
                    <span class="cdc-tag">10 种选题准则</span>
                    <span class="cdc-tag">MMLE / EM 标定</span>
                    <span class="cdc-tag">绝对拟合与 Q3</span>
                    <span class="cdc-tag">属性层级 AHM</span>
                    <span class="cdc-tag">Q 矩阵自动修正</span>
                    <span class="cdc-tag">多级评分</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("请在左侧选择一个内置题库，或上传自己的 Q 矩阵。")

        section("系统能做什么", "八大功能模块")
        c1, c2, c3 = st.columns(3)
        feats = [
            ("1", "题库诊断", "校验 Q 矩阵完备性、属性覆盖、关联图连通性，"
                             "并给出<b>结构可辨识性</b>分析与 PMR 理论上限。"),
            ("2", "自适应施测", "10 种选题准则（含 4 种非参数）× 3 种估计方法，逐题动态选题，"
                              "实时展示后验收敛并导出个人诊断报告。"),
            ("3", "大规模仿真", "蒙特卡洛批量仿真，评估 PMR/AMR、用题效率、"
                              "题目曝光均匀性，支持 <b>n_jobs 多进程</b>加速与曝光控制。"),
            ("4", "准则对比", "在同一批被试、同一随机流下横向对比各选题准则，"
                            "选出最贴合你题库的方案。"),
            ("5", "CDM 参数标定", "MMLE/EM 估计 DINA / DINO / G-DINA / RRUM / ACDM / LLM，"
                                "含经验贝叶斯属性先验、信息准则择优与似然比检验。"),
            ("6", "绝对拟合诊断", "题目级 χ²/G² 绝对拟合与 <b>Yen Q3</b> 局部依赖，"
                                "识别模型误设与 Q 矩阵漏标的属性。"),
            ("7", "群体报告", "班级/群体掌握率排行、薄弱知识点、<b>确定性分档</b>、"
                            "分组对比、个体 vs 群体常模，并内置 <b>Q 矩阵自动修正</b>。"),
            ("8", "多级评分", "<b>顺序 G-DINA</b> 处理 0/1/2/3 分的过程性评分："
                            "专用条件 EM 标定、档位概率曲线、多级自适应施测。"),
        ]
        for col, (idx, h, d) in zip([c1, c2, c3] * 3, feats):
            with col:
                st.markdown(
                    f'<div class="cdc-feat"><div class="ic">{idx}</div>'
                    f'<div class="h">{h}</div><div class="d">{d}</div></div>',
                    unsafe_allow_html=True,
                )
                st.write("")
        return

    bank = cfg["bank"]
    model_name = cfg["model"]

    if page == "质量看板":
        page_dashboard(bank, model_name)
    elif page == "在线施测":
        page_administer(bank, model_name)
    elif page == "蒙特卡洛仿真":
        page_simulation(bank, model_name)
    elif page == "准则对比":
        page_compare(bank, model_name)
    elif page == "题库诊断":
        page_qmatrix(bank)
    elif page == "群体报告":
        page_group(bank, model_name)
    elif page == "参数标定":
        page_calibration(bank)
    elif page == "多级评分":
        page_polytomous()
    else:
        page_about(bank)


def page_about(bank: dict) -> None:
    hero("系统说明", "CD-CAT 的方法学基础、系统已实现的能力清单，以及推荐的完整使用流程。",
         kicker="Documentation",
         tags=["方法学", "使用流程", "参考文献"])

    q = bank["q"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("题目数 J", q.n_items)
    m2.metric("属性数 K", q.n_attrs)
    m3.metric("模式总数 2^K", q.n_patterns)
    m4.metric("选项目数", len(CRITERIA))

    st.markdown(
        f"""
### 什么是 CD-CAT

传统 CAT 估计的是连续能力值 θ，而**认知诊断自适应测验 (CD-CAT)** 估计的是
**离散的属性掌握模式 α**（例如"会通分、不会约分、会分数加减"），
目标是以尽可能少的题目精准识别每个人的知识结构。

### 本系统包含的方法

| 环节 | 已实现方法 |
|---|---|
| 响应模型 | DINA（合取）、DINO（析取）、G-DINA（饱和）、RRUM（精简重参数化）、ACDM（加性）、LLM（线性 logistic） |
| 估计方法 | MAP（最大后验）、EAP（期望后验）、MLE（最大似然） |
| 选题准则 | 参数化：PWKL、MPWKL、KL、GDI、SHE、CDI；非参数（不需要题目参数）：MIV、NPSHE、ERHD、AMI |
| 终止规则 | 模式级置信、后验阈值、固定题量、估计稳定性、属性熵、复合规则 |
| 曝光控制 | 硬上限、随机化选题、分层曝光、Sympson-Hetter（含自动标定） |
| 参数标定 | DINA/DINO 的 MMLE/EM 算法 |
| 题库分析 | Q 矩阵诊断、结构可辨识性、题目区分度 / 信息量 / 冗余检测 |
| 结果报告 | 个人诊断报告、评估报告、被试明细、对比总表 |

### 当前题库

- 名称：**{bank['name']}**
- 说明：{bank.get('description', '—')}
- 规模：{q.n_items} 题 × {q.n_attrs} 属性（{q.n_patterns} 种模式）
- 属性：{', '.join(bank['attrs'])}

### 推荐使用流程

1. **题库诊断** —— 检查 Q 矩阵完备性、属性覆盖、关联图连通性，并查看**结构可辨识性**与 PMR 理论上限。
2. **题目分析** —— 在同页评估每道题的区分度与信息量，检出冗余题目。
3. **参数标定** —— 用历史作答数据估计题目参数（真实上线必备；演示可用预设参数）。
4. **在线施测** —— 走一遍单被试流程，确认选题与终止行为符合预期，导出个人报告。
5. **准则对比** —— 在固定题量下横向比较各准则，选出最适合本题库的准则。
6. **蒙特卡洛仿真** —— 用选定方案做最终评估，确认 PMR/AMR、曝光指标与结构上限达成情况。

### 关键设计理念

- **结构可辨识性优先**：先确认 Q 矩阵理论上能区分哪些模式，再评估算法表现。实际 PMR 应与理论上限对比，而非盲目追求 1.0。
- **终止规则要兼顾模式级与属性级**：仅看属性边际置信会过早停止，本系统默认使用「模式置信 + 属性边际」双重判据。
- **曝光控制不可省略**：高区分度题目会被反复选中，导致题库过载与题目泄露，务必启用曝光控制。
"""
    )

    section("参考文献")
    st.markdown(
        """
- Xu, Chang & Douglas (2003). *A simulation study to compare CAT strategies for cognitive diagnosis.*
- Cheng (2009). *When cognitive diagnosis meets computerized adaptive testing: CD-CAT.*
- Kaplan, de la Torre & Barrada (2015). *New item selection methods for cognitive diagnosis computerized adaptive testing.*
- de la Torre (2011). *The generalized DINA model framework.*
- Ma & de la Torre (2020). *GDINA: An R package for cognitive diagnosis modeling.*
"""
    )


if __name__ == "__main__":
    main()
