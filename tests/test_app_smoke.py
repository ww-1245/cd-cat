"""Streamlit 界面冒烟测试：确保全部 8 个页面均可无异常渲染。

使用 streamlit.testing.v1.AppTest 在无浏览器环境下驱动 app.py，
捕获页面级运行时异常（AttributeError / KeyError / 类型错误等）。

这些测试依赖 streamlit，若未安装则整体跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "demo" / "app.py")

#: 侧边栏的全部功能页（与 demo/app.py 的 ``pages`` 保持一致）
NAV = [
    "质量看板", "在线施测", "蒙特卡洛仿真", "准则对比", "题库诊断",
    "群体报告", "参数标定", "多级评分", "系统说明",
]


def _fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _no_exception(at: AppTest) -> None:
    assert not at.exception, [str(e) for e in at.exception]


def test_app_boots_without_exception() -> None:
    """应用首次加载不应崩溃（题库默认选中，直接进入首页）。"""
    at = _fresh()
    _no_exception(at)


def test_all_pages_render() -> None:
    """依次切换全部功能页，确保每页都能渲染。"""
    at = _fresh()
    _no_exception(at)

    # 侧边栏的页面选择器是最后一个 radio 组件
    pages = NAV
    for name in pages:
        target = None
        for radio in at.sidebar.radio:
            if set(radio.options) == set(pages):
                target = radio
                break
        assert target is not None, "未找到页面导航组件"
        target.set_value(name)
        at.run()
        _no_exception(at)


def test_diagnosis_page_shows_identifiability() -> None:
    """题库诊断页应输出可辨识性指标（PMR 理论上限）。"""
    at = _fresh()
    pages = NAV
    for radio in at.sidebar.radio:
        if set(radio.options) == set(pages):
            radio.set_value("题库诊断")
            break
    at.run()
    _no_exception(at)

    labels = [m.label for m in at.metric]
    assert "PMR 理论上限" in labels


def test_administer_flow() -> None:
    """在线施测页点击「开始新的施测」后应产出结果指标。"""
    at = _fresh()
    pages = NAV
    for radio in at.sidebar.radio:
        if set(radio.options) == set(pages):
            radio.set_value("在线施测")
            break
    at.run()

    buttons = [b for b in at.button if "开始新的施测" in b.label]
    assert buttons, "未找到施测按钮"
    buttons[0].click()
    at.run()
    _no_exception(at)

    labels = [m.label for m in at.metric]
    assert "作答题目数" in labels


def test_about_page_renders() -> None:
    """系统说明页渲染，且包含当前题库信息。"""
    at = _fresh()
    pages = NAV
    for radio in at.sidebar.radio:
        if set(radio.options) == set(pages):
            radio.set_value("系统说明")
            break
    at.run()
    _no_exception(at)


def _goto(at: AppTest, name: str) -> None:
    pages = NAV
    for radio in at.sidebar.radio:
        if set(radio.options) == set(pages):
            radio.set_value(name)
            break
    at.run()


def test_calibration_page_initial_state() -> None:
    """参数标定页初始应提示先生成数据，且不报错。"""
    at = _fresh()
    _goto(at, "参数标定")
    _no_exception(at)
    infos = " ".join(i.value for i in at.info)
    assert "请先生成合成数据或上传作答矩阵" in infos


def test_calibration_full_flow() -> None:
    """完整跑通：生成合成数据 -> 多模型标定 -> 拟合指标/诊断结果输出。"""
    at = _fresh()
    _goto(at, "参数标定")

    # 缩小规模以保证测试速度
    at.number_input(key="cal_n").set_value(400)
    at.run()

    gen = [b for b in at.button if b.key == "cal_gen_btn"]
    assert gen, "未找到「生成合成数据」按钮"
    gen[0].click()
    at.run()
    _no_exception(at)

    run = [b for b in at.button if b.key == "cal_run"]
    assert run, "未找到「开始标定与分析」按钮"
    run[0].click()
    at.run()
    _no_exception(at)

    # 标定完成后应出现结果区块，且拟合指标表已渲染
    labels = [m.label for m in at.metric]
    assert labels, "标定后未渲染任何指标"
    assert any("最优" in str(x) or "模型" in str(x) for x in labels) or at.dataframe
    assert at.dataframe, "未渲染模型拟合指标表"

    # 非参数基线：应有独立表格，且在页面上有对应说明
    md = " ".join(str(m.value) for m in at.markdown)
    assert "非参数基线" in md, "未渲染非参数基线说明"
    assert len(at.dataframe) >= 2, "未渲染非参数基线表"


def test_calibration_all_six_models() -> None:
    """六模型同时标定不应崩溃（含 ACDM / LLM 两个加性扩展模型）。"""
    at = _fresh()
    _goto(at, "参数标定")

    # 先生成数据，配置区（含模型多选）才会渲染
    at.number_input(key="cal_n").set_value(300)
    at.run()
    gen = [b for b in at.button if b.key == "cal_gen_btn"]
    assert gen
    gen[0].click()
    at.run()
    _no_exception(at)

    at.multiselect(key="cal_models").set_value(
        ["DINA", "DINO", "G-DINA", "RRUM", "ACDM", "LLM"]
    )
    at.run()

    run = [b for b in at.button if b.key == "cal_run"]
    assert run
    run[0].click()
    at.run()
    _no_exception(at)
    assert at.dataframe, "六模型标定未渲染拟合指标表"


def test_calibration_absolute_fit_section() -> None:
    """标定后应渲染「绝对拟合与局部依赖诊断」区块及其指标。"""
    at = _fresh()
    _goto(at, "参数标定")
    at.number_input(key="cal_n").set_value(300)
    at.run()
    [b for b in at.button if b.key == "cal_gen_btn"][0].click()
    at.run()
    [b for b in at.button if b.key == "cal_run"][0].click()
    at.run()
    _no_exception(at)

    labels = [m.label for m in at.metric]
    for expected in ("不合格题目数", "Q3 超标题目对", "平均 Q3"):
        assert expected in labels, f"缺少指标 {expected}；实际: {labels}"

    # 区块标题现由 section() 渲染为自定义 markdown（带强调竖条），不再是 st.subheader
    md = " ".join(str(m.value) for m in at.markdown)
    assert "绝对拟合与局部依赖诊断" in md, "缺少「绝对拟合与局部依赖诊断」区块"


def test_plotly_charts_buildable_from_pipeline() -> None:
    """demo.cdm_charts 的所有图都可由一条流水线结果构造。"""
    pytest.importorskip("plotly")
    import numpy as np

    from cdcat import QMatrix, run_cdm_analysis
    from cdcat.utils import build_test_items
    from demo import cdm_charts as cc

    q = QMatrix.from_random(12, 3, seed=9)
    items = build_test_items(12, 3, seed=9)
    from cdcat import DINA

    model = DINA(q, slip=items["slip"], guess=items["guess"])
    rng = np.random.default_rng(9)
    alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, 300)]
    X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)

    analysis = run_cdm_analysis(
        q, X, models=("DINA", "G-DINA"),
        true_slip=items["slip"], true_guess=items["guess"],
        true_patterns=alphas, max_iter=100,
    )
    best = analysis.best_model
    res = analysis.models[best]
    diag = analysis.diagnosis[best]

    figs = [
        cc.fig_ll_convergence(analysis.models),
        cc.fig_param_recovery(res.slip, items["slip"]),
        cc.fig_item_discrimination(1.0 - res.slip - res.guess),
        cc.fig_fit_comparison(analysis.fits),
        cc.fig_mastery_proportion(diag.mastery_proportion()),
        cc.fig_attr_reliability(diag.attr_reliability),
        cc.fig_pattern_distribution(diag),
    ]
    for f in figs:
        assert f is not None and len(f.data) > 0


# ---------------------------------------------------------------------- #
# 新增页面：群体报告 / 多级评分
# ---------------------------------------------------------------------- #
def _click(at: AppTest, key: str) -> None:
    btn = [b for b in at.button if b.key == key]
    assert btn, f"未找到按钮 {key}"
    btn[0].click()
    at.run()


def _metric_labels(at: AppTest) -> list:
    return [m.label for m in at.metric]


def test_group_page_initial_state() -> None:
    """群体报告页初始应提示先准备数据。"""
    at = _fresh()
    _goto(at, "群体报告")
    _no_exception(at)
    infos = " ".join(i.value for i in at.info)
    assert "请先生成合成数据" in infos


def test_group_page_full_flow() -> None:
    """生成合成数据 -> 出群体报告 -> 渲染掌握率/确定性/典型模式等指标。"""
    at = _fresh()
    _goto(at, "群体报告")

    at.number_input(key="grp_n").set_value(120)
    at.run()
    _click(at, "grp_gen")

    _click(at, "grp_run")
    _no_exception(at)

    labels = _metric_labels(at)
    for expected in ("有效被试数", "人均掌握属性数", "完全掌握人数",
                     "待定判定占比", "属性判准率 AMR", "模式判准率 PMR"):
        assert expected in labels, f"缺少指标 {expected}；实际: {labels}"
    assert at.dataframe, "未渲染属性明细表"


def test_group_page_with_hierarchy() -> None:
    """选择属性层级后，合法模式数应被压缩，且不报错。"""
    at = _fresh()
    _goto(at, "群体报告")
    at.number_input(key="grp_n").set_value(120)
    at.run()
    _click(at, "grp_gen")

    at.selectbox(key="grp_hier_kind").set_value("线型（逐级先备）")
    at.run()
    _click(at, "grp_run")
    _no_exception(at)
    assert "有效被试数" in _metric_labels(at)


def test_group_page_qmatrix_fix_block() -> None:
    """Q 矩阵自动修正区块应能跑出 PVAF 指标。"""
    at = _fresh()
    _goto(at, "群体报告")
    at.number_input(key="grp_n").set_value(120)
    at.run()
    _click(at, "grp_gen")
    _click(at, "grp_run")
    _click(at, "qf_run")
    _no_exception(at)

    labels = _metric_labels(at)
    for expected in ("疑似需修正题目", "全库平均 PVAF"):
        assert expected in labels, f"缺少指标 {expected}；实际: {labels}"


def test_polytomous_page_initial_state() -> None:
    """多级评分页初始应提示点击按钮开始。"""
    at = _fresh()
    _goto(at, "多级评分")
    _no_exception(at)
    infos = " ".join(i.value for i in at.info)
    assert "点击上方按钮开始" in infos


def test_polytomous_page_full_flow() -> None:
    """生成多级题库 -> 条件 EM 标定 -> 参数恢复 ——> 多级自适应仿真。"""
    at = _fresh()
    _goto(at, "多级评分")

    at.number_input(key="poly_j").set_value(10)
    at.number_input(key="poly_k").set_value(3)
    at.number_input(key="poly_n").set_value(250)
    at.run()

    _click(at, "poly_run")
    _no_exception(at)
    labels = _metric_labels(at)
    for expected in ("题目数", "步骤数", "迭代", "BIC",
                     "步骤参数相关系数", "步骤参数 MAE"):
        assert expected in labels, f"缺少指标 {expected}；实际: {labels}"

    # 多级自适应仿真
    at.number_input(key="poly_nadm").set_value(40)
    at.number_input(key="poly_rep").set_value(1)
    at.run()
    _click(at, "poly_sim_btn")
    _no_exception(at)
    labels = _metric_labels(at)
    for expected in ("PMR", "AMR", "平均用题量"):
        assert expected in labels, f"缺少指标 {expected}；实际: {labels}"


# ---------------------------------------------------------------------- #
# 一页式质量看板
# ---------------------------------------------------------------------- #
def test_dashboard_initial_state() -> None:
    """看板首屏应提示点击运行体检。"""
    at = _fresh()
    _goto(at, "质量看板")
    _no_exception(at)
    infos = " ".join(i.value for i in at.info)
    assert "运行体检" in infos


def test_dashboard_full_flow() -> None:
    """运行体检 -> 四张仪表 + 指标 + 逐题体检表 + 知识地图。"""
    at = _fresh()
    _goto(at, "质量看板")

    # 缩小样本量，避免测试跑太久
    at.number_input(key="dash_n").set_value(300)
    at.number_input(key="dash_len").set_value(12)
    at.run()

    _click(at, "dash_run")
    _no_exception(at)

    labels = _metric_labels(at)
    for expected in ("平均用题量", "结构上限 PMR", "Q3 超标题目对", "标定模型"):
        assert expected in labels, f"缺少指标 {expected}；实际: {labels}"

    # 仪表盘是 indicator 图，知识地图与掌握率图是 plotly 图
    charts = at.get("plotly_chart")
    assert len(charts) >= 4, f"图表数量不足：{len(charts)}"

    # 逐题体检表与热力单元格
    md = " ".join(m.value for m in at.markdown)
    assert "cdc-tbl" in md, "未渲染逐题体检表"
    assert "cdc-heat" in md, "表格缺少条件着色"


def test_dashboard_html_report_button() -> None:
    """看板里的「生成 HTML 报告」应产出可下载的自包含文件。"""
    at = _fresh()
    _goto(at, "质量看板")
    at.number_input(key="dash_n").set_value(250)
    at.number_input(key="dash_len").set_value(12)
    at.run()
    _click(at, "dash_run")
    _no_exception(at)

    _click(at, "dash_rep_btn")
    _no_exception(at)
    report = at.session_state["dash_report"]
    assert isinstance(report, str) and "<style" in report
    # 自包含：不得引用外部资源
    assert "http://" not in report and "https://" not in report


def test_compare_page_with_miv() -> None:
    """准则对比页：无参数准则 MIV 与参数化准则同场对比，应正常出表与图。

    这一页此前没有端到端覆盖；MIV 加进可选列表后，顺带把这条路径钉住，
    防止「注册表里有、但界面选了就崩」这类问题溜过去。
    """
    at = _fresh()
    _goto(at, "准则对比")
    _no_exception(at)

    # 缩小规模控制耗时：50 被试 × 1 次重复
    at.multiselect[0].set_value(["PWKL", "MIV"])
    at.number_input(key="cmp_n").set_value(50)
    at.number_input(key="cmp_rep").set_value(1)
    at.run()

    _click(at, "cmp_run")
    _no_exception(at)

    md = " ".join(m.value for m in at.markdown)
    assert "对比总表" in md, "未渲染对比总表"
    assert len(at.dataframe) >= 1, "对比总表为空"
    assert len(at.get("plotly_chart")) >= 2, "效率-精度散点 / 分组柱状图缺失"
