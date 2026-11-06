"""自包含 HTML 报告测试。

重点：产物必须**真的自包含**（没有外链、没有 script），以及用户输入被正确转义。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cdcat import DINA, HTMLReport, QMatrix, build_cdm_report, run_cdm_analysis


# ====================================================================== #
# 工具
# ====================================================================== #
def _analysis(n=400, J=16, K=4, seed=5):
    rng = np.random.default_rng(seed)
    m = np.zeros((J, K), dtype=int)
    for j in range(J):
        m[j, rng.choice(K, size=1 + (j % 2), replace=False)] = 1
    q = QMatrix(m, attr_names=[f"A{i + 1}" for i in range(K)])
    model = DINA(q, slip=rng.uniform(0.05, 0.2, J), guess=rng.uniform(0.05, 0.25, J))
    alpha = rng.integers(0, 2, size=(n, K))
    w = 1 << np.arange(K - 1, -1, -1)
    X = (rng.random((n, J)) < model._get_prob()[(alpha * w).sum(1)]).astype(float)
    return run_cdm_analysis(q, X, models=("DINA",)), X


def _no_external(html: str) -> bool:
    return ("http://" not in html and "https://" not in html
            and "<script" not in html.lower() and "<link" not in html.lower())


# ====================================================================== #
# 1. 低层构件
# ====================================================================== #
class TestHTMLReport:
    def test_skeleton_and_self_contained(self):
        rep = HTMLReport("测试报告", subtitle="副标题", author="某单位")
        rep.heading("第一节")
        rep.paragraph("正文")
        html = rep.render()
        assert html.startswith("<!DOCTYPE html>")
        assert 'lang="zh-CN"' in html
        assert "<title>测试报告</title>" in html
        assert "副标题" in html and "某单位" in html
        assert _no_external(html)

    def test_kpis(self):
        rep = HTMLReport("t")
        rep.kpis([("被试数", 1200, "人", "accent"), ("PMR", "0.92", None, "ok")])
        html = rep.render()
        assert "1200" in html and "0.92" in html
        assert html.count('class="kpi') >= 2
        assert "kpi accent" in html and "kpi ok" in html

    def test_kpi_tolerates_short_tuples(self):
        rep = HTMLReport("t")
        rep.kpis([("只有一个值", 3)])
        assert "只有一个值" in rep.render()

    def test_table_from_dataframe(self):
        pd = pytest.importorskip("pandas")
        rep = HTMLReport("t")
        rep.heading("表")
        rep.table(pd.DataFrame({"题目": ["I1", "I2"], "slip": [0.1, 0.2]}))
        html = rep.render()
        assert "<table" in html and "slip" in html
        assert 'class="num"' in html            # 数值列右对齐

    def test_table_truncation_note(self):
        pd = pytest.importorskip("pandas")
        rep = HTMLReport("t")
        rep.table(pd.DataFrame({"v": range(100)}), max_rows=10)
        html = rep.render()
        assert html.count("<tr>") == 11         # 表头 + 10 行
        assert "仅显示前 10 行" in html

    def test_table_highlight(self):
        pd = pytest.importorskip("pandas")
        rep = HTMLReport("t")
        rep.table(
            pd.DataFrame({"题目": ["I1", "I2"], "区分度": [0.1, 0.9]}),
            highlight={"区分度": lambda v: "bad" if isinstance(v, float) and v < 0.3 else None},
        )
        html = rep.render()
        assert "bad" in html

    def test_table_from_pairs(self):
        rep = HTMLReport("t")
        rep.table((["a", "b"], [[1, 2], [3, 4]]))
        assert "<th>a</th>" in rep.render()

    def test_escaping(self):
        rep = HTMLReport("<script>alert(1)</script>")
        rep.heading("<b>标题</b>")
        rep.paragraph("<img src=x onerror=alert(2)>")
        rep.callout("<script>x</script>", "warn")
        html = rep.render()
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
        assert "onerror" in html and "<img src=x" not in html

    def test_callout_levels(self):
        rep = HTMLReport("t")
        for lv in ("info", "ok", "warn", "bad"):
            rep.callout("文本", lv)
        html = rep.render()
        assert "callout warn" in html and "callout bad" in html

    def test_figure_inlined(self):
        rep = HTMLReport("t")
        plt = pytest.importorskip("matplotlib.pyplot")
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [1, 0])
        rep.figure(fig, "示例图")
        html = rep.render()
        assert html.count("data:image/png;base64,") == 1
        assert "示例图" in html
        assert _no_external(html)

    def test_image_from_bytes(self, tmp_path):
        rep = HTMLReport("t")
        png = (b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        rep.image(png, "字节图")
        assert "data:image/png;base64," in rep.render()

        p = tmp_path / "a.jpg"
        p.write_bytes(png)
        rep.image(str(p), "文件图")
        assert "data:image/jpeg;base64," in rep.render()

    def test_save_creates_dirs(self, tmp_path):
        rep = HTMLReport("t")
        rep.heading("节")
        out = tmp_path / "deep" / "nested" / "r.html"
        rep.save(str(out))
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_sections_and_toc(self):
        rep = HTMLReport("t")
        for i in range(3):
            rep.heading(f"第{i + 1}节")
        html = rep.render()
        assert rep.n_sections == 3
        assert html.count('<section class="card"') == 3
        assert html.count('<nav class="toc"') == 1
        assert "第3节" in html

    def test_default_section_when_none(self):
        rep = HTMLReport("t")
        rep.paragraph("没有显式开节")
        assert rep.n_sections == 1


# ====================================================================== #
# 2. 高层报告
# ====================================================================== #
class TestBuildCDMReport:
    def test_sections_present(self):
        analysis, X = _analysis()
        rep = build_cdm_report(analysis, X, title="诊断报告", subtitle="示例")
        html = rep.render()
        assert "概览" in html
        assert "模型比较" in html
        assert "被试诊断" in html
        assert "口径说明" in html
        assert _no_external(html)

    def test_figures_optional(self):
        analysis, X = _analysis()
        without = build_cdm_report(analysis, X, include_figures=False)
        assert without.render().count("data:image/png;base64,") == 0
        with_fig = build_cdm_report(analysis, X, include_figures=True)
        assert with_fig.render().count("data:image/png;base64,") >= 1

    def test_missing_data_warning(self):
        analysis, X = _analysis()
        x = X.copy()
        x[::20, 0] = np.nan
        rep = build_cdm_report(analysis, x, include_figures=False)
        assert "作答率" in rep.render()

    def test_no_responses(self):
        analysis, _ = _analysis()
        rep = build_cdm_report(analysis, None, include_figures=False)
        assert "被试数" in rep.render()

    def test_save_and_size(self, tmp_path):
        analysis, X = _analysis()
        rep = build_cdm_report(analysis, X, include_figures=False)
        out = tmp_path / "r.html"
        rep.save(str(out))
        assert out.stat().st_size > 2000
        assert _no_external(out.read_text(encoding="utf-8"))


# ====================================================================== #
# 3. 数值格式化
# ====================================================================== #
class TestFormatting:
    def test_fmt(self):
        from cdcat.report import _fmt

        assert _fmt(None) == "—"
        assert _fmt(float("nan")) == "—"
        assert _fmt(3.0) == "3"
        assert _fmt(0.123456) == "0.1235"
        assert _fmt("abc") == "abc"
