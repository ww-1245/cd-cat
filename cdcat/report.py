"""自包含 HTML 报告：单文件、离线可看、可直接发邮件。

为什么要自包含
--------------
报告一旦要发给学校、教研组或存档，最怕"图片丢了""样式没加载"。本模块把
CSS 写在 ``<style>`` 里、把 matplotlib 图**内联成 base64 PNG**，产出的就是
一个独立的 ``.html``——拷到任何机器上双击都能看，不需要网络，不需要附带
文件夹，也不会因为换了个浏览器就排版崩掉。

为此**不用 Plotly**：plotly.js 有好几 MB，内联进去要么文件巨大，要么就得
依赖 CDN（那就不叫离线自包含了）。静态图用 matplotlib 渲染成 PNG 更实在。

用法
----
低层按块拼装::

    rep = HTMLReport("某校诊断报告", subtitle="2024 春季期末")
    rep.kpis([("被试数", "1,200"), ("题量", "40")])
    rep.heading("模型比较")
    rep.table(df)
    rep.figure(fig, "各属性掌握率")
    rep.save("report.html")

高层一次成文::

    build_cdm_report(analysis, responses=X).save("report.html")
"""

from __future__ import annotations

import base64
import html as _html
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["HTMLReport", "build_cdm_report"]

#: 与 Web 界面一致的语义配色
_ACCENT = "#4F46E5"
_ACCENT_SOFT = "#EEF2FF"
_INK = "#111827"
_MUTED = "#6B7280"
_LINE = "#E5E7EB"
_BG = "#F8FAFC"

_CSS = f"""
:root {{
  --ink: {_INK};
  --muted: {_MUTED};
  --line: {_LINE};
  --bg: {_BG};
  --accent: {_ACCENT};
  --accent-soft: {_ACCENT_SOFT};
  --ok: #0D9488;
  --warn: #B45309;
  --bad: #BE123C;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
               "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif;
  font-size: 15px;
  line-height: 1.7;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 72px; }}
header.hero {{
  background: linear-gradient(135deg, #312E81 0%, {_ACCENT} 55%, #6366F1 100%);
  color: #fff; border-radius: 18px; padding: 34px 32px; margin-bottom: 26px;
  box-shadow: 0 12px 30px -12px rgba(49, 46, 129, .55);
}}
header.hero h1 {{ margin: 0 0 6px; font-size: 25px; letter-spacing: .3px; }}
header.hero .sub {{ font-size: 14px; opacity: .88; margin: 0; }}
header.hero .meta {{ margin-top: 16px; font-size: 12.5px; opacity: .75; }}
.layout {{ display: grid; grid-template-columns: 216px minmax(0, 1fr); gap: 26px; }}
nav.toc {{
  position: sticky; top: 24px; align-self: start;
  background: #fff; border: 1px solid var(--line); border-radius: 14px;
  padding: 14px 12px; font-size: 13px;
}}
nav.toc .t {{ font-weight: 700; color: var(--muted); font-size: 11px;
  letter-spacing: .12em; text-transform: uppercase; margin: 2px 8px 10px; }}
nav.toc a {{ display: block; padding: 6px 9px; border-radius: 8px;
  color: #374151; text-decoration: none; }}
nav.toc a:hover {{ background: var(--accent-soft); color: var(--accent); }}
main {{ min-width: 0; }}
section.card {{
  background: #fff; border: 1px solid var(--line); border-radius: 14px;
  padding: 22px 24px; margin-bottom: 20px;
}}
section.card > h2 {{
  margin: 0 0 4px; font-size: 17px; letter-spacing: .2px;
  padding-bottom: 12px; border-bottom: 1px solid var(--line);
}}
section.card > h2 .n {{
  display: inline-block; min-width: 22px; height: 22px; line-height: 22px;
  text-align: center; background: var(--accent-soft); color: var(--accent);
  border-radius: 6px; font-size: 12px; margin-right: 9px; font-weight: 700;
}}
p.note {{ color: var(--muted); font-size: 13.5px; margin: 10px 0 0; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
  gap: 12px; margin: 16px 0 4px; }}
.kpi {{ border: 1px solid var(--line); border-radius: 12px; padding: 13px 15px;
  background: linear-gradient(180deg, #fff, #FBFCFE); }}
.kpi .l {{ font-size: 11.5px; color: var(--muted); letter-spacing: .04em; }}
.kpi .v {{ font-size: 22px; font-weight: 700; margin-top: 3px;
  font-variant-numeric: tabular-nums; }}
.kpi .h {{ font-size: 11.5px; color: var(--muted); margin-top: 2px; }}
.kpi.ok .v {{ color: var(--ok); }}
.kpi.warn .v {{ color: var(--warn); }}
.kpi.bad .v {{ color: var(--bad); }}
.kpi.accent .v {{ color: var(--accent); }}
.tblwrap {{ overflow-x: auto; margin-top: 14px; }}
table.tbl {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
table.tbl th, table.tbl td {{
  padding: 8px 11px; border-bottom: 1px solid var(--line); white-space: nowrap;
}}
table.tbl thead th {{
  position: sticky; top: 0; background: #F9FAFB; text-align: left;
  font-weight: 600; color: #374151; font-size: 12.5px;
  border-bottom: 1px solid #D1D5DB;
}}
table.tbl tbody tr:hover {{ background: #FAFBFF; }}
table.tbl td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.tbl td.bad {{ color: var(--bad); font-weight: 600; }}
table.tbl td.ok {{ color: var(--ok); }}
figure.fig {{ margin: 16px 0 0; }}
figure.fig img {{ width: 100%; height: auto; border: 1px solid var(--line);
  border-radius: 10px; display: block; }}
figure.fig figcaption {{ font-size: 12.5px; color: var(--muted);
  margin-top: 8px; text-align: center; }}
.callout {{ border-left: 3px solid var(--accent); background: var(--accent-soft);
  padding: 11px 15px; border-radius: 0 10px 10px 0; margin: 14px 0 0;
  font-size: 13.5px; }}
.callout.warn {{ border-color: var(--warn); background: #FFFBEB; }}
.callout.bad {{ border-color: var(--bad); background: #FFF1F2; }}
.callout.ok {{ border-color: var(--ok); background: #F0FDFA; }}
footer {{ text-align: center; color: var(--muted); font-size: 12px;
  margin-top: 30px; }}
@media (max-width: 860px) {{
  .layout {{ grid-template-columns: 1fr; }}
  nav.toc {{ position: static; }}
}}
@media print {{
  body {{ background: #fff; font-size: 11.5pt; }}
  .wrap {{ max-width: none; padding: 0; }}
  nav.toc {{ display: none; }}
  .layout {{ display: block; }}
  section.card {{ border: none; padding: 0 0 14px; break-inside: avoid; }}
  header.hero {{ box-shadow: none; border-radius: 0; }}
  figure.fig {{ break-inside: avoid; }}
}}
"""


# ---------------------------------------------------------------------- #
# 低层构件
# ---------------------------------------------------------------------- #
@dataclass
class _Section:
    anchor: str
    title: str
    blocks: List[str] = field(default_factory=list)


class HTMLReport:
    """按块拼装的自包含 HTML 报告。

    Examples
    --------
    >>> rep = HTMLReport("诊断报告", subtitle="示例")   # doctest: +SKIP
    >>> rep.kpis([("被试数", 1200, None)])
    >>> rep.save("out.html")                            # doctest: +SKIP
    """

    def __init__(self, title: str, subtitle: Optional[str] = None,
                 author: Optional[str] = None) -> None:
        self.title = str(title)
        self.subtitle = subtitle
        self.author = author
        self._sections: List[_Section] = []
        self._pending: List[str] = []      # 第一个 heading 之前的块
        self._n_head = 0

    # ---- 结构 ----
    def heading(self, title: str, anchor: Optional[str] = None) -> str:
        """新开一节，返回锚点。"""
        self._n_head += 1
        a = anchor or f"s{self._n_head}"
        self._sections.append(_Section(anchor=a, title=str(title)))
        return a

    def _target(self) -> List[str]:
        if not self._sections:
            self.heading("概览")
        return self._sections[-1].blocks

    # ---- 内容块 ----
    def paragraph(self, text: str) -> "HTMLReport":
        self._target().append(f'<p class="note">{_html.escape(str(text))}</p>')
        return self

    def raw(self, markup: str) -> "HTMLReport":
        """直接插入一段 HTML（调用方负责安全）。"""
        self._target().append(str(markup))
        return self

    def kpis(self, items: Iterable[Sequence[Any]]) -> "HTMLReport":
        """指标卡。每项 ``(标签, 值[, 提示[, 色调]])``，色调 ∈ accent/ok/warn/bad。"""
        cells = []
        for it in items:
            label, value = it[0], it[1]
            hint = it[2] if len(it) > 2 else None
            tone = it[3] if len(it) > 3 else ""
            tone = tone if tone in ("accent", "ok", "warn", "bad") else ""
            hint_html = (f'<div class="h">{_html.escape(str(hint))}</div>'
                         if hint else "")
            cells.append(
                f'<div class="kpi {tone}"><div class="l">{_html.escape(str(label))}</div>'
                f'<div class="v">{_html.escape(str(value))}</div>{hint_html}</div>'
            )
        self._target().append('<div class="kpis">' + "".join(cells) + "</div>")
        return self

    def callout(self, text: str, level: str = "info") -> "HTMLReport":
        lv = level if level in ("info", "ok", "warn", "bad") else "info"
        cls = "" if lv == "info" else lv
        self._target().append(
            f'<div class="callout {cls}">{_html.escape(str(text))}</div>'
        )
        return self

    def table(self, data, columns: Optional[Sequence[str]] = None,
              max_rows: int = 40, highlight: Optional[Dict[str, Any]] = None
              ) -> "HTMLReport":
        """渲染表格。

        ``data`` 可以是 DataFrame、list[dict] 或 ``(headers, rows)``。
        ``highlight`` 为 ``{列名: 判定函数}``，函数返回 ``"bad"``/``"ok"``/``None``
        决定单元格着色——用来一眼看出哪一行有问题。
        """
        headers, rows = _as_table(data, columns)
        total = len(rows)
        cut = rows[:max_rows]

        head = "".join(f"<th>{_html.escape(str(h))}</th>" for h in headers)
        body = []
        for r in cut:
            tds = []
            for c, v in enumerate(r):
                cls = ""
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cls = "num"
                if highlight:
                    name = headers[c] if c < len(headers) else None
                    fn = highlight.get(name) if name else None
                    if fn is not None:
                        try:
                            mark = fn(v)
                        except Exception:
                            mark = None
                        if mark in ("bad", "ok"):
                            cls += f" {mark}"
                tds.append(f'<td class="{cls.strip()}">{_html.escape(_fmt(v))}</td>')
            body.append("<tr>" + "".join(tds) + "</tr>")

        markup = ('<div class="tblwrap"><table class="tbl"><thead><tr>'
                  + head + "</tr></thead><tbody>" + "".join(body)
                  + "</tbody></table></div>")
        self._target().append(markup)
        if total > max_rows:
            self._target().append(
                f'<p class="note">仅显示前 {max_rows} 行，共 {total} 行；'
                "完整数据请见随附的 CSV。</p>"
            )
        return self

    def figure(self, fig, caption: Optional[str] = None, dpi: int = 150
               ) -> "HTMLReport":
        """插入 matplotlib Figure（转成 base64 PNG 内联）。"""
        b64 = _fig_to_base64(fig, dpi=dpi)
        cap = f"<figcaption>{_html.escape(str(caption))}</figcaption>" if caption else ""
        self._target().append(
            f'<figure class="fig"><img alt="{_html.escape(str(caption or "图"))}"'
            f' src="data:image/png;base64,{b64}">{cap}</figure>'
        )
        return self

    def image(self, path_or_bytes, caption: Optional[str] = None,
              mime: str = "png") -> "HTMLReport":
        """插入已有图片文件（同样内联，不产生外部依赖）。"""
        if isinstance(path_or_bytes, (bytes, bytearray)):
            raw = bytes(path_or_bytes)
        else:
            with open(path_or_bytes, "rb") as fh:
                raw = fh.read()
            mime = str(path_or_bytes).rsplit(".", 1)[-1].lower()
            mime = {"jpg": "jpeg"}.get(mime, mime)
        b64 = base64.b64encode(raw).decode("ascii")
        cap = f"<figcaption>{_html.escape(str(caption))}</figcaption>" if caption else ""
        self._target().append(
            f'<figure class="fig"><img alt="图" src="data:image/{mime};base64,{b64}">'
            f"{cap}</figure>"
        )
        return self

    # ---- 输出 ----
    def render(self) -> str:
        toc = "".join(
            f'<a href="#{s.anchor}">{_html.escape(s.title)}</a>'
            for s in self._sections
        )
        body = []
        for i, s in enumerate(self._sections, start=1):
            inner = "\n".join(s.blocks)
            body.append(
                f'<section class="card" id="{s.anchor}">'
                f'<h2><span class="n">{i}</span>{_html.escape(s.title)}</h2>'
                f"{inner}</section>"
            )
        sub = (f'<p class="sub">{_html.escape(str(self.subtitle))}</p>'
               if self.subtitle else "")
        meta = (f'<div class="meta">{_html.escape(str(self.author))}</div>'
                if self.author else "")
        return (
            '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{_html.escape(self.title)}</title>\n"
            f"<style>{_CSS}</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
            f'<header class="hero"><h1>{_html.escape(self.title)}</h1>{sub}{meta}'
            "</header>\n"
            f'<div class="layout"><nav class="toc"><div class="t">目录</div>{toc}'
            "</nav><main>\n" + "\n".join(body) + "\n</main></div>\n"
            '<footer>由 cdcat 生成 · 单文件自包含报告</footer>\n'
            "</div>\n</body>\n</html>\n"
        )

    def save(self, path: str, encoding: str = "utf-8") -> str:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding=encoding) as fh:
            fh.write(self.render())
        return path

    @property
    def n_sections(self) -> int:
        return len(self._sections)


# ---------------------------------------------------------------------- #
# 内部工具
# ---------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:                                   # NaN
            return "—"
        if v == int(v) and abs(v) < 1e6:
            return f"{int(v)}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def _as_table(data, columns: Optional[Sequence[str]] = None
              ) -> Tuple[List[str], List[List[Any]]]:
    """把 DataFrame / list[dict] / (headers, rows) 统一成 (表头, 行)。"""
    if isinstance(data, tuple) and len(data) == 2:
        heads = [str(h) for h in data[0]]
        rows = [[v for v in r] for r in data[1]]
        return heads, rows

    try:                                             # pandas DataFrame
        if hasattr(data, "columns") and hasattr(data, "itertuples"):
            cols = list(columns) if columns else [str(c) for c in data.columns]
            sub = data[cols] if set(cols) <= {str(c) for c in data.columns} else data
            heads = [str(c) for c in sub.columns]
            rows = [[v for v in r] for r in sub.itertuples(index=False, name=None)]
            return heads, rows
    except Exception:
        pass

    rows_in = list(data)
    if not rows_in:
        return list(columns or []), []
    if isinstance(rows_in[0], dict):
        heads = list(columns) if columns else list(rows_in[0].keys())
        return [str(h) for h in heads], [[r.get(h) for h in heads] for r in rows_in]
    heads = list(columns) if columns else [f"列{i + 1}" for i in range(len(rows_in[0]))]
    return [str(h) for h in heads], [[v for v in r] for r in rows_in]


def _fig_to_base64(fig, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


#: 拟合指标表的中文列名（英文列名直接摆进中文报告很别扭，也容易被截断）
_FIT_COLS = {
    "model": "模型",
    "log_likelihood": "对数似然",
    "n_params": "参数数",
    "n_observations": "观测数",
    "aic": "AIC",
    "bic": "BIC",
    "aicc": "AICc",
    "deviance": "偏差 -2LL",
    "saturated_ll": "饱和对数似然",
    "saturated_deviance": "饱和 -2LL",
    "cfi": "CFI",
    "tli": "TLI",
    "rmsea": "RMSEA",
}

#: 报告表格里实际展示的列（去掉常数项与过于技术性的列）
_FIT_SHOW = ["model", "log_likelihood", "n_params", "aic", "bic", "aicc",
             "cfi", "tli", "rmsea"]


def _localize(df, mapping: Dict[str, str], keep: Sequence[str]):
    """按给定顺序挑列并改成中文列名。"""
    cols = [c for c in keep if c in df.columns]
    if not cols:
        return df
    out = df[cols].copy()
    out.columns = [mapping.get(c, c) for c in cols]
    return out


# ---------------------------------------------------------------------- #
# 高层：CDM 分析报告
# ---------------------------------------------------------------------- #
def build_cdm_report(
    analysis,
    responses=None,
    *,
    title: str = "认知诊断分析报告",
    subtitle: Optional[str] = None,
    author: Optional[str] = None,
    include_figures: bool = True,
    max_item_rows: int = 30,
) -> HTMLReport:
    """把 :class:`~cdcat.pipeline.CDMAnalysis` 组装成一份完整报告。

    Parameters
    ----------
    analysis : CDMAnalysis
    responses : ndarray, optional
        作答矩阵，用于算样本量与作答率；不给则从诊断结果推断。
    title, subtitle, author : str, optional
    include_figures : bool
        是否嵌入 matplotlib 图（需要 matplotlib）。
    max_item_rows : int
        题目参数表最多显示多少行。

    Returns
    -------
    HTMLReport
    """
    rep = HTMLReport(title, subtitle=subtitle, author=author)
    q = analysis.q_matrix
    best = analysis.best_model
    diag = analysis.diagnosis.get(best) if analysis.diagnosis else None

    # ---- 1. 概览 ----
    rep.heading("概览")
    n_exam = int(responses.shape[0]) if responses is not None else (
        diag.n_examinees if diag is not None else 0
    )
    kpis = [
        ("题库规模", f"{q.n_items} × {q.n_attrs}", "题 × 属性", "accent"),
        ("合法模式数", f"{q.n_patterns}", "2^K", ""),
        ("被试数", f"{n_exam:,}" if n_exam else "—", None, ""),
        ("最优模型", best or "—", "按 BIC 择优", "accent"),
    ]
    if diag is not None:
        kpis.append(("平均分类信度", f"{diag.mean_confidence():.3f}",
                     "MAP 后验概率均值", "ok" if diag.mean_confidence() > 0.8 else "warn"))
    rep.kpis(kpis)

    if analysis.comparison is not None:
        try:
            rep.callout(analysis.comparison.verdict(), "ok")
        except Exception:
            pass
    if responses is not None:
        import numpy as np

        arr = np.asarray(responses, dtype=float)
        filled = float(np.isfinite(arr).sum()) / max(arr.size, 1)
        if filled < 0.999:
            rep.callout(
                f"作答矩阵存在缺失，实际作答率 {filled:.1%}；"
                "诊断与标定已按缺失值处理，但仍建议核对数据完整性。", "warn"
            )

    # ---- 2. 模型比较 ----
    if analysis.comparison is not None:
        rep.heading("模型比较")
        try:
            df = _localize(analysis.comparison.to_dataframe(), _FIT_COLS, _FIT_SHOW)
            rep.table(df, highlight={"RMSEA": lambda v: (
                "bad" if isinstance(v, (int, float)) and v > 0.08 else None)})
            last = [ln for ln in analysis.comparison.summary().strip().splitlines() if ln.strip()]
            rep.callout(last[-1].strip() if last else "见上表。", "ok")
        except Exception:
            rep.paragraph("模型比较结果不可用。")

    # ---- 3. 题目参数 ----
    if best and best in analysis.models:
        rep.heading("题目参数（最优模型）")
        res = analysis.models[best]
        try:
            df = res.to_dataframe() if hasattr(res, "to_dataframe") else None
        except Exception:
            df = None
        if df is None:
            import pandas as pd

            rows = []
            slip = getattr(res, "slip", None)
            guess = getattr(res, "guess", None)
            for j in range(q.n_items):
                rows.append({
                    "题目": f"I{j + 1}",
                    "slip": float(slip[j]) if slip is not None else None,
                    "guess": float(guess[j]) if guess is not None else None,
                    "区分度": (float(1 - slip[j] - guess[j])
                               if slip is not None and guess is not None else None),
                })
            df = pd.DataFrame(rows)
        rep.table(
            df, max_rows=max_item_rows,
            highlight={"区分度": lambda v: ("bad" if isinstance(v, (int, float))
                                            and v < 0.3 else None)},
        )
        rep.callout(
            "区分度 = 1 − slip − guess。低于 0.3 的题目对知识状态的区分力很弱，"
            "建议复核命题或考虑替换。", "info"
        )

    # ---- 4. 拟合指标 ----
    if analysis.fits:
        rep.heading("模型拟合指标")
        import pandas as pd

        rows = []
        for name, fit in analysis.fits.items():
            d = fit.to_dict() if hasattr(fit, "to_dict") else dict(fit)
            d = {"model": name, **{k: v for k, v in d.items() if k != "model"}}
            rows.append(d)
        rep.table(_localize(pd.DataFrame(rows), _FIT_COLS, _FIT_SHOW))
        rep.callout(
            "CFI / TLI 以独立模型为基线，RMSEA 越小越好。注意：这里的饱和模型按"
            "**反应向量**自由估计，在 2^J 远大于样本量时会严重过拟合，"
            "因此 CDM 的 CFI/TLI 系统性偏低，不宜套用 SEM 里 0.90 的门槛；"
            "RMSEA 与绝对拟合（题目级 χ²、Q3）更值得参考。", "warn"
        )

    # ---- 5. 被试诊断 ----
    if diag is not None:
        rep.heading("被试诊断")
        names = list(getattr(diag, "attr_names", None) or
                     [f"A{k + 1}" for k in range(q.n_attrs)])
        import pandas as pd

        rate = diag.mastery_proportion()
        rel = (diag.attr_reliability.mean(axis=0)
               if getattr(diag, "attr_reliability", None) is not None else None)
        rows = []
        for k, name in enumerate(names):
            rows.append({
                "属性": name,
                "掌握比例": float(rate[k]),
                "平均分类信度": float(rel[k]) if rel is not None else None,
            })
        rep.table(pd.DataFrame(rows),
                  highlight={"平均分类信度": lambda v: ("bad" if isinstance(v, float)
                                                        and v < 0.6 else None)})

        dist = diag.pattern_distribution()
        top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:8]
        if top:
            import numpy as np

            rows = []
            for idx, cnt in top:
                bits = (int(idx) >> np.arange(q.n_attrs - 1, -1, -1)) & 1
                rows.append({
                    "模式": "".join(str(b) for b in bits),
                    "人数": cnt,
                    "占比": cnt / max(diag.n_examinees, 1),
                })
            rep.table(pd.DataFrame(rows))
            rep.callout(
                f"共出现 {len(dist)} 种知识状态；上表为人数最多的 8 种。", "info"
            )

        if include_figures:
            try:
                from .calibration_plots import plot_mastery_proportion

                fig = plot_mastery_proportion(diag.mastery_proportion(), names)
                rep.figure(fig, "各属性掌握比例")
            except Exception:
                pass

    # ---- 6. 综合图 ----
    if include_figures:
        try:
            from .calibration_plots import calibration_report_figure

            fig = calibration_report_figure(analysis)
            rep.heading("综合图")
            rep.figure(fig, "收敛曲线、题目区分度、属性掌握与模式分布")
        except Exception:
            pass

    # ---- 7. 绝对拟合 ----
    if getattr(analysis, "absolute_fit", None) is not None:
        rep.heading("绝对拟合与局部依赖")
        af = analysis.absolute_fit
        try:
            rep.table(af.item_fit.to_dataframe(), max_rows=25,
                      highlight={"p值": lambda v: (
                          "bad" if isinstance(v, (int, float)) and v < 0.05 else None)})
            rep.callout(
                "p 值小于 .05 表示该题的观测答对率曲线与模型预测有系统偏差。"
                "样本量大时该检验很敏感，需结合最大残差一起看。", "info"
            )
        except Exception:
            rep.paragraph("题目级拟合明细不可用。")
        try:
            ld = af.local_dependence
            pairs = ld.flagged_pairs(top_n=10)
            if len(pairs) > 0:
                rep.paragraph(f"Q3 超过阈值 {ld.threshold:g} 的题目对（最多列 10 对）：")
                rep.table(pairs)
                rep.callout(
                    "较大的 Q3 通常意味着两题共享了 Q 矩阵没写出的属性，"
                    "或题干互相提示；建议复核这两题的 Q 行。", "warn"
                )
            else:
                rep.callout("未发现明显的局部依赖（题目对 Q3 均低于阈值）。", "ok")
        except Exception:
            pass

    rep.heading("口径说明")
    rep.paragraph(
        "本报告由 cdcat 自动生成，图表以内联 base64 方式嵌入，"
        "不依赖任何外部样式表或脚本，可离线打开与归档。"
    )
    rep.paragraph(
        "模型择优采用 BIC；被试知识状态以 MAP 为主、EAP 边际概率为辅；"
        "分类信度为 MAP 模式的后验概率，越接近 1 越可靠。"
    )
    return rep
