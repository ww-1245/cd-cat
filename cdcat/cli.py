"""``cdcat`` 命令行入口。

把库里的标定、仿真、诊断、报告能力暴露成一组子命令，便于批处理、
服务器无界面环境、以及写进 Makefile / CI 流水线。

用法
----
::

    # 用内置题库快速试跑
    cdcat demo

    # 标定：Q 矩阵 + 作答数据 -> 题目参数 + 拟合指标
    cdcat calibrate -q q.csv -r resp.csv -m DINA -o out/

    # 蒙特卡洛仿真：输出判准率、用题量、曝光均匀性
    cdcat simulate -q q.csv -m DINA -s PWKL -n 300 -R 5 -o out/

    # 绝对拟合 / 局部依赖诊断
    cdcat diagnose -q q.csv -r resp.csv -o out/

    # 后验预测检验：模型能否再现数据的二阶结构
    cdcat ppmc -q q.csv -r resp.csv -o out/ --replicates 500

    # 属性层级结构比较（BIC 择优）
    cdcat hier-test -q q.csv -r resp.csv -o out/

    # 题库装配：按覆盖与题量约束挑一份短测验
    cdcat assemble -q q.csv --params params.json -k 16 -o out/

    # 新题在线标定（借用已有模型的后验）
    cdcat online-calib -q q.csv -r resp.csv --new-items 41 42 43 -o out/

    # DIF 检验（分组公平性）
    cdcat dif -q q.csv -r resp.csv --groups groups.csv -o out/

    # 非参数分类：不标定参数，直接用距离判别
    cdcat nonparam -q q.csv -r resp.csv --methods gdd hamming -o out/

    # Q 矩阵自动修正建议
    cdcat validate-q -q q.csv -r resp.csv -o out/

    # 群体（班级）诊断报告
    cdcat report --bank bank.json -r resp.csv -o out/

    # 单文件 HTML 报告（图表内联，可离线归档）
    cdcat report-html -q q.csv -r resp.csv -o out/

    # 查看可用的内置题库与准则
    cdcat presets
    cdcat list

所有子命令都支持 ``--json`` 输出机器可读结果，方便被其它脚本消费。
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from .io import (
    read_bank,
    read_delimited,
    read_params,
    read_q_matrix,
    read_responses,
    write_json,
    write_params,
    write_table,
)
from .presets import PRESET_BANKS, build_preset_bank, list_presets

# 准则清单直接从注册表取，新增准则时命令行自动跟上，不必二次维护
from .selection import SELECTION_REGISTRY as _SELECTORS

__all__ = ["main", "build_parser"]


# ====================================================================== #
# 输出小工具
# ====================================================================== #
def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _rule(title: str = "") -> None:
    line = "=" * 72
    if title:
        _say(line)
        _say(f"  {title}")
        _say(line)
    else:
        _say(line)


def _display_width(text: str) -> int:
    """终端显示宽度：东亚全角字符占 2 列，其余占 1 列。

    直接用 ``len()`` 会让中文表头与数字列错位，因为等宽字体里中文占两格。
    """
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _pad(text: str, width: int) -> str:
    """按显示宽度右侧补空格。"""
    return str(text) + " " * max(width - _display_width(text), 0)


def _kv(pairs: Sequence[tuple], indent: int = 2) -> None:
    width = max((_display_width(k) for k, _ in pairs), default=0)
    for k, v in pairs:
        _say(f"{' ' * indent}{_pad(k, width)} : {v}")


def _table(rows: Sequence[Sequence[Any]], header: Optional[Sequence[str]] = None) -> None:
    """在终端打印等宽表格（按显示宽度对齐，兼容中日韩字符）。"""
    if header is not None:
        rows = [list(header)] + [list(r) for r in rows]
    if not rows:
        _say("(空)")
        return
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [max(_display_width(r[i]) for r in str_rows) for i in range(len(str_rows[0]))]
    for idx, r in enumerate(str_rows):
        _say("  " + "  ".join(_pad(c, widths[i]) for i, c in enumerate(r)))
        if idx == 0 and header is not None:
            _say("  " + "  ".join("-" * w for w in widths))


def _save_fig(fig, path: str) -> str:
    """保存 matplotlib 图（自动建目录）。"""
    import warnings

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with warnings.catch_warnings():
        # 含极坐标子图时 tight_layout 会告警，但 bbox_inches='tight' 已能正确裁边
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - 布局失败不应阻断出图
            pass
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ====================================================================== #
# 输入解析
# ====================================================================== #
def _load_q(args) -> tuple:
    """返回 ``(QMatrix, 来源描述)``。"""
    if args.preset:
        bank = build_preset_bank(args.preset)
        return bank["q"], f"内置题库「{args.preset}」"
    if args.q:
        return read_q_matrix(args.q), os.path.basename(args.q)
    raise SystemExit("错误：请用 -q 指定 Q 矩阵文件，或用 --preset 指定内置题库")


def _load_responses(args, n_items: int) -> np.ndarray:
    if not args.responses:
        raise SystemExit("错误：该子命令需要 -r 指定作答矩阵文件")
    resp, ids = read_responses(args.responses, n_items=n_items)
    if ids is None:
        ids = [f"E{i + 1}" for i in range(resp.shape[0])]
    args._examinee_ids = ids
    return resp


def _load_groups(args, examinee_ids: Sequence[str]) -> Optional[List[str]]:
    """读取分组表并**映射到作答矩阵的行顺序**。

    支持两种格式：

    1. 两列：``被试ID, 组名`` —— 按 ID 精确匹配（推荐，与行序无关）
    2. 单列：``组名`` —— 顺序须与作答矩阵行顺序一致
    """
    path = getattr(args, "groups", None)
    if not path:
        return None
    rows, _header = read_delimited(path)
    if not rows:
        raise SystemExit(f"错误：分组表为空: {path}")

    if len(rows[0]) >= 2:
        mapping = {str(r[0]).strip(): str(r[1]).strip() for r in rows if len(r) >= 2}
        missing = [i for i in examinee_ids if str(i) not in mapping]
        if missing:
            shown = ", ".join(str(m) for m in missing[:3])
            raise SystemExit(
                f"错误：分组表缺少 {len(missing)} 名被试的组别（例如 {shown}）"
            )
        return [mapping[str(i)] for i in examinee_ids]

    labels = [str(r[0]).strip() for r in rows]
    if len(labels) != len(examinee_ids):
        raise SystemExit(
            f"错误：分组表有 {len(labels)} 行，作答矩阵有 {len(examinee_ids)} 行，无法一一对应"
        )
    return labels


def _load_bank(args) -> Dict[str, Any]:
    """题库来源：--bank 优先，其次 --preset。"""
    if args.bank:
        return read_bank(args.bank)
    if args.preset:
        return build_preset_bank(args.preset)
    raise SystemExit("错误：请用 --bank 指定题库 JSON，或用 --preset 指定内置题库")


def _load_params(args, q):
    """题目参数来源：--params 文件 > --bank 文件 > --preset。"""
    from .models import create_model

    if getattr(args, "params", None):
        name, params = read_params(args.params)
        return name, create_model(name, q, **params)
    if getattr(args, "bank", None):
        bank = read_bank(args.bank)
        return args.model, create_model(args.model, q, slip=bank["slip"], guess=bank["guess"])
    if getattr(args, "preset", None):
        bank = build_preset_bank(args.preset)
        return args.model, create_model(args.model, q, slip=bank["slip"], guess=bank["guess"])
    raise SystemExit("错误：无法确定题目参数，请用 --params / --bank / --preset 之一")


def _outdir(args) -> str:
    out = getattr(args, "out", None) or "cdcat_out"
    os.makedirs(out, exist_ok=True)
    return out


def _pair(x: float, y: float) -> str:
    return f"({x:.3f}, {y:.3f})"


# ====================================================================== #
# 子命令：presets / list
# ====================================================================== #
def cmd_presets(args) -> int:
    _rule("内置题库")
    rows = []
    for name in list_presets():
        spec = PRESET_BANKS[name]
        rows.append([name, spec["n_items"], spec["n_attrs"], spec["description"]])
    _table(rows, header=["名称", "题目数", "属性数", "说明"])
    _say()
    _say("用法示例：cdcat simulate --preset \"分数运算 (4 属性)\" -o out/")
    return 0


def _canonical_names(registry: Dict[str, Any]) -> List[str]:
    """按实现类去重，返回可读的组件名。

    注册表里每个组件登记了多个别名（如 ``"pattern"`` 与 ``"patternconfidence"``），
    直接列键名会重复且难读，因此按类去重后取类名，并对少数类名做美化。
    """
    display = {"GDINA": "G-DINA", "MPWKL": "MPWKL"}
    seen: set = set()
    out: List[str] = []
    for value in registry.values():
        cls = value if isinstance(value, type) else type(value)
        if cls in seen:
            continue
        seen.add(cls)
        out.append(display.get(cls.__name__, cls.__name__))
    return out


def cmd_list(args) -> int:
    from .models import _MODEL_REGISTRY
    from .selection import SELECTION_REGISTRY
    from .selection.exposure import _REGISTRY as _EXPOSURE_REGISTRY
    from .termination import _RULE_REGISTRY

    _rule("可用组件")
    sections = [
        ("响应模型", _canonical_names(_MODEL_REGISTRY)),
        ("选题准则", list(SELECTION_REGISTRY)),
        ("终止规则", _canonical_names(_RULE_REGISTRY)),
        ("曝光控制", _canonical_names(_EXPOSURE_REGISTRY)),
    ]
    for title, names in sections:
        _say(f"\n{title}（{len(names)}）:")
        _table([[n] for n in sorted(names)])
    return 0


# ====================================================================== #
# 子命令：demo
# ====================================================================== #
def cmd_demo(args) -> int:
    """一键跑通：内置题库 -> 仿真 -> 诊断 -> 群体报告。"""
    bank = build_preset_bank(args.preset)
    out = _outdir(args)
    q = bank["q"]
    _rule(f"CD-CAT 快速演示  ·  {args.preset}")
    _kv([
        ("题目数 J", q.n_items),
        ("属性数 K", q.n_attrs),
        ("属性模式数 2^K", q.n_patterns),
        ("输出目录", out),
    ])

    # ---- 1) 仿真 ----
    from .simulation import CDCATSimulator

    _say("\n[1/3] 蒙特卡洛仿真 ……")
    sim = CDCATSimulator(
        q,
        model=args.model,
        model_params={"slip": bank["slip"], "guess": bank["guess"]},
        selector=args.selector,
        estimator="MAP",
        min_items=5,
        max_items=args.max_items,
        true_model=args.model,
        true_model_params={"slip": bank["slip"], "guess": bank["guess"]},
    )
    result = sim.run(n_examinees=args.n_examinees, n_replications=args.n_rep, seed=args.seed)
    _kv([
        ("模式判准率 PMR", f"{result.pmr():.4f}"),
        ("属性判准率 AMR", f"{result.amr():.4f}"),
        ("平均用题量", f"{result.avg_items():.2f}"),
        ("题库利用率", f"{result.test_utilization():.4f}"),
        ("曝光率标准差", f"{result.exposure_uniformity():.4f}"),
    ])
    csv_path = write_table(
        [["selector", "PMR", "AMR", "avg_items", "test_utilization", "exposure_sd"],
         [args.selector, f"{result.pmr():.4f}", f"{result.amr():.4f}",
          f"{result.avg_items():.2f}", f"{result.test_utilization():.4f}",
          f"{result.exposure_uniformity():.4f}"]],
        os.path.join(out, "simulation_summary.csv"),
    )

    # ---- 2) 群体报告 ----
    _say("\n[2/3] 群体诊断报告 ……")
    from .evaluation.group import build_group_report

    records = list(result.records)
    group = build_group_report(
        records,
        attr_names=bank["attrs"],
        q_matrix=q,
        item_names=q.item_names,
        label=f"{args.preset} · {args.model}",
        include_truth=True,
    )
    write_json(group.to_dict(), os.path.join(out, "group_report.json"))
    with open(os.path.join(out, "group_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(group.to_text())
    _kv([
        ("被试数", group.n_examinees),
        ("最弱属性", group.attr_names[group.weakest_attrs(1)[0]]),
        ("平均掌握属性数", f"{group.mean_mastered:.2f} / {q.n_attrs}"),
    ])
    if args.figures:
        try:
            from .figures import plot_group_report

            fig = plot_group_report(group)
            _save_fig(fig, os.path.join(out, "group_report.png"))
            _say(f"       -> {os.path.join(out, 'group_report.png')}")
        except ImportError:
            _say("       (未安装 matplotlib，跳过绘图)")

    # ---- 3) 小结 ----
    _say("\n[3/3] 完成")
    _say(f"       -> {os.path.join(out, 'simulation_summary.csv')}")
    _say(f"       -> {os.path.join(out, 'group_report.txt')}")
    return 0


# ====================================================================== #
# 子命令：calibrate
# ====================================================================== #
def cmd_calibrate(args) -> int:
    from .calibration import calibrate

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)

    _rule("参数标定（MMLE / EM）")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("模型", args.model),
        ("估计属性先验 π(α)", "是" if args.estimate_prior else "否"),
    ])

    model, res = calibrate(
        q, resp, model_type=args.model,
        max_iter=args.max_iter, tol=args.tol,
        estimate_prior=args.estimate_prior,
        verbose=args.verbose,
    )

    _say("\n拟合指标")
    _kv([
        ("迭代次数", res.n_iterations),
        ("是否收敛", "是" if res.converged else "否"),
        ("-2LL", f"{res.log_likelihood * -2:.2f}"),
        ("自由参数数", res.n_params),
        ("AIC", f"{res.aic:.2f}"),
        ("BIC", f"{res.bic:.2f}"),
        ("AICc", f"{res.aicc:.2f}"),
    ])

    # 题目参数表
    rows = []
    names = q.item_names or [f"I{i + 1}" for i in range(q.n_items)]
    for i, name in enumerate(names):
        s = float(res.slip[i]) if res.slip is not None else float("nan")
        g = float(res.guess[i]) if res.guess is not None else float("nan")
        rows.append([name, f"{s:.4f}", f"{g:.4f}", f"{1 - s - g:.4f}"])
    _say("\n题目参数（前 10 题）")
    _table(rows[:10], header=["题目", "slip", "guess", "区分度 1-s-g"])

    # 落盘
    write_params(args.model, model, os.path.join(out, "item_params.json"))
    write_table([["item", "slip", "guess", "discrimination"]] + rows,
                os.path.join(out, "item_params.csv"))
    write_json(
        {
            "model": args.model,
            "n_items": q.n_items,
            "n_attrs": q.n_attrs,
            "n_examinees": int(resp.shape[0]),
            "n_iterations": res.n_iterations,
            "converged": bool(res.converged),
            "minus2LL": float(res.log_likelihood * -2),
            "n_params": int(res.n_params),
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "AICc": float(res.aicc),
            "attr_prior": None if res.attr_prior is None else np.asarray(res.attr_prior).tolist(),
            "ll_history": [float(v) for v in res.log_likelihood_history],
        },
        os.path.join(out, "fit_indices.json"),
    )
    # 导出成完整题库，便于后续 simulate / report 复用
    write_json(
        {
            "name": args.model + " 标定题库",
            "description": f"由 {os.path.basename(args.responses)} 标定",
            "attrs": list(q.attr_names),
            "q_matrix": np.asarray(q.matrix).tolist(),
            "item_names": q.item_names,
            "slip": np.asarray(res.slip).tolist(),
            "guess": np.asarray(res.guess).tolist(),
        },
        os.path.join(out, "calibrated_bank.json"),
    )

    if args.figures:
        try:
            from .calibration_plots import plot_item_discrimination, plot_ll_convergence

            fig = plot_ll_convergence({args.model: res})
            _save_fig(fig, os.path.join(out, "ll_convergence.png"))
            fig = plot_item_discrimination(1.0 - res.slip - res.guess, q.item_names)
            _save_fig(fig, os.path.join(out, "item_discrimination.png"))
            _say(f"  -> {os.path.join(out, 'll_convergence.png')}")
            _say(f"  -> {os.path.join(out, 'item_discrimination.png')}")
        except ImportError:
            _say("(未安装 matplotlib，跳过绘图)")

    _say("\n输出文件")
    for f in ("item_params.csv", "item_params.json", "fit_indices.json", "calibrated_bank.json"):
        _say(f"  -> {os.path.join(out, f)}")
    if args.json:
        _say("\n（结构化结果已写入 fit_indices.json）")
    return 0


# ====================================================================== #
# 子命令：simulate
# ====================================================================== #
def cmd_simulate(args) -> int:
    from .simulation import CDCATSimulator

    out = _outdir(args)
    compare = args.compare or []

    # 题库与真实模型参数
    if args.bank or args.preset:
        bank = _load_bank(args)
        q = bank["q"]
        true_name = args.model
        true_params = {"slip": bank["slip"], "guess": bank["guess"]}
        src = os.path.basename(args.bank) if args.bank else f"内置题库「{args.preset}」"
    else:
        q, src = _load_q(args)
        true_name, true_params = args.model, None

    _rule("蒙特卡洛仿真")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("真实模型", true_name),
        ("估计模型", args.model),
        ("被试数 × 重复", f"{args.n_examinees} × {args.n_rep}"),
        ("最大题量", args.max_items),
        ("并行进程", args.n_jobs),
        ("随机种子", args.seed),
    ])

    def _make(selector: str) -> CDCATSimulator:
        return CDCATSimulator(
            q,
            model=args.model,
            model_params=true_params,
            selector=selector,
            estimator=args.estimator,
            min_items=args.min_items,
            max_items=args.max_items,
            true_model=true_name,
            true_model_params=true_params,
            exposure_control=args.exposure,
        )

    selectors = compare if compare else [args.selector]
    results = {}
    if len(selectors) > 1:
        first = _make(selectors[0])
        results = first.compare_criteria(
            selectors,
            n_examinees=args.n_examinees,
            n_replications=args.n_rep,
            seed=args.seed,
        )
    else:
        sim = _make(selectors[0])
        results[selectors[0]] = sim.run(
            n_examinees=args.n_examinees,
            n_replications=args.n_rep,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

    _say("\n结果")
    rows = []
    payload: List[dict] = []
    for name, res in results.items():
        rows.append([
            name, f"{res.pmr():.4f}", f"{res.amr():.4f}", f"{res.avg_items():.2f}",
            f"{res.test_utilization():.4f}", f"{res.exposure_uniformity():.4f}",
        ])
        payload.append({
            "selector": name,
            "PMR": res.pmr(),
            "AMR": res.amr(),
            "avg_items": res.avg_items(),
            "test_utilization": res.test_utilization(),
            "exposure_sd": res.exposure_uniformity(),
            "n_examinees": res.n_examinees,
        })
    _table(rows, header=["准则", "PMR", "AMR", "平均题量", "题库利用率", "曝光率SD"])

    # 结构上限
    first_res = next(iter(results.values()))
    ceiling = first_res.pmr_vs_ceiling()
    _say("\n结构上限对比")
    _kv([
        ("实际 PMR", f"{ceiling['pmr']:.4f}"),
        ("理论上限", f"{ceiling['ceiling']:.4f}"),
        ("达成效率", f"{ceiling['efficiency']:.4f}"),
        ("不可辨识模式数", ceiling["n_unidentifiable"]),
    ])
    _say(f"  {ceiling['verdict']}")

    write_json(payload, os.path.join(out, "simulation.json"))
    if args.figures:
        try:
            from .figures import plot_convergence

            for name, res in results.items():
                fig = plot_convergence(res)
                _save_fig(fig, os.path.join(out, f"convergence_{name}.png"))
        except ImportError:
            _say("(未安装 matplotlib，跳过绘图)")
    _say(f"\n  -> {os.path.join(out, 'simulation.json')}")
    return 0


# ====================================================================== #
# 子命令：diagnose
# ====================================================================== #
def cmd_diagnose(args) -> int:
    from .calibration import calibrate
    from .diagnostics import analyse_absolute_fit
    from .evaluation.item_analysis import analyse_items

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)
    names = q.item_names or [f"I{i + 1}" for i in range(q.n_items)]

    _rule("题库诊断：绝对拟合 · 局部依赖 · 题目分析")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("模型", args.model),
    ])

    model, res = calibrate(q, resp, model_type=args.model, max_iter=args.max_iter, tol=args.tol)
    af = analyse_absolute_fit(model, resp, item_names=names)
    ia = analyse_items(model, seed=0)
    item_fit = af.item_fit
    ld = af.local_dependence

    n_bad = int((item_fit.p_value < 0.05).sum())
    _say("\n绝对拟合")
    _kv([
        ("不合格题目数", f"{n_bad} / {q.n_items}"),
        ("最大残差 均值/最大",
         f"{item_fit.max_abs_residual.mean():.4f} / {item_fit.max_abs_residual.max():.4f}"),
        ("Q3 超标对数", f"{ld.n_flagged}（阈值 {ld.threshold}）"),
        ("平均 Q3", f"{ld.mean_q3:.4f}"),
    ])

    rows = []
    for i in range(q.n_items):
        rows.append([
            names[i],
            f"{float(item_fit.chi_square[i]):.3f}",
            f"{float(item_fit.df[i]):.0f}",
            f"{float(item_fit.p_value[i]):.4f}",
            "是" if float(item_fit.p_value[i]) < 0.05 else "",
            f"{float(item_fit.max_abs_residual[i]):.4f}",
            f"{float(ia.discrimination[i]):.4f}",
            f"{float(ia.information[i]):.4f}",
        ])
    order = np.argsort(-np.asarray(item_fit.chi_square))
    rows_sorted = [rows[i] for i in order]
    _say("\n题目级诊断（前 10 题，按 χ² 降序）")
    _table(rows_sorted[:10],
           header=["题目", "χ²", "df", "p 值", "不合格", "最大残差", "区分度", "信息量"])

    if ld.n_flagged:
        _say("\nQ3 超标题目对（前 10）")
        pairs = ld.top_pairs(10)
        _table(pairs.astype(str).values.tolist(), header=list(pairs.columns))

    write_table(
        [["item", "chi2", "df", "p_value", "misfit", "max_residual", "discrimination", "information"]]
        + rows_sorted,
        os.path.join(out, "item_diagnostics.csv"),
    )
    write_json(
        {
            "model": args.model,
            "n_items": q.n_items,
            "n_misfit_p05": n_bad,
            "max_residual_mean": float(item_fit.max_abs_residual.mean()),
            "max_residual_max": float(item_fit.max_abs_residual.max()),
            "q3_flagged_pairs": int(ld.n_flagged),
            "q3_threshold": float(ld.threshold),
            "q3_mean": float(ld.mean_q3),
            "q3_max": float(ld.max_q3),
            "flagged_pairs": [
                {"item_a": names[a], "item_b": names[b], "Q3": float(v)}
                for a, b, v in ld.pairs[:50]
            ],
        },
        os.path.join(out, "diagnostics.json"),
    )
    write_params(args.model, model, os.path.join(out, "item_params.json"))

    if args.figures:
        try:
            from .calibration_plots import plot_item_fit, plot_q3_heatmap

            fig = plot_item_fit(item_fit, top_n=25)
            _save_fig(fig, os.path.join(out, "item_fit.png"))
            fig = plot_q3_heatmap(ld)
            _save_fig(fig, os.path.join(out, "q3_heatmap.png"))
            _say(f"  -> {os.path.join(out, 'item_fit.png')}")
            _say(f"  -> {os.path.join(out, 'q3_heatmap.png')}")
        except ImportError:
            _say("(未安装 matplotlib，跳过绘图)")

    _say(f"\n  -> {os.path.join(out, 'item_diagnostics.csv')}")
    _say(f"  -> {os.path.join(out, 'diagnostics.json')}")
    return 0


# ====================================================================== #
# 子命令：validate-q
# ====================================================================== #
def cmd_validate_q(args) -> int:
    from .calibration import calibrate
    from .qmatrix.refinement import validate_q_matrix

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)

    _rule("Q 矩阵校验与自动修正建议")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("模型", args.model),
        ("判定阈值 PVAF", args.pvaf),
    ])

    model, res = calibrate(q, resp, model_type=args.model, max_iter=args.max_iter, tol=args.tol)
    report = validate_q_matrix(q, resp, model, pvaf_threshold=args.pvaf)

    _kv([
        ("疑似需修正题目数", f"{report.n_suspect} / {q.n_items}"),
        ("建议新增属性数", report.n_additions),
        ("建议删除属性数", report.n_removals),
    ])

    names = q.item_names or [f"I{i + 1}" for i in range(q.n_items)]
    rows = []
    for s in report.suggestions[: args.top]:
        cur = s.current_attrs_str()
        sug = s.suggested_attrs_str()
        rows.append([names[s.item_idx], cur or "（空）", sug or "（空）",
                     f"{s.pvaf_current:.3f}", f"{s.pvaf_suggested:.3f}"])
    if rows:
        _say("\n修正建议（按改善量降序）")
        _table(rows, header=["题目", "原 Q 行", "建议 Q 行", "原 PVAF", "建议 PVAF"])
    else:
        _say("\n未发现需要修正的 Q 行。")

    write_json(report.to_dict(), os.path.join(out, "q_validation.json"))
    write_table([["item", "current", "suggested", "pvaf_current", "pvaf_suggested", "delta"]]
                + [[names[s.item_idx], s.current_attrs_str(), s.suggested_attrs_str(),
                    f"{s.pvaf_current:.4f}", f"{s.pvaf_suggested:.4f}",
                    f"{s.pvaf_suggested - s.pvaf_current:+.4f}"] for s in report.suggestions],
                os.path.join(out, "q_validation.csv"))
    _say(f"\n  -> {os.path.join(out, 'q_validation.csv')}")
    return 0


# ====================================================================== #
# 子命令：report
# ====================================================================== #
def cmd_report(args) -> int:
    from .evaluation.group import build_group_report

    if args.q:
        q, src = read_q_matrix(args.q), os.path.basename(args.q)
    else:
        bank = _load_bank(args)
        q, src = bank["q"], (os.path.basename(args.bank) if args.bank else f"内置题库「{args.preset}」")

    resp = _load_responses(args, q.n_items)
    ids = getattr(args, "_examinee_ids", None)
    groups = _load_groups(args, ids)
    model_name, model = _load_params(args, q)
    out = _outdir(args)

    _rule("群体（班级）诊断报告")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("题目参数", model_name),
        ("缺失作答处理", args.missing),
    ] + ([("分组表", f"{os.path.basename(args.groups)} · {len(set(groups))} 组")]
         if groups is not None else []))

    group = build_group_report(
        None,
        attr_names=q.attr_names,
        q_matrix=q,
        item_names=q.item_names,
        model=model,
        responses=resp,
        examinee_ids=ids,
        group_labels=groups,
        label=args.label or "班级诊断",
        missing=args.missing,
    )

    _say("\n总体")
    _kv([
        ("有效被试数", group.n_examinees),
        ("平均掌握属性数", f"{group.mean_mastered:.2f} / {q.n_attrs}"),
        ("完全掌握人数", f"{group.n_full_mastery} ({group.full_mastery_rate:.1%})"),
    ])
    _say("\n各属性掌握率")
    order = np.argsort(group.attr_mastery_rate)
    rows = [[group.attr_names[i], f"{group.attr_mastery_rate[i]:.3f}",
             f"{group.attr_mean_prob[i]:.3f}"] for i in order]
    _table(rows, header=["属性", "掌握率", "平均掌握概率"])

    weak = group.weakest_attrs(args.top)
    _say("\n薄弱知识点 TOP-%d：%s" % (len(weak), "、".join(group.attr_names[i] for i in weak)))

    if groups is not None:
        gdf = group.group_dataframe()
        if not gdf.empty:
            _say("\n分组对比")
            _table(gdf.astype(str).values.tolist(), header=gdf.columns.tolist())

    write_json(group.to_dict(), os.path.join(out, "group_report.json"))
    with open(os.path.join(out, "group_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(group.to_text())
    df = group.attr_dataframe()
    write_table([df.columns.tolist()] + df.astype(str).values.tolist(),
                os.path.join(out, "attribute_mastery.csv"))
    if groups is not None and not group.group_dataframe().empty:
        gdf = group.group_dataframe()
        write_table([gdf.columns.tolist()] + gdf.astype(str).values.tolist(),
                    os.path.join(out, "group_comparison.csv"))
    if args.figures:
        try:
            from .figures import plot_group_report

            fig = plot_group_report(group)
            _save_fig(fig, os.path.join(out, "group_report.png"))
        except ImportError:
            _say("(未安装 matplotlib，跳过绘图)")
    _say(f"\n  -> {os.path.join(out, 'group_report.txt')}")
    _say(f"  -> {os.path.join(out, 'attribute_mastery.csv')}")
    if groups is not None and not group.group_dataframe().empty:
        _say(f"  -> {os.path.join(out, 'group_comparison.csv')}")
    return 0


# ====================================================================== #
# 子命令：ppmc / hier-test / assemble / online-calib / dif / report-html
# ====================================================================== #
def cmd_ppmc(args) -> int:
    """后验预测检验：用复制数据检验"模型能否再现数据的二阶结构"。"""
    from .calibration import calibrate
    from .diagnostics import posterior_predictive_check

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)

    _rule("后验预测检验（PPMC）")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("模型", args.model),
        ("复制次数", args.replicates),
        ("复制侧重估", "否" if args.no_refit else "是"),
    ])

    model, res = calibrate(q, resp, model_type=args.model,
                           max_iter=args.max_iter, tol=args.tol)
    pp = posterior_predictive_check(
        model, resp,
        prior=res.attr_prior,
        n_replicates=args.replicates,
        alpha=args.alpha,
        refit=not args.no_refit,
        max_items_pair=args.max_items_pair,
    )
    df = pp.to_dataframe()
    _say("")
    _table(
        [[r["统计量"], f"{r['观测值']:.4f}", f"{r['复制均值']:.4f}",
          f"{r['复制 SD']:.4f}", f"{r['PPP']:.4f}", r["判读"]]
         for _, r in df.iterrows()],
        header=["统计量", "观测值", "复制均值", "复制 SD", "PPP", "判读"],
    )
    _say("")
    _say(pp.verdict())
    _say("提示：PPP 接近 0.5 表示复制数据与真实数据同分布（模型可信）；"
         "接近 0 或 1 说明该统计量未被模型再现。")

    write_table([df.columns.tolist()] + df.astype(str).values.tolist(),
                os.path.join(out, "ppmc.csv"))
    if args.json:
        write_json(pp.to_dict(), os.path.join(out, "ppmc.json"))
    _say(f"\n  -> {os.path.join(out, 'ppmc.csv')}")
    return 0


def cmd_hier_test(args) -> int:
    """属性层级比较：用 BIC 判断数据支持哪一种先备结构。"""
    from .diagnostics import compare_hierarchies

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)

    structures = args.structures or ["none", "linear", "convergent", "divergent"]
    _rule("属性层级结构比较（BIC）")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("模型", args.model),
        ("候选结构", "、".join(structures)),
    ])

    try:
        rep = compare_hierarchies(
            q, resp, model_type=args.model, structures=structures,
            max_iter=args.max_iter, tol=args.tol, verbose=args.verbose,
        )
    except ValueError as exc:
        raise SystemExit(f"错误：{exc}") from exc
    df = rep.to_dataframe()
    _say("")
    _table(
        [[r["label"], r["structure"], f"{int(r['n_legal'])}/{int(r['n_params'])}",
          f"{float(r['loglik']):.2f}", f"{float(r['bic']):.2f}",
          f"{float(r['delta_bic']):.2f}", f"{float(r['bic_weight']):.3f}",
          f"{float(r['illegal_ratio']):.2f}"]
         for _, r in df.iterrows()],
        header=["候选", "结构", "合法模式/参数", "logL", "BIC", "ΔBIC", "权重", "非法率"],
    )
    _say("")
    _say(rep.verdict())
    _say("注意：层级约束落在参数空间边界上，χ² 近似偏保守，故以 BIC 为主判据。")

    write_table([df.columns.tolist()] + df.astype(str).values.tolist(),
                os.path.join(out, "hierarchy_comparison.csv"))
    if args.json:
        write_json(rep.to_dict(), os.path.join(out, "hierarchy_comparison.json"))
    _say(f"\n  -> {os.path.join(out, 'hierarchy_comparison.csv')}")
    return 0


def cmd_assemble(args) -> int:
    """题库装配：在覆盖与题量约束下挑一份短测验。"""
    from .selection import assemble_test_bank

    q, src = _load_q(args)
    out = _outdir(args)
    model, model_name = _load_model_for_q(args, q)

    must = None
    if args.must:
        must = [int(x) - 1 for x in args.must]
        bad = [i + 1 for i in must if not (0 <= i < q.n_items)]
        if bad:
            raise SystemExit(f"错误：--must 中的题号超出范围：{bad}")

    _rule("题库装配（短测验选题）")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("题目参数", model_name),
        ("目标题量", args.n_items or "未指定"),
        ("每属性下限", args.min_per_attr),
        ("每属性上限", args.max_per_attr or "不限"),
    ])

    try:
        res = assemble_test_bank(
            q, model,
            n_items=args.n_items,
            must_include=must,
            min_per_attr=args.min_per_attr,
            max_per_attr=args.max_per_attr,
            local_search=not args.no_search,
            max_rounds=args.max_rounds,
        )
    except ValueError as exc:
        raise SystemExit(f"错误：{exc}") from exc

    _say("")
    amr = np.asarray(res.expected_amr, dtype=float).ravel()
    _kv([
        ("已选题数", res.n_items),
        ("期望模式判准率", f"{res.expected_pmr:.4f}"),
        ("期望属性判准率", f"{amr.mean():.4f}（最弱属性 {amr.min():.4f}）"),
        ("属性覆盖", "、".join(f"{n}={c}" for n, c in zip(res.attr_names, res.attr_coverage))),
        ("局部搜索交换次数", res.n_swaps),
    ])
    _say("")
    _say("入选题目：" + "、".join(res.selected_names))

    df = res.to_dataframe()
    write_table([df.columns.tolist()] + df.astype(str).values.tolist(),
                os.path.join(out, "assembly.csv"))
    write_json(res.to_dict(), os.path.join(out, "assembly.json"))
    if args.export_bank:
        with open(os.path.join(out, "assembled_items.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(res.selected_names))
        _say(f"\n  -> {os.path.join(out, 'assembled_items.txt')}")
    _say(f"  -> {os.path.join(out, 'assembly.csv')}")
    return 0


def cmd_online_calib(args) -> int:
    """新题在线标定：借用已有模型的后验给新题定参数。"""
    from .calibration import calibrate, calibrate_new_items

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)

    new_items = [int(x) - 1 for x in args.new_items]
    bad = [i + 1 for i in new_items if not (0 <= i < q.n_items)]
    if bad:
        raise SystemExit(f"错误：--new-items 中的题号超出范围：{bad}")
    old_items = None
    if args.old_items:
        old_items = [int(x) - 1 for x in args.old_items]

    _rule("新题在线标定")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("新题数", len(new_items)),
        ("锚题数", len(old_items) if old_items else f"其余 {q.n_items - len(new_items)} 题"),
    ])

    model, res = calibrate(q, resp, model_type=args.model,
                           max_iter=args.max_iter, tol=args.tol)
    rep = calibrate_new_items(
        model, resp, new_items, old_items=old_items,
        prior=res.attr_prior,
        min_effective_n=args.min_n,
    )
    df = rep.to_dataframe()
    _say("")
    _table(
        [[r["item"], f"{float(r['n_masters']):.0f}", f"{float(r['n_nonmasters']):.0f}",
          f"{float(r['slip']):.3f}", f"{float(r['guess']):.3f}",
          f"{float(r['discrimination']):.3f}", r["status"]]
         for _, r in df.iterrows()],
        header=["题号", "掌握组人数", "未掌握组人数", "slip", "guess", "区分度", "状态"],
    )
    _say("")
    _say(rep.verdict())
    _say("提示：判定可靠性主要取决于掌握组 / 未掌握组的**有效人数**，而非总样本量。")

    write_table([df.columns.tolist()] + df.astype(str).values.tolist(),
                os.path.join(out, "online_calibration.csv"))
    if args.json:
        write_json(rep.to_dict(), os.path.join(out, "online_calibration.json"))
    _say(f"\n  -> {os.path.join(out, 'online_calibration.csv')}")
    return 0


def cmd_dif(args) -> int:
    """DIF 检验：按分组比较题目条件答对率。"""
    from .calibration import calibrate
    from .diagnostics import dif_analysis

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    ids = getattr(args, "_examinee_ids", None) or [f"E{i + 1}" for i in range(resp.shape[0])]
    groups = _load_groups(args, ids)
    if groups is None:
        raise SystemExit("错误：DIF 检验需要 --groups 指定分组表")
    out = _outdir(args)

    _rule("DIF 检验（分组公平性）")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("分组", "、".join(f"{g}:{groups.count(g)}" for g in sorted(set(groups)))),
        ("检验水平", args.alpha),
    ])

    model, _res = calibrate(q, resp, model_type=args.model,
                            max_iter=args.max_iter, tol=args.tol)
    rep = dif_analysis(model, resp, groups, alpha=args.alpha,
                       item_names=q.item_names, min_stratum=args.min_stratum)
    df = rep.to_dataframe()
    _say("")
    _table(
        [[r["item"], f"{float(r['p_ref']):.3f}", f"{float(r['p_focal']):.3f}",
          f"{float(r['p_value']):.4f}", f"{float(r['delta_ets']):.3f}",
          str(r["ets_class"]), str(r["favors"])] for _, r in df.iterrows()],
        header=["题号", "参照组 p", "焦点组 p", "p 值", "ΔETS", "等级", "偏向"],
    )
    _say("")
    _say(rep.summary())
    _say("提示：显著项需人工复核题干是否含分组特有的背景知识，"
         "而非直接判定题目有偏。")

    write_table([df.columns.tolist()] + df.astype(str).values.tolist(),
                os.path.join(out, "dif.csv"))
    if args.json:
        write_json(rep.to_dict(), os.path.join(out, "dif.json"))
    _say(f"\n  -> {os.path.join(out, 'dif.csv')}")
    return 0


def cmd_nonparam(args) -> int:
    """非参数分类：不估计题目参数，直接用距离判别知识状态。"""
    from .estimation.nonparametric import NONPARAMETRIC_METHODS, classify_nonparametric
    from .qmatrix import AttributeHierarchy

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)

    methods = [m.strip().lower() for m in (args.methods or ["gdd", "hamming"])]
    for m in methods:
        if m not in NONPARAMETRIC_METHODS:
            raise SystemExit(
                f"错误：未知非参数方法 '{m}'，可选: {sorted(NONPARAMETRIC_METHODS)}"
            )

    # 层级约束：复用 AttributeHierarchy 的快捷构造
    hier = None
    hname = (args.hierarchy or "none").strip().lower()
    if hname != "none":
        ctor = {
            "linear": AttributeHierarchy.linear,
            "convergent": AttributeHierarchy.convergent,
            "divergent": AttributeHierarchy.divergent,
        }.get(hname)
        if ctor is None:
            raise SystemExit("错误：--hierarchy 只支持 none / linear / convergent / divergent")
        hier = ctor(q.n_attrs, list(q.attr_names or []))

    truth = None
    if args.truth:
        truth, _ids = read_responses(args.truth, n_items=q.n_attrs)
        if truth.shape[0] != resp.shape[0]:
            raise SystemExit(
                f"错误：真实模式 {truth.shape[0]} 行与作答矩阵 {resp.shape[0]} 行不一致"
            )

    _rule("非参数分类（不需要题目参数）")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("方法", "、".join(f"{m}（{NONPARAMETRIC_METHODS[m]}）" for m in methods)),
        ("层级约束", hname if hier is not None else "无"),
        ("温度 T", args.temperature),
    ])

    results = {}
    for m in methods:
        res = classify_nonparametric(
            q, resp, method=m, hierarchy=hier,
            temperature=args.temperature, attr_names=q.attr_names,
        )
        if truth is not None:
            acc = res.accuracy(truth)
            res.pmr, res.amr = acc["pmr"], acc["amr"]
        results[m] = res

    _say("")
    rows = []
    for m, res in results.items():
        rows.append([
            m,
            "0",
            f"{res.mean_confidence():.4f}",
            f"{np.nanmean(res.distance_gap()):.4f}",
            "—" if res.pmr is None else f"{res.pmr:.4f}",
            "—" if res.amr is None else f"{res.amr:.4f}",
        ])
    _table(rows, header=["方法", "参数数", "平均把握", "平均距离差", "PMR", "AMR"])

    _say("")
    for m, res in results.items():
        _say(res.summary())
        _say("")

    if truth is None:
        _say("提示：未提供 --truth，无法计算 PMR/AMR；只输出判别结果与把握度。")
    else:
        _say("提示：非参数方法的把握度是 exp(-d/T) 归一化的**伪后验**，"
             "与参数模型的后验概率不可直接比较。")

    # 落盘：汇总表 + 各方法的被试模式
    summary_rows = [["method", "n_params", "mean_confidence", "mean_distance_gap",
                     "pmr", "amr"]]
    for m, res in results.items():
        summary_rows.append([
            m, "0", f"{res.mean_confidence():.6f}",
            f"{np.nanmean(res.distance_gap()):.6f}",
            "" if res.pmr is None else f"{res.pmr:.6f}",
            "" if res.amr is None else f"{res.amr:.6f}",
        ])
    write_table(summary_rows, os.path.join(out, "nonparametric.csv"))
    _say(f"\n  -> {os.path.join(out, 'nonparametric.csv')}")

    for m, res in results.items():
        path = os.path.join(out, f"patterns_{m}.csv")
        head = [["examinee"] + list(res.attr_names) + ["confidence", "distance_gap"]]
        gaps = res.distance_gap()
        for i in range(res.n_examinees):
            head.append(
                [f"E{i + 1}"]
                + [str(int(v)) for v in res.patterns[i]]
                + [f"{res.confidence[i]:.6f}", f"{gaps[i]:.6f}"]
            )
        write_table(head, path)
        _say(f"  -> {path}")

    if args.json:
        payload = {
            m: {
                "method": res.method,
                "n_params": 0,
                "mean_confidence": res.mean_confidence(),
                "mean_distance_gap": float(np.nanmean(res.distance_gap())),
                "pmr": res.pmr,
                "amr": res.amr,
                "mastery_proportion": [float(v) for v in res.mastery_proportion()],
                "attr_names": list(res.attr_names),
                "hierarchy": res.hierarchy,
                "temperature": res.temperature,
            }
            for m, res in results.items()
        }
        path = write_json(payload, os.path.join(out, "nonparametric.json"))
        _say(f"  -> {path}")
    return 0


def cmd_report_html(args) -> int:
    """一站式 HTML 报告：标定 → 拟合 → 图 → 单文件归档。"""
    from .pipeline import run_cdm_analysis
    from .report import build_cdm_report

    q, src = _load_q(args)
    resp = _load_responses(args, q.n_items)
    out = _outdir(args)

    _rule("单文件 HTML 报告")
    _kv([
        ("Q 矩阵", f"{src} · {q.n_items} 题 × {q.n_attrs} 属性"),
        ("作答矩阵", f"{os.path.basename(args.responses)} · {resp.shape[0]} 名被试"),
        ("模型", "、".join(args.models)),
    ])

    analysis = run_cdm_analysis(
        q, resp, models=args.models, max_iter=args.max_iter, tol=args.tol,
        absolute_fit=True, verbose=args.verbose,
    )
    report = build_cdm_report(
        analysis, resp,
        title=args.title or "认知诊断分析报告",
        subtitle=args.subtitle,
        author=args.author,
        include_figures=not args.no_figures,
        max_item_rows=args.max_item_rows,
    )
    path = report.save(os.path.join(out, "report.html"))
    size_kb = os.path.getsize(path) / 1024.0
    _say("")
    _kp = [
        ("最优模型", analysis.best_model),
        ("报告体积", f"{size_kb:.0f} KB（图表内联，可离线打开）"),
    ]
    if analysis.fits and analysis.best_model in analysis.fits:
        fit = analysis.fits[analysis.best_model]
        if fit.pmr is not None:
            _kp.append(("模式判准率", f"{fit.pmr:.4f}"))
        if fit.amr is not None:
            _kp.append(("属性判准率", f"{fit.amr:.4f}"))
    _kv(_kp)
    _say(f"\n  -> {path}")
    return 0


def _load_model_for_q(args, q):
    """题库装配只需要"某套题目参数"，优先级：--params > --bank > 预设随机值。"""
    from .models import create_model

    path = getattr(args, "params", None)
    if path:
        name, kwargs = read_params(path)
        return create_model(name, q, **kwargs), f"{name}（{os.path.basename(path)}）"

    if getattr(args, "bank", None):
        bank = _load_bank(args)
        name = bank.get("model", args.model)
        kwargs = {}
        if bank.get("slip") is not None:
            kwargs["slip"] = np.asarray(bank["slip"], dtype=float)
        if bank.get("guess") is not None:
            kwargs["guess"] = np.asarray(bank["guess"], dtype=float)
        if kwargs:
            return create_model(name, q, **kwargs), f"{name}（{os.path.basename(args.bank)}）"

    return create_model(args.model, q), f"{args.model}（默认参数）"


# ====================================================================== #
# 参数解析
# ====================================================================== #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdcat",
        description="CD-CAT 认知诊断自适应测验：标定 / 仿真 / 诊断 / 报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  cdcat demo\n"
            "  cdcat simulate --preset \"分数运算 (4 属性)\" -n 300 -R 5 -o out/\n"
            "  cdcat calibrate -q q.csv -r resp.csv -m DINA -o out/\n"
            "  cdcat report --bank out/calibrated_bank.json -r resp.csv -o out/\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"cdcat {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<命令>")

    # ---- 通用输入参数 ----
    def add_input(p, need_resp: bool = False):
        p.add_argument("-q", "--q", metavar="Q.CSV", help="Q 矩阵 CSV 文件")
        p.add_argument("--preset", metavar="NAME", help="使用内置题库（见 cdcat presets）")
        if need_resp:
            p.add_argument("-r", "--responses", metavar="RESP.CSV",
                           help="作答矩阵 CSV（行=被试，列=题目）")
        p.add_argument("-o", "--out", metavar="DIR", default="cdcat_out", help="输出目录")
        p.add_argument("-m", "--model", default="DINA",
                       choices=["DINA", "DINO", "G-DINA", "GDINA", "RRUM", "ACDM", "LLM"],
                       help="响应模型（默认 DINA）")
        p.add_argument("--json", action="store_true", help="同时输出机器可读结果")
        p.add_argument("--figures", action="store_true", help="生成 matplotlib 静态图")
        p.add_argument("--verbose", action="store_true")

    def add_calib(p):
        p.add_argument("--max-iter", type=int, default=500, dest="max_iter")
        p.add_argument("--tol", type=float, default=1e-6)
        p.add_argument("--estimate-prior", action="store_true", default=True,
                       dest="estimate_prior", help="估计经验贝叶斯属性先验（默认开启）")

    # ---- presets ----
    sub.add_parser("presets", help="列出内置题库").set_defaults(func=cmd_presets)

    # ---- list ----
    sub.add_parser("list", help="列出可用的模型 / 准则 / 规则 / 曝光控制").set_defaults(func=cmd_list)

    # ---- demo ----
    p = sub.add_parser("demo", help="一键跑通演示（内置题库 → 仿真 → 群体报告）")
    p.add_argument("--preset", default=list_presets()[0], help="内置题库名称")
    p.add_argument("-m", "--model", default="DINA")
    p.add_argument("-s", "--selector", default="PWKL", choices=sorted(_SELECTORS))
    p.add_argument("-n", "--n-examinees", type=int, default=300, dest="n_examinees")
    p.add_argument("-R", "--n-rep", type=int, default=5, dest="n_rep")
    p.add_argument("--max-items", type=int, default=40, dest="max_items")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", default="cdcat_out")
    p.add_argument("--figures", action="store_true")
    p.set_defaults(func=cmd_demo)

    # ---- calibrate ----
    p = sub.add_parser("calibrate", help="MMLE/EM 标定题目参数")
    add_input(p, need_resp=True)
    add_calib(p)
    p.set_defaults(func=cmd_calibrate)

    # ---- simulate ----
    p = sub.add_parser("simulate", help="蒙特卡洛仿真（可多准则对比）")
    add_input(p)
    p.add_argument("--bank", metavar="BANK.JSON", help="题库 JSON（含题目参数）")
    p.add_argument("-s", "--selector", default="PWKL", help="选题准则", choices=sorted(_SELECTORS))
    p.add_argument("--compare", nargs="+", metavar="SEL", choices=sorted(_SELECTORS),
                   help="多准则横向对比")
    p.add_argument("-n", "--n-examinees", type=int, default=300, dest="n_examinees")
    p.add_argument("-R", "--n-rep", type=int, default=5, dest="n_rep")
    p.add_argument("--min-items", type=int, default=5, dest="min_items")
    p.add_argument("--max-items", type=int, default=40, dest="max_items")
    p.add_argument("-e", "--estimator", default="MAP", choices=["MAP", "EAP", "MLE"])
    p.add_argument("--exposure", default=None, help="曝光控制策略（默认无控制）")
    p.add_argument("-j", "--n-jobs", type=int, default=1, dest="n_jobs")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    # ---- diagnose ----
    p = sub.add_parser("diagnose", help="绝对拟合 / 局部依赖 / 题目分析")
    add_input(p, need_resp=True)
    add_calib(p)
    p.set_defaults(func=cmd_diagnose)

    # ---- validate-q ----
    p = sub.add_parser("validate-q", help="Q 矩阵校验与自动修正建议")
    add_input(p, need_resp=True)
    add_calib(p)
    p.add_argument("--pvaf", type=float, default=0.95, help="PVAF 判定阈值（默认 0.95）")
    p.add_argument("--top", type=int, default=20, help="最多展示多少条建议")
    p.set_defaults(func=cmd_validate_q)

    # ---- report ----
    p = sub.add_parser("report", help="群体（班级）诊断报告")
    add_input(p, need_resp=True)
    p.add_argument("--bank", metavar="BANK.JSON", help="题库 JSON（含题目参数）")
    p.add_argument("--params", metavar="PARAMS.JSON", help="题目参数 JSON")
    p.add_argument("--groups", metavar="GROUPS.CSV",
                   help="分组表 CSV（首列被试 ID，次列组名/班级）；可与 --responses 一起给")
    p.add_argument("--label", default=None, help="报告标题")
    p.add_argument("--top", type=int, default=3, help="薄弱知识点个数")
    p.add_argument("--missing", default="skip",
                   choices=["skip", "wrong", "ignore"],
                   help="缺失作答处理：skip=跳过该题 / wrong=按答错 / ignore=报错")
    p.set_defaults(func=cmd_report)

    # ---- ppmc ----
    p = sub.add_parser("ppmc", help="后验预测检验（模型能否再现数据的二阶结构）")
    add_input(p, need_resp=True)
    add_calib(p)
    p.add_argument("--replicates", type=int, default=500, help="复制数据集数（默认 500）")
    p.add_argument("--alpha", type=float, default=0.05, help="PPP 判读的显著性水平")
    p.add_argument("--no-refit", action="store_true", dest="no_refit",
                   help="复制侧不重估题目参数（更快，但两侧口径不再对称）")
    p.add_argument("--max-items-pair", type=int, default=150, dest="max_items_pair",
                   help="题目对统计量的最多题数（超出则随机抽样，默认 150）")
    p.set_defaults(func=cmd_ppmc)

    # ---- hier-test ----
    p = sub.add_parser("hier-test", help="比较属性层级结构（BIC 择优）")
    add_input(p, need_resp=True)
    add_calib(p)
    p.add_argument("--structures", nargs="+", metavar="S",
                   choices=["none", "linear", "convergent", "divergent"],
                   help="候选结构（默认全部四种）")
    p.set_defaults(func=cmd_hier_test)

    # ---- assemble ----
    p = sub.add_parser("assemble", help="题库装配：按覆盖与题量约束挑短测验")
    add_input(p)
    p.add_argument("--bank", metavar="BANK.JSON", help="题库 JSON（含题目参数）")
    p.add_argument("--params", metavar="PARAMS.JSON", help="题目参数 JSON")
    p.add_argument("-k", "--n-items", type=int, default=None, dest="n_items",
                   help="目标题量（默认 2K+4）")
    p.add_argument("--min-per-attr", type=int, default=1, dest="min_per_attr")
    p.add_argument("--max-per-attr", type=int, default=None, dest="max_per_attr")
    p.add_argument("--must", nargs="+", metavar="I", help="必须入选的题号（1 起）")
    p.add_argument("--no-search", action="store_true", dest="no_search",
                   help="跳过成对交换局部搜索（更快）")
    p.add_argument("--max-rounds", type=int, default=200, dest="max_rounds")
    p.add_argument("--export-bank", action="store_true", dest="export_bank",
                   help="额外输出所选题目清单 txt")
    p.set_defaults(func=cmd_assemble)

    # ---- online-calib ----
    p = sub.add_parser("online-calib", help="新题在线标定（借用已有模型的后验）")
    add_input(p, need_resp=True)
    add_calib(p)
    p.add_argument("--new-items", nargs="+", metavar="I", required=True,
                   dest="new_items", help="新题题号（1 起，可多个）")
    p.add_argument("--old-items", nargs="+", metavar="I", dest="old_items",
                   help="锚题题号（默认用其余全部题目）")
    p.add_argument("--min-n", type=float, default=10.0, dest="min_n",
                   help="有效人数下限（低于此值标记为不稳定）")
    p.set_defaults(func=cmd_online_calib)

    # ---- dif ----
    p = sub.add_parser("dif", help="DIF 检验（分组公平性）")
    add_input(p, need_resp=True)
    add_calib(p)
    p.add_argument("--groups", metavar="GROUPS.CSV", help="分组表 CSV（首列 ID，次列组名）")
    p.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    p.add_argument("--min-stratum", type=int, default=1, dest="min_stratum",
                   help="每个能力层的最少人数")
    p.set_defaults(func=cmd_dif)

    # ---- nonparam ----
    p = sub.add_parser("nonparam", help="非参数分类（不需要题目参数）")
    add_input(p, need_resp=True)
    p.add_argument("--methods", nargs="+", default=["gdd", "hamming"],
                   help="判别方法：gdd（方差加权）/ hamming（等权），可多选")
    p.add_argument("--hierarchy", default="none",
                   help="层级约束 none / linear / convergent / divergent")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="伪后验温度 T（距离单位为等效题数，默认 1.0）")
    p.add_argument("--truth", metavar="TRUTH.CSV",
                   help="真实属性模式 CSV（行=被试，列=属性），给出则计算 PMR/AMR")
    p.set_defaults(func=cmd_nonparam)

    # ---- report-html ----
    p = sub.add_parser("report-html", help="生成单文件 HTML 诊断报告（图表内联）")
    add_input(p, need_resp=True)
    add_calib(p)
    p.add_argument("--models", nargs="+", default=["DINA", "G-DINA"],
                   help="要比较的模型（默认 DINA G-DINA）")
    p.add_argument("--title", default=None, help="报告标题")
    p.add_argument("--subtitle", default=None, help="副标题")
    p.add_argument("--author", default=None, help="作者 / 机构")
    p.add_argument("--no-figures", action="store_true", dest="no_figures",
                   help="不内联图表（体积更小）")
    p.add_argument("--max-item-rows", type=int, default=30, dest="max_item_rows",
                   help="题目表最多行数")
    p.set_defaults(func=cmd_report_html)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _say("\n已中断")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
