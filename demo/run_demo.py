"""CD-CAT 系统完整演示脚本（命令行版）。

不需要 Web 界面即可完整体验系统全部能力，并生成 PNG 图表。

运行::

    python demo/run_demo.py

输出：
- 控制台打印各环节结果
- ``demo/output/`` 目录下生成可视化图表

演示共 10 步：建库 → 标定 → 施测 → 准则对比 → 仿真 → 结构分析 →
出图 → 多模型 CDM 分析 → 绝对拟合与经验贝叶斯先验诊断 → 落地工具链。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdcat import (  # noqa: E402
    QMatrix,
    DINA,
    DINO,
    CDCATSimulator,
    EMCalibrator,
    Evaluator,
    build_test_items,
    analyse_identifiability,
    analyse_items,
    build_examinee_report,
    create_model,
    run_cdm_analysis,
)
from cdcat.utils import pattern_to_str  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def section(text: str) -> None:
    print(f"\n--- {text} ---")


# ====================================================================== #
def step1_build_bank():
    banner("步骤 1 / 10  构建题库与 Q 矩阵诊断")
    q = QMatrix.from_random(
        n_items=60, n_attrs=5, min_attrs_per_item=1, max_attrs_per_item=2,
        seed=42, attr_names=["通分", "约分", "分数加减", "分数乘除", "应用题"],
    )
    items = build_test_items(60, 5, min_attrs_per_item=1, max_attrs_per_item=2, seed=42)
    print(q)
    print(f"\n属性：{q.attr_names}")
    print(f"属性考查次数：{q.attribute_exposure().tolist()}")
    print(f"基础校验：{q.validate() or '通过，无问题'}")

    diag = q.diagnose()
    print(f"关联图连通：{diag.is_connected}")
    print(f"重复 Q 向量组数：{len(diag.duplicate_rows)}")
    print(f"理想反应矩阵形状 E: {q.ideal_response().shape}  (2^K x J)")
    print("\n首 5 道题的 Q 向量：")
    for i in range(5):
        attrs = [q.attr_names[k] for k in q.required_attrs(i)]
        print(f"  题 {q.item_names[i]:4s} -> 考查 {' + '.join(attrs)}")
    return q, items


def step2_calibrate(q):
    banner("步骤 2 / 10  题目参数标定 (MMLE/EM)")
    true_items = build_test_items(60, 5, min_attrs_per_item=1, max_attrs_per_item=2, seed=42)
    true_model = DINA(q, slip=true_items["slip"], guess=true_items["guess"])

    section("生成模拟作答数据（1500 名考生）")
    rng = np.random.default_rng(0)
    n = 1500
    alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, n)]
    X = np.array([true_model.simulate(a, rng=rng) for a in alphas], dtype=float)
    print(f"作答矩阵形状：{X.shape}")
    print(f"整体答对率：{X.mean():.3f}")

    section("EM 迭代")
    cal = EMCalibrator(q, model_type="DINA", max_iter=300, tol=1e-7)
    result = cal.fit(X, verbose=True)
    print()
    print(result.summary())

    section("参数恢复精度")
    print(f"  slip  相关系数  : {np.corrcoef(result.slip, true_items['slip'])[0,1]:.4f}")
    print(f"  guess 相关系数  : {np.corrcoef(result.guess, true_items['guess'])[0,1]:.4f}")
    print(f"  slip  平均绝对误差: {np.abs(result.slip - true_items['slip']).mean():.4f}")
    print(f"  guess 平均绝对误差: {np.abs(result.guess - true_items['guess']).mean():.4f}")

    # 用标定参数评估考生
    est_patterns = cal.examinee_patterns(X, result.slip, result.guess)
    attr_acc = np.mean([np.mean(est_patterns[i] == alphas[i]) for i in range(n)])
    full_acc = np.mean([np.array_equal(est_patterns[i], alphas[i]) for i in range(n)])
    print(f"\n  考生属性判准率 (全卷): {attr_acc:.4f}")
    print(f"  考生模式判准率 (全卷): {full_acc:.4f}")

    return result


def step3_administer(q, cal_result):
    banner("步骤 3 / 10  单被试在线施测演示")
    sim = CDCATSimulator(
        q, model="DINA",
        model_params={"slip": cal_result.slip, "guess": cal_result.guess},
        selector="PWKL", estimator="MAP", max_items=30, min_items=3,
    )
    rng = np.random.default_rng(99)
    true_pattern = np.array([1, 0, 1, 1, 0], dtype=np.int8)
    print(f"考生真实掌握模式：{pattern_to_str(true_pattern)}")
    print("  " + "  ".join(
        f"{q.attr_names[i]}={'掌握' if true_pattern[i] else '未掌握'}" for i in range(5)
    ))

    rec = sim.administer(true_pattern, rng=rng)
    section("施测过程")
    print(f"{'次序':>4} {'题号':>6} {'考查属性':<24} {'作答':>4} {'MAP估计模式':>12}")
    final_res = sim.estimator.estimate(
        np.array(rec.responses, dtype=float), np.array(rec.administered, dtype=int)
    )
    for t, (item, resp) in enumerate(zip(rec.administered, rec.responses)):
        attrs = "+".join(q.attr_names[k] for k in q.required_attrs(item))
        marg = rec.marginal_history[t + 1]
        est = pattern_to_str((marg >= 0.5).astype(np.int8))
        print(f"{t+1:>4} {q.item_names[item]:>6} {attrs:<24} {'✓' if resp else '✗':>4} {est:>12}")

    section("施测结果")
    print(f"  作答题目数   : {rec.n_answered}")
    print(f"  估计模式     : {pattern_to_str(rec.est_pattern)}")
    print(f"  真实模式     : {pattern_to_str(true_pattern)}")
    print(f"  模式是否判对 : {'是' if rec.pattern_correct else '否'}")
    print(f"  属性判准率   : {rec.attribute_accuracy:.3f}")
    print(f"  终止原因     : {rec.stop_reason}")
    print("  各属性后验   : " + "  ".join(
        f"{q.attr_names[i]}={final_res.marginal[i]:.3f}" for i in range(5)
    ))
    return rec


def step4_compare(q, cal_result):
    banner("步骤 4 / 10  选题准则横向对比")
    sim = CDCATSimulator(
        q, model="DINA",
        model_params={"slip": cal_result.slip, "guess": cal_result.guess},
        selector="PWKL",
    )
    # MIV 放在最后：它不读 slip/guess，作为"参数到底带来多少增益"的对照
    # 前三行是参数化准则，后四行完全不读题目参数（只读 Q 矩阵）
    criteria = ["KL", "PWKL", "MPWKL", "GDI", "SHE", "CDI", "MIV", "NPSHE", "ERHD", "AMI"]
    section(f"固定 20 题下对比 {len(criteria)} 种准则（200 被试 x 3 次重复）")
    results = sim.compare_criteria(
        criteria, n_examinees=200, n_replications=3, seed=0, fixed_length=20,
    )
    ev = Evaluator()
    print()
    print(ev.compare_text(results))

    section("双维度评价（题量由固定，看精度与曝光）")
    for name, res in results.items():
        print(
            f"  {name:6s}  PMR={res.pmr():.4f}  AMR={res.amr():.4f}  "
            f"利用率={res.test_utilization():.3f}  曝光SD={res.exposure_uniformity():.4f}"
        )
    return results


def step5_simulation(q, cal_result):
    banner("步骤 5 / 10  蒙特卡洛仿真与评估")
    sim = CDCATSimulator(
        q, model="DINA",
        model_params={"slip": cal_result.slip, "guess": cal_result.guess},
        selector="PWKL", estimator="MAP", max_items=30,
    )
    section("自适应终止（模式级置信 0.8）下的完整仿真：300 被试 x 5 次重复")
    result = sim.run(n_examinees=300, n_replications=5, seed=7)
    print()
    print(result.summary())

    section("收敛曲线：判准率随题量增长")
    curve = sim.convergence_curve(n_examinees=150, max_items=25, seed=7)
    print(f"{'题量':>6} {'PMR':>8} {'AMR':>8}")
    for t in [1, 3, 5, 8, 10, 15, 20, 25]:
        print(f"{t:>6} {curve['pmr'][t]:>8.4f} {curve['amr'][t]:>8.4f}")

    section("完整评估报告")
    print(Evaluator().full_report(result, title="PWKL 自适应测验评估"))
    return result, curve


def step6_structure_analysis(q, cal_result, admin_rec, sim_result):
    banner("步骤 6 / 10  结构可辨识性、题目分析与个人报告")
    section("结构可辨识性分析")
    rep = analyse_identifiability(q)
    print(f"  模式总数           : {rep.n_patterns}")
    print(f"  可唯一辨识模式数   : {rep.n_distinct_response_vectors}")
    print(f"  不可辨识模式数     : {rep.n_unidentifiable_patterns}")
    print(f"  PMR 理论上限       : {rep.expected_pmr_ceiling:.4f}")
    print(f"  Q 矩阵是否完备     : {'是' if rep.is_complete else '否'}")
    if rep.equivalence_classes:
        print("  不可区分模式等价组（前 3 组）:")
        for grp in rep.equivalence_classes[:3]:
            strs = " ≡ ".join(pattern_to_str(q.attribute_patterns()[i]) for i in grp)
            print(f"    {strs}")
    if rep.warnings:
        print("  改进建议:")
        for w in rep.warnings[:4]:
            print(f"    - {w}")

    section("实际 PMR 与理论上限对比")
    cmp = sim_result.pmr_vs_ceiling()
    print(f"  实际 PMR     : {cmp['pmr']:.4f}")
    print(f"  理论上限     : {cmp['ceiling']:.4f}")
    print(f"  差距         : {cmp['gap']:.4f}")
    print(f"  算法效率     : {cmp['efficiency']:.1%}")
    print(f"  结论         : {cmp['verdict']}")

    section("题目信息量与冗余分析")
    model_obj = create_model(
        "DINA", q, slip=cal_result.slip, guess=cal_result.guess
    )
    ia = analyse_items(model_obj, n_simulate=4000, seed=2024)
    print(f"  平均区分度       : {ia.discrimination.mean():.4f}")
    print(f"  最强题区分度     : {ia.discrimination.max():.4f}")
    print(f"  最弱题区分度     : {ia.discrimination.min():.4f}")
    print(f"  结构冗余题组数   : {len(ia.structural_duplicates)}")
    print("\n  区分度最高的 5 道题:")
    for i in ia.strongest_items[:5]:
        attrs = "+".join(q.attr_names[k] for k in q.required_attrs(i))
        print(f"    {q.item_names[i]:6s} D={ia.discrimination[i]:.3f}  I={ia.information[i]:.3f}  [{attrs}]")
    print("  区分度最低的 5 道题:")
    for i in ia.weakest_items[:5]:
        attrs = "+".join(q.attr_names[k] for k in q.required_attrs(i))
        print(f"    {q.item_names[i]:6s} D={ia.discrimination[i]:.3f}  I={ia.information[i]:.3f}  [{attrs}]")
    if ia.structural_duplicates:
        print("  结构冗余题组（Q 向量完全相同）:")
        for grp in ia.structural_duplicates[:5]:
            attrs = "+".join(q.attr_names[k] for k in q.required_attrs(grp[0]))
            print(f"    [{attrs}] -> {', '.join(q.item_names[i] for i in grp[:8])}")

    section("个人诊断报告")
    er = build_examinee_report(admin_rec, q.attr_names, q.item_names, q)
    print(er.to_text())
    return rep, ia


def step7_plot(q, cal_result, admin_rec, compare_results, sim_result, curve):
    banner("步骤 7 / 10  生成可视化图表")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # ---- 图 1：后验收敛曲线 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    mh = admin_rec.marginal_history
    colors = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#BA7517"]
    for k in range(mh.shape[1]):
        ax.plot(range(len(mh)), mh[:, k], marker="o", markersize=4,
                color=colors[k % len(colors)], label=q.attr_names[k], linewidth=2)
    ax.axhline(0.5, ls=":", color="#888780", lw=1)
    ax.axhline(0.8, ls="--", color="#5DCAA5", lw=1)
    ax.axhline(0.2, ls="--", color="#5DCAA5", lw=1)
    ax.set_xlabel("已作答题数")
    ax.set_ylabel("P(掌握该属性)")
    ax.set_title("属性掌握概率后验收敛过程（单被试）")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = OUT / "01_posterior_convergence.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  已保存 {p1}")

    # ---- 图 2：准则对比（PMR/AMR 分组柱状 + 区分度） ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = list(compare_results.keys())
    x = np.arange(len(names))
    w = 0.38
    pmrs = [compare_results[n].pmr() for n in names]
    amrs = [compare_results[n].amr() for n in names]
    axes[0].bar(x - w / 2, pmrs, w, label="模式判准率 PMR", color="#534AB7")
    axes[0].bar(x + w / 2, amrs, w, label="属性判准率 AMR", color="#1D9E75")
    for i, (p, a) in enumerate(zip(pmrs, amrs)):
        axes[0].text(i - w / 2, p + 0.012, f"{p:.3f}", ha="center", fontsize=9)
        axes[0].text(i + w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("判准率")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("各准则判准率对比（固定 20 题）")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.25, axis="y")

    # 右图：PMR 放大对比，突出准则间差异
    order = np.argsort(pmrs)
    axes[1].barh([names[i] for i in order], [pmrs[i] for i in order],
                 color="#7F77DD", height=0.6)
    for i, idx in enumerate(order):
        axes[1].text(pmrs[idx] + 0.004, i, f"{pmrs[idx]:.4f}", va="center", fontsize=9)
    axes[1].set_xlabel("模式判准率 PMR")
    axes[1].set_title("PMR 放大对比（横轴起点 0.7）")
    lo = max(0.7, min(pmrs) - 0.02)
    axes[1].set_xlim(lo, 1.002)
    axes[1].grid(alpha=0.25, axis="x")
    fig.tight_layout()
    p2 = OUT / "02_criteria_comparison.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  已保存 {p2}")

    # ---- 图 3：收敛曲线 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["n_items"], curve["pmr"], marker="s", color="#534AB7",
            label="模式判准率 PMR", linewidth=2.5, markersize=5)
    ax.plot(curve["n_items"], curve["amr"], marker="o", color="#1D9E75",
            label="属性判准率 AMR", linewidth=2.5, markersize=5)
    ax.set_xlabel("题量")
    ax.set_ylabel("判准率")
    ax.set_title("判准率随题量收敛曲线")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p3 = OUT / "03_convergence_curve.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  已保存 {p3}")

    # ---- 图 4：题目曝光 + 属性判准 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    usage = sim_result.item_usage()
    axes[0].bar(range(len(usage)), usage, color="#378ADD", width=0.85)
    axes[0].set_xlabel("题目编号")
    axes[0].set_ylabel("曝光次数")
    axes[0].set_title("题目曝光分布")
    axes[0].grid(alpha=0.25, axis="y")

    attr_acc = sim_result.attribute_accuracy_by_attr()
    axes[1].bar(q.attr_names, attr_acc, color="#1D9E75", width=0.6)
    for i, v in enumerate(attr_acc):
        axes[1].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("判准率")
    axes[1].set_title("各属性判准率")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p4 = OUT / "04_exposure_attribute.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  已保存 {p4}")

    # ---- 图 5：Q 矩阵热图 ----
    fig, ax = plt.subplots(figsize=(5.5, 11))
    ax.imshow(q.matrix, aspect="auto", cmap="Greens", interpolation="nearest")
    ax.set_xticks(range(q.n_attrs))
    ax.set_xticklabels(q.attr_names, rotation=45, ha="right")
    ax.set_yticks(range(q.n_items))
    ax.set_yticklabels(q.item_names, fontsize=6)
    ax.set_title("Q 矩阵 (绿色=考查该属性)")
    fig.tight_layout()
    p5 = OUT / "05_qmatrix.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"  已保存 {p5}")

    # ---- 图 6：题目区分度 + 结构可辨识性 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    model_obj = create_model("DINA", q, slip=cal_result.slip, guess=cal_result.guess)
    ia = analyse_items(model_obj, n_simulate=4000, seed=2024)
    order = np.argsort(-ia.discrimination)[:25]
    vals = ia.discrimination[order]
    labels = [q.item_names[i] for i in order]
    colors_d = ["#1D9E75" if v >= 0.5 else "#D85A30" if v < 0.3 else "#BA7517" for v in vals]
    axes[0].barh(range(len(vals)), vals, color=colors_d, height=0.7)
    axes[0].set_yticks(range(len(vals)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("区分度 D")
    axes[0].set_title("题目区分度 Top 25")
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(alpha=0.25, axis="x")

    rep = analyse_identifiability(q)
    distinct = rep.n_distinct_response_vectors
    ambig = rep.n_patterns - distinct
    axes[1].pie(
        [distinct, ambig],
        labels=["可唯一辨识", "不可区分"],
        colors=["#1D9E75", "#D85A30"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white"),
        textprops=dict(fontsize=10),
    )
    axes[1].set_title(f"结构可辨识性（PMR 上限 {rep.expected_pmr_ceiling:.3f}）")
    fig.tight_layout()
    p6 = OUT / "06_item_discrimination.png"
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    print(f"  已保存 {p6}")

    return [p1, p2, p3, p4, p5, p6]


# ====================================================================== #
def step8_cdm_analysis(q):
    """多模型 MMLE/EM 标定 + 拟合指标 + 诊断 + 报告图。"""
    banner("步骤 8 / 10  多模型 CDM 标定与诊断分析 (DINA / DINO / G-DINA / RRUM / ACDM / LLM)")

    true_items = build_test_items(
        60, 5, min_attrs_per_item=1, max_attrs_per_item=2, seed=42
    )
    true_model = DINA(q, slip=true_items["slip"], guess=true_items["guess"])

    section("生成模拟作答数据（1500 名考生，真实模型 = DINA）")
    rng = np.random.default_rng(1)
    n = 1500
    alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, n)]
    X = np.array([true_model.simulate(a, rng=rng) for a in alphas], dtype=float)
    print(f"作答矩阵形状：{X.shape}")

    section("六模型标定 + 模型比较（按 BIC 择优）")
    analysis = run_cdm_analysis(
        q, X,
        models=("DINA", "DINO", "G-DINA", "RRUM", "ACDM", "LLM"),
        true_slip=true_items["slip"],
        true_guess=true_items["guess"],
        true_patterns=alphas,
        max_iter=300,
        tol=1e-7,
        nonparametric=("gdd", "hamming"),
    )

    print()
    print(analysis.fit_table().to_string(index=False))
    print()
    print(analysis.comparison.summary())
    print()
    section("非参数基线（不估计题目参数，0 个自由参数）")
    print(analysis.nonparametric_table().to_string(index=False))
    for name, res in analysis.nonparametric.items():
        print(f"  {name}: 平均判别把握={res.mean_confidence():.4f}  "
              f"平均距离差={np.nanmean(res.distance_gap()):.4f}")

    if analysis.lrt:
        lrt = analysis.lrt
        print(
            f"\n似然比检验 [{lrt.get('restricted')} vs {lrt.get('full')}]: "
            f"LR={lrt['lr_statistic']:.2f}, df={lrt['df']}, "
            f"p={lrt['p_value']:.4g}, "
            f"{'显著' if lrt['significant'] else '不显著'}"
        )

    if analysis.recovery:
        section("参数恢复评价")
        for k, v in analysis.recovery.items():
            print(
                f"  {k:<14s} r={v['correlation']:.4f}  "
                f"MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}"
            )

    section(f"被试知识状态诊断（最优模型 {analysis.best_model}）")
    print(analysis.diagnosis[analysis.best_model].summary())

    section("导出报告")
    xlsx = OUT / "08_cdm_analysis.xlsx"
    try:
        analysis.to_excel(str(xlsx))
        print(f"  已保存 {xlsx}")
    except ImportError as exc:
        print(f"  跳过 Excel 导出：{exc}")

    section("生成标定分析报告图")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cdcat.calibration_plots import calibration_report_figure

    fig = calibration_report_figure(analysis)
    png = OUT / "07_cdm_calibration_report.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {png}")

    return [png, xlsx] if xlsx.exists() else [png]


def step9_absolute_fit(q):
    """绝对拟合诊断 + Yen Q3 局部依赖 + 经验贝叶斯属性先验。"""
    banner("步骤 9 / 10  绝对拟合诊断、局部依赖 Q3 与经验贝叶斯先验 π(α)")

    true_items = build_test_items(
        30, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=11
    )
    q4 = QMatrix.from_random(30, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=11)

    # ---- 场景 1：正确模型（真实 DINO，用 DINO 拟合）----
    section("场景 A：模型设定正确（真实 = DINO，拟合 = DINO）")
    truth = DINO(q4, slip=true_items["slip"], guess=true_items["guess"])
    rng = np.random.default_rng(5)
    n = 2000
    alphas = q4.attribute_patterns()[rng.integers(0, q4.n_patterns, n)]
    X = np.array([truth.simulate(a, rng=rng) for a in alphas], dtype=float)

    ana_ok = run_cdm_analysis(
        q4, X, models=("DINO",), true_slip=true_items["slip"],
        true_guess=true_items["guess"], true_patterns=alphas,
        max_iter=300, tol=1e-7, absolute_fit=True,
    )
    af_ok = ana_ok.absolute_fit
    print(f"  不合格题目         : {len(af_ok.item_fit.poorly_fitting(0.05))}/{q4.n_items}")
    print(f"  最大残差均值       : {af_ok.item_fit.max_abs_residual.mean():.4f}")
    print(f"  Q3 超阈值(>0.2)对数: {af_ok.local_dependence.n_flagged}")
    print(f"  最大 Q3            : {af_ok.local_dependence.max_q3:.4f}")

    # ---- 场景 2：模型误设（真实 DINO，用 DINA 拟合）----
    section("场景 B：模型误设（真实 = DINO，拟合 = DINA）—— 应当大规模报警")
    ana_bad = run_cdm_analysis(
        q4, X, models=("DINA",), true_slip=true_items["slip"],
        true_guess=true_items["guess"], true_patterns=alphas,
        max_iter=300, tol=1e-7, absolute_fit=True,
    )
    af_bad = ana_bad.absolute_fit
    n_bad = len(af_bad.item_fit.poorly_fitting(0.05))
    print(f"  不合格题目         : {n_bad}/{q4.n_items} ({n_bad/q4.n_items:.0%})")
    print(f"  最大残差均值       : {af_bad.item_fit.max_abs_residual.mean():.4f}")
    print(f"  Q3 超阈值(>0.2)对数: {af_bad.local_dependence.n_flagged}")
    print(f"  最大 Q3            : {af_bad.local_dependence.max_q3:.4f}")
    print("\n  >> 正确模型下几乎无警报，模型误设时不合格率与 Q3 同时飙升，")
    print("     说明该诊断能有效识别「模型-数据不匹配」。")

    section("不合格题目明细（场景 B，残差最大的 8 题）")
    print(af_bad.item_fit.to_dataframe().head(8).to_string(index=False))

    # ---- 场景 3：注入局部依赖（隐藏属性）----
    section("场景 C：注入局部依赖（第 6、19 题共享 Q 矩阵未记录的隐藏属性）")
    inj = (5, 18)                       # 0 基索引 -> 展示为 I6 / I19
    hidden = rng.random(n) < 0.5
    X2 = X.copy()
    for j in inj:
        hi = rng.random(n) < 0.85
        lo = rng.random(n) < 0.15
        X2[:, j] = np.where(hidden, hi, lo).astype(float)
    ana_ld = run_cdm_analysis(
        q4, X2, models=("DINO",), max_iter=300, tol=1e-7, absolute_fit=True,
    )
    ld = ana_ld.absolute_fit.local_dependence
    print(f"  全体平均 Q3  : {ld.mean_q3:.4f}   最大 Q3: {ld.max_q3:.4f}")
    print(f"  超阈值对数   : {ld.n_flagged}")
    print("  Q3 最高的题目对:")
    for a, b, v in ld.pairs[:3]:
        print(f"    I{a + 1:<3d} × I{b + 1:<3d}  Q3={v:.4f}")
    hit = any(frozenset((a, b)) == frozenset(inj) for a, b, _ in ld.pairs[:3])
    print(f"  >> 被注入的 (I6, I19) 是否被检出在 top-3: {'是' if hit else '否'}")

    # ---- 场景 4：经验贝叶斯属性先验 π(α) ----
    section("场景 D：经验贝叶斯属性先验 π(α)（属性分布非均匀时收益显著）")
    q30 = QMatrix.from_random(30, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=7)
    it30 = build_test_items(30, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=7)
    truth30 = DINA(q30, slip=it30["slip"], guess=it30["guess"])
    rng2 = np.random.default_rng(3)
    s = q30.n_patterns
    w = rng2.dirichlet(np.ones(s) * 0.4)
    true_prior = w / w.sum()
    nn = 1500
    al30 = q30.attribute_patterns()[rng2.choice(s, size=nn, p=true_prior)]
    X30 = np.array([truth30.simulate(a, rng=rng2) for a in al30], dtype=float)

    for label, kw in [("固定均匀先验", dict(estimate_prior=False)),
                      ("估计先验 π(α)", dict(estimate_prior=True))]:
        cal = EMCalibrator(q30, model_type="DINA", max_iter=500, tol=1e-7, **kw)
        res = cal.fit(X30)
        est = cal.examinee_patterns(X30, res.slip, res.guess)
        pmr = np.mean([np.array_equal(est[i], al30[i]) for i in range(nn)])
        amr = np.mean([np.mean(est[i] == al30[i]) for i in range(nn)])
        print(f"  {label}: 迭代 {res.n_iterations:>3d} 次  自由参数 {res.n_params}  "
              f"slipMAE={np.abs(res.slip - it30['slip']).mean():.4f}  "
              f"guessMAE={np.abs(res.guess - it30['guess']).mean():.4f}  "
              f"PMR={pmr:.4f}  AMR={amr:.4f}")
    cal = EMCalibrator(q30, model_type="DINA", max_iter=500, tol=1e-7)
    res = cal.fit(X30)
    print(f"  估计先验与真实先验相关系数: {np.corrcoef(res.attr_prior, true_prior)[0, 1]:.4f}")

    # ---- 出图 ----
    section("生成绝对拟合诊断报告图")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cdcat.calibration_plots import absolute_fit_report_figure

    png_a = OUT / "09_absolute_fit.png"
    fig = absolute_fit_report_figure(ana_ok)
    fig.savefig(png_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {png_a}（良好拟合）")

    png_b = OUT / "10_absolute_fit_misfit.png"
    fig = absolute_fit_report_figure(ana_bad)
    fig.savefig(png_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {png_b}（模型误设）")

    return [png_a, png_b]


def step10_toolkit(q, cal_result):
    """落地工具链：装配 · 在线标定 · 层级比较 · PPMC · DIF · HTML 报告。"""
    banner("步骤 10 / 10  落地工具链：题库装配 · 在线标定 · 层级比较 · PPMC · HTML 报告")

    from cdcat import (
        assemble_test_bank,
        build_cdm_report,
        calibrate_new_items,
        compare_hierarchies,
        dif_analysis,
        posterior_predictive_check,
    )

    model = DINA(q, slip=cal_result.slip, guess=cal_result.guess)
    rng = np.random.default_rng(2024)
    n = 600
    alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, n)]
    X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)
    print(f"模拟数据：{n} 名考生 × {q.n_items} 题")

    # ---- A. 题库装配 ----
    section("A. 题库装配：从 60 题里挑 18 题，期望判准率能保多少")
    asm = assemble_test_bank(q, model, n_items=18)
    print(f"  入选题数      : {asm.n_items}")
    print(f"  期望 PMR      : {asm.expected_pmr:.4f}")
    print(f"  期望 AMR      : {np.mean(asm.expected_amr):.4f}")
    print("  属性覆盖      : " + "、".join(
        f"{n_}={c}" for n_, c in zip(asm.attr_names, asm.attr_coverage)))
    print("  入选题目：" + "、".join(asm.selected_names))

    # ---- B. 新题在线标定 ----
    section("B. 新题在线标定：假设最后 8 题尚无参数，用已有模型反推")
    new_items = list(range(q.n_items - 8, q.n_items))
    rep = calibrate_new_items(model, X, new_items)
    truth_slip = np.asarray(cal_result.slip)[new_items]
    truth_guess = np.asarray(cal_result.guess)[new_items]
    est_slip = np.array([it.slip for it in rep.items])
    est_guess = np.array([it.guess for it in rep.items])
    print(f"  slip  MAE={np.abs(est_slip - truth_slip).mean():.4f}  "
          f"r={np.corrcoef(est_slip, truth_slip)[0, 1]:.4f}")
    print(f"  guess MAE={np.abs(est_guess - truth_guess).mean():.4f}  "
          f"r={np.corrcoef(est_guess, truth_guess)[0, 1]:.4f}")
    print(f"  可用题数：{len(rep.usable_items)} / {len(rep.items)}")

    # ---- C. 属性层级比较 ----
    section("C. 属性层级比较：数据支持「无层级」还是「线型先备」")
    hrep = compare_hierarchies(q, X, structures=("none", "linear"), max_iter=200)
    for c in hrep.candidates:
        if c.error:
            continue
        print(f"  {c.label:<10s} 合法模式 {c.n_legal:>3d}  参数 {c.n_params:>4d}  "
              f"BIC={c.bic:.1f}  ΔBIC={c.delta_bic:.1f}  权重={c.bic_weight:.3f}")
    print(f"  >> {hrep.verdict()}")

    # ---- D. 后验预测检验 ----
    section("D. 后验预测检验：复制数据能否再现观测的二阶结构")
    pp = posterior_predictive_check(
        model, X, n_replicates=100, refit=False, seed=7)
    bad = set(pp.misfit_stats)
    for name in pp.stat_names:
        print(f"  {name:<12s} 观测={pp.observed[name]:>8.4f}  "
              f"PPP={pp.ppp[name]:.3f}  {'失配' if name in bad else '可接受'}")
    print(f"  >> {pp.verdict()}")

    # ---- E. DIF 检验 ----
    section("E. DIF 检验：随机分两班，检查是否有题目偏向某一组")
    groups = ["A" if i % 2 == 0 else "B" for i in range(n)]
    dif = dif_analysis(model, X, groups, item_names=q.item_names)
    print(f"  显著题目数：{dif.n_significant} / {q.n_items}"
          f"（名义水平 {dif.alpha:.2f}，随机分组下期望约 "
          f"{int(round(dif.alpha * q.n_items))} 题）")
    print(f"  ETS 效应分级：A 级 {dif.n_A} / B 级 {dif.n_B} / C 级 {dif.n_C}"
          "（A=可忽略，B=中等，C=显著）")
    print("  说明：随机分组下显著项应接近名义水平；真实数据中显著项需人工复核。")

    # ---- F. 单文件 HTML 报告 ----
    section("F. 单文件 HTML 报告（图表内联，可离线归档）")
    ana = run_cdm_analysis(q, X, models=["DINA"], max_iter=200)
    html = build_cdm_report(
        ana, X,
        title="演示题库 · 认知诊断报告",
        subtitle=f"{n} 名考生 · {q.n_items} 题 × {q.n_attrs} 属性",
        author="cdcat demo",
    )
    path = OUT / "11_report.html"
    html.save(str(path))
    print(f"  已保存 {path}（{path.stat().st_size / 1024:.0f} KB）")
    return [path]


def main() -> None:
    print("\n" + "#" * 72)
    print("#  CD-CAT 认知诊断计算机自适应测验 · 完整演示")
    print("#" * 72)

    q, _ = step1_build_bank()
    cal_result = step2_calibrate(q)
    admin_rec = step3_administer(q, cal_result)
    compare_results = step4_compare(q, cal_result)
    sim_result, curve = step5_simulation(q, cal_result)
    step6_structure_analysis(q, cal_result, admin_rec, sim_result)
    files = step7_plot(q, cal_result, admin_rec, compare_results, sim_result, curve)
    files += step8_cdm_analysis(q)
    files += step9_absolute_fit(q)
    files += step10_toolkit(q, cal_result)

    banner("演示完成")
    print(f"共生成 {len(files)} 个文件，位于: {OUT}")
    for f in files:
        print(f"  - {f.name}")
    print("\n提示：运行 `streamlit run demo/app.py` 可打开完整交互式界面。")


if __name__ == "__main__":
    main()
