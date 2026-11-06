"""诊断扩展模块测试：Mantel-Haenszel DIF 与后验预测模型检验（PPMC）。

两个模块都属"检验类"功能，测试重点不是数值精度，而是
**判别力**（有问题的数据能被检出）与**特异性**（没问题的数据不乱报）。
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import DINA, DINO, QMatrix, diagnose_patterns
from cdcat.calibration import calibrate
from cdcat.diagnostics import (
    DIFItem,
    DIFReport,
    PPMCResult,
    dif_analysis,
    posterior_predictive_check,
)
from cdcat.diagnostics.ppmc import PPMC_STATISTICS


# ====================================================================== #
# 工具
# ====================================================================== #
def _q(n_attrs=4, n_items=20) -> QMatrix:
    m = np.zeros((n_items, n_attrs), dtype=int)
    per = n_items // 4
    m[0:per] = [1, 0, 0, 0]
    m[per:2 * per] = [0, 1, 0, 0]
    m[2 * per:3 * per] = [0, 0, 1, 1]
    m[3 * per:] = [1, 1, 0, 1]
    return QMatrix(m, attr_names=[f"A{i + 1}" for i in range(n_attrs)])


def _simulate(q, alpha, slip, guess, rng, model_cls=DINA):
    model = model_cls(q, slip=slip, guess=guess)
    w = 1 << np.arange(q.n_attrs - 1, -1, -1)
    p = model._get_prob()[(alpha * w).sum(axis=1)]
    return model, (rng.random(p.shape) < p).astype(np.float64)


# ====================================================================== #
# 数据夹具（模块级：每个场景只拟合一次，供多个用例复用）
# ====================================================================== #
@pytest.fixture(scope="module")
def planted():
    """焦点组在第 5 题 slip 被抬高（参照组不受影响）→ 该题应被检出。"""
    q = _q()
    rng = np.random.default_rng(2024)
    n = 1200
    group = np.array(["甲组"] * 700 + ["乙组"] * 500, dtype=object)
    alpha = rng.integers(0, 2, size=(n, q.n_attrs))

    slip = np.full(q.n_items, 0.10)
    guess = np.full(q.n_items, 0.12)
    _, x_ref = _simulate(q, alpha.copy(), slip, guess, rng)

    slip_f = slip.copy()
    slip_f[5] = 0.55
    _, x_foc = _simulate(q, alpha.copy(), slip_f, guess, rng)

    resp = np.where((group == "乙组")[:, None], x_foc, x_ref)
    model, _ = calibrate(q, resp, model_type="DINA")
    return q, model, resp, group


@pytest.fixture(scope="module")
def correct():
    """DINA 生成 → DINA 拟合：模型正确，不应报系统性失配。"""
    q = _q()
    rng = np.random.default_rng(31)
    alpha = rng.integers(0, 2, size=(700, q.n_attrs))
    _, resp = _simulate(q, alpha, np.full(q.n_items, 0.10),
                        np.full(q.n_items, 0.15), rng)
    model, _ = calibrate(q, resp, model_type="DINA")
    return q, model, resp


@pytest.fixture(scope="module")
def misspecified():
    """DINO 生成 → DINA 拟合：理想反应结构误设，局部依赖必然被检出。"""
    q = _q()
    rng = np.random.default_rng(32)
    alpha = rng.integers(0, 2, size=(700, q.n_attrs))
    _, resp = _simulate(q, alpha, np.full(q.n_items, 0.10),
                        np.full(q.n_items, 0.15), rng, model_cls=DINO)
    model, _ = calibrate(q, resp, model_type="DINA")
    return q, model, resp


# ====================================================================== #
# 1. MH-DIF
# ====================================================================== #
class TestDIF:
    def test_planted_item_detected(self, planted):
        q, model, resp, group = planted
        rep = dif_analysis(model, resp, group)
        assert isinstance(rep, DIFReport)
        assert rep.condition == "attribute_pattern"
        assert rep.ref_label == "甲组" and rep.focal_label == "乙组"

        flagged = [it.item_idx for it in rep.flagged(min_class="B")]
        assert 5 in flagged, f"植入 DIF 的第 5 题未检出，实际检出 {flagged}"

        item = rep.items[5]
        assert item.ets_class == "C"
        assert item.p_value < 0.001
        # 符号约定：参照组答对率更高 ⇒ Δ>0 ⇒ favors="参照组"
        assert item.p_ref > item.p_focal
        assert item.delta_ets > 0
        assert item.favors == "参照组"

    def test_low_false_positive_rate(self, planted):
        q, model, resp, group = planted
        rep = dif_analysis(model, resp, group)
        # 其余 19 题是干净的，C 级误报应当为零、B 级也只是零星
        assert rep.n_C == 1
        assert rep.n_B <= 2

    def test_stratification_uses_attribute_pattern(self, planted):
        q, model, resp, group = planted
        rep = dif_analysis(model, resp, group)
        # 分层数不应超过 2^K，也不应退化成"全样本一层"
        assert 1 <= rep.n_strata_used <= q.n_patterns

    def test_ref_group_explicit(self, planted):
        q, model, resp, group = planted
        rep = dif_analysis(model, resp, group, ref_group="乙组")
        assert rep.ref_label == "乙组"
        assert rep.focal_label == "甲组"
        # 视角反转后，同一道题应变成"有利于焦点组"的反向结果
        assert rep.items[5].delta_ets < 0

    def test_alpha_hat_forms(self, planted):
        q, model, resp, group = planted
        diag = diagnose_patterns(model, resp)
        by_index = dif_analysis(model, resp, group, alpha_hat=diag.pattern_indices)
        by_matrix = dif_analysis(model, resp, group, alpha_hat=diag.map_patterns)
        assert by_index.items[5].ets_class == "C"
        assert by_matrix.items[5].ets_class == "C"
        assert by_index.items[5].mh_chi2 == pytest.approx(
            by_matrix.items[5].mh_chi2, rel=1e-9
        )

    def test_ets_grading_boundaries(self):
        from cdcat.diagnostics.dif import _ets_class

        assert _ets_class(0.0, 0.9) == "A"
        assert _ets_class(0.99, 0.001) == "A"          # 未达 1.0 一律 A
        assert _ets_class(1.2, 0.001) == "B"           # 达 1.0 但不足 1.5
        assert _ets_class(1.6, 0.20) == "B"            # 达 1.5 但不显著 → B
        assert _ets_class(1.6, 0.001) == "C"
        assert _ets_class(-1.8, 0.001) == "C"          # 取绝对值

    def test_outputs(self, planted):
        q, model, resp, group = planted
        rep = dif_analysis(model, resp, group)
        df = rep.to_dataframe()
        assert len(df) == q.n_items
        assert {"item_idx", "delta_ets", "ets_class"} <= set(df.columns)

        d = rep.to_dict()
        assert d["condition"] == "attribute_pattern"
        assert len(d["items"]) == q.n_items

        assert "DIF" in rep.summary(top_n=5)
        assert len(rep.largest(top_n=3)) == 3
        assert isinstance(rep.items[0], DIFItem)

    def test_unknown_group_raises(self, planted):
        q, model, resp, group = planted
        with pytest.raises(ValueError):
            dif_analysis(model, resp, group, ref_group="丙组")


# ====================================================================== #
# 2. PPMC
# ====================================================================== #
class TestPPMC:
    def test_correct_model_not_flagged(self, correct):
        q, model, resp = correct
        res = posterior_predictive_check(model, resp, n_replicates=120, seed=7)
        assert isinstance(res, PPMCResult)
        assert res.used_refit is True
        assert res.alpha_source == "estimated"
        assert set(res.stat_names) == set(PPMC_STATISTICS)
        assert res.misfit_stats == [], f"正确模型误报：{res.ppp}"

    def test_misspecified_model_flagged(self, misspecified):
        q, model, resp = misspecified
        res = posterior_predictive_check(model, resp, n_replicates=120, seed=7)
        # 三个题目对统计量都应给出极小的 PPP（观测的局部依赖远超模型预期）
        for name in ("pair_srmsr", "q3_max", "joint_max"):
            assert res.ppp[name] < 0.05, f"{name} 未检出误设：PPP={res.ppp[name]:.3f}"
        assert "pair_srmsr" in res.misfit_stats

    def test_observed_beats_replicate_under_misfit(self, misspecified):
        q, model, resp = misspecified
        res = posterior_predictive_check(model, resp, n_replicates=120, seed=7)
        for i, name in enumerate(res.stat_names):
            if name in ("pair_srmsr", "q3_max", "joint_max"):
                assert res.observed[name] > res.replicated[:, i].mean()

    def test_statistics_subset(self, correct):
        q, model, resp = correct
        res = posterior_predictive_check(
            model, resp, statistics=("score_sd",), n_replicates=40, seed=1
        )
        assert res.stat_names == ["score_sd"]
        assert set(res.ppp) == {"score_sd"}

    def test_alpha_given_path(self, correct):
        q, model, resp = correct
        diag = diagnose_patterns(model, resp)
        res = posterior_predictive_check(
            model, resp, alpha_hat=diag.pattern_indices, n_replicates=40, seed=1
        )
        assert res.alpha_source == "given"
        assert res.used_refit is False
        # 属性已知时判别力会下降，但仍应给出合法的 PPP
        assert all(0.0 <= v <= 1.0 for v in res.ppp.values())

    def test_no_refit_runs(self, correct):
        q, model, resp = correct
        res = posterior_predictive_check(
            model, resp, n_replicates=40, seed=1, refit=False
        )
        assert res.used_refit is False
        assert set(res.ppp) == {"pair_srmsr", "q3_max", "joint_max", "score_sd"}

    def test_missing_values_tolerated(self, correct):
        q, model, resp = correct
        x = resp.copy()
        rng = np.random.default_rng(5)
        x[rng.random(x.shape) < 0.05] = np.nan
        res = posterior_predictive_check(model, x, n_replicates=120, seed=3)
        # 缺失不该被转成虚假的题目对关联
        assert res.ppp["pair_srmsr"] > 0.05
        assert res.ppp["joint_max"] > 0.05

    def test_deterministic_with_seed(self, correct):
        q, model, resp = correct
        a = posterior_predictive_check(model, resp, n_replicates=40, seed=99)
        b = posterior_predictive_check(model, resp, n_replicates=40, seed=99)
        assert a.ppp == b.ppp
        assert np.allclose(a.replicated, b.replicated)

    def test_outputs(self, misspecified):
        q, model, resp = misspecified
        res = posterior_predictive_check(model, resp, n_replicates=40, seed=2)
        df = res.to_dataframe()
        assert len(df) == len(res.stat_names)
        assert {"统计量", "观测值", "PPP", "判读"} <= set(df.columns)

        d = res.to_dict()
        assert d["n_items"] == q.n_items
        assert isinstance(d["verdict"], str)

        text = res.summary()
        assert ("PPMC" in text) or ("后验预测" in text)
        assert res.verdict()

    def test_validation(self, correct):
        q, model, resp = correct
        with pytest.raises(ValueError):
            posterior_predictive_check(model, resp[:, :-1], n_replicates=40)
        with pytest.raises(ValueError):
            posterior_predictive_check(model, resp, statistics=("不存在的量",))
        with pytest.raises(ValueError):
            posterior_predictive_check(model, resp, n_replicates=5)
        with pytest.raises(ValueError):
            posterior_predictive_check(model, resp, prior=np.ones(3))


# ====================================================================== #
# 3. 属性层级结构比较
# ====================================================================== #
def _hier_q() -> QMatrix:
    """18 题 / 4 属性的 Q 矩阵，单属性题与多属性题都有。"""
    m = np.zeros((18, 4), dtype=int)
    m[0:4] = [1, 0, 0, 0]
    m[4:8] = [0, 1, 0, 0]
    m[8:11] = [0, 0, 1, 0]
    m[11:14] = [0, 0, 0, 1]
    m[14:16] = [1, 1, 0, 0]
    m[16:18] = [1, 1, 1, 1]
    return QMatrix(m, attr_names=["A1", "A2", "A3", "A4"])


def _hier_data(hier, n=900, seed=7, slip=0.10, guess=0.15):
    """按给定层级的合法模式抽样并生成作答。"""
    from cdcat import AttributeHierarchy  # noqa: F401

    q = _hier_q()
    rng = np.random.default_rng(100)
    legal = q.attribute_patterns() if hier.is_empty else hier.legal_patterns()
    alpha = legal[rng.integers(0, len(legal), size=n)]
    model = DINA(q, slip=np.full(q.n_items, slip), guess=np.full(q.n_items, guess))
    w = 1 << np.arange(q.n_attrs - 1, -1, -1)
    p = model._get_prob()[(alpha * w).sum(axis=1)]
    return q, (rng.random(p.shape) < p).astype(np.float64)


class TestHierarchyComparison:
    def test_recovers_linear(self):
        from cdcat import AttributeHierarchy, compare_hierarchies

        q, X = _hier_data(AttributeHierarchy.linear(4))
        rep = compare_hierarchies(q, X)
        assert rep.best_label == "linear"
        assert rep.best.is_best
        assert rep.best.bic_weight > 0.5

    def test_none_wins_without_structure(self):
        from cdcat import AttributeHierarchy, compare_hierarchies

        q, X = _hier_data(AttributeHierarchy.none(4))
        rep = compare_hierarchies(q, X)
        assert rep.best_label == "none"
        assert "不支持任何层级约束" in rep.verdict()

    def test_restricted_prior_reduces_params(self):
        """层级约束必须真的降低自由参数数，且不被先验平滑复活。"""
        from cdcat import AttributeHierarchy, compare_hierarchies

        q, X = _hier_data(AttributeHierarchy.linear(4))
        rep = compare_hierarchies(q, X)
        lin = rep.get("linear")
        non = rep.get("none")
        assert lin.n_legal == 5 and non.n_legal == 16
        assert lin.n_params < non.n_params
        # 线性先验下非法模式应几乎不占质量
        prior = lin.result.attr_prior
        legal = np.asarray(lin.hierarchy.legal_mask(), dtype=bool)
        assert prior[~legal].sum() < 1e-3
        # 约束模型的似然不应高于无约束模型
        assert lin.loglik <= non.loglik + 1e-6

    def test_custom_candidates(self):
        from cdcat import AttributeHierarchy, compare_hierarchies

        q, X = _hier_data(AttributeHierarchy.linear(4))
        cands = {
            "A1先决": AttributeHierarchy.divergent(4),
            "末位综合": AttributeHierarchy.convergent(4),
            "空": AttributeHierarchy.none(4),
        }
        rep = compare_hierarchies(q, X, candidates=cands)
        assert {c.label for c in rep.candidates} == set(cands)
        assert rep.get("空").n_legal == 16

    def test_illegal_ratio_meaningful(self):
        from cdcat import AttributeHierarchy, compare_hierarchies

        # 收敛真值：发散约束与它冲突（发散要求 A1 掌握才谈得上后面，
        # 收敛数据里却存在"A1 未掌握而 A4 已掌握"的合法模式），故会大量判非法。
        # 注意线型模式集是收敛/发散合法集的子集，用线型数据测不出差异。
        q, X = _hier_data(AttributeHierarchy.convergent(4))
        rep = compare_hierarchies(q, X)
        assert rep.get("convergent").illegal_ratio < 0.05
        assert rep.get("divergent").illegal_ratio > 0.20
        assert rep.get("linear").illegal_ratio > 0.20

    def test_outputs(self):
        from cdcat import AttributeHierarchy, compare_hierarchies

        q, X = _hier_data(AttributeHierarchy.convergent(4))
        rep = compare_hierarchies(q, X)
        df = rep.to_dataframe()
        assert len(df) == len(rep.candidates)
        assert {"label", "n_legal", "bic", "delta_bic", "bic_weight"} <= set(df.columns)
        assert df["bic"].is_monotonic_increasing

        d = rep.to_dict()
        assert d["best"] == rep.best_label
        assert len(d["candidates"]) == len(rep.candidates)
        assert "属性层级结构比较" in rep.summary()

    def test_validation(self):
        from cdcat import AttributeHierarchy, compare_hierarchies

        q, X = _hier_data(AttributeHierarchy.linear(4), n=120)
        with pytest.raises(ValueError):
            compare_hierarchies(q, X[:, :-1])
        with pytest.raises(ValueError):
            compare_hierarchies(q, X, candidates={"只有一个": AttributeHierarchy.none(4)})
        with pytest.raises(ValueError):
            compare_hierarchies(q, X, structures=("不存在", "linear"))
        # 未指定 none 时自动补上，作为无约束基准
        rep = compare_hierarchies(q, X, structures=("linear", "convergent"))
        assert "none" in {c.label for c in rep.candidates}
        with pytest.raises(ValueError):
            compare_hierarchies(
                q, X, candidates={"错属性数": AttributeHierarchy.none(3),
                                  "对": AttributeHierarchy.none(4)}
            )
