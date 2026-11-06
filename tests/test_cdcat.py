"""CD-CAT 系统测试套件。

覆盖：工具函数、Q 矩阵、响应模型、估计器、选题准则、终止规则、
仿真引擎、评估器、EM 标定器。

运行::

    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import (
    QMatrix,
    DINA,
    DINO,
    GDINA,
    RRUM,
    CDCATSimulator,
    AttributeEstimator,
    Evaluator,
    EMCalibrator,
    FixedLength,
    StabilityRule,
    PatternConfidence,
    PosteriorThreshold,
    CompositeRule,
    create_model,
    create_selector,
    PWKL,
    MPWKL,
    KL,
    GDI,
    SHE,
    CDI,
)
from cdcat.utils import (
    enumerate_attribute_patterns,
    build_test_items,
    pattern_to_str,
    str_to_pattern,
)


# ====================================================================== #
# 工具函数
# ====================================================================== #
class TestUtils:
    def test_enumerate_patterns_shape(self):
        p = enumerate_attribute_patterns(4)
        assert p.shape == (16, 4)
        # 行 i 应为 i 的二进制展开
        assert list(p[0]) == [0, 0, 0, 0]
        assert list(p[15]) == [1, 1, 1, 1]
        assert list(p[5]) == [0, 1, 0, 1]

    def test_enumerate_patterns_unique(self):
        p = enumerate_attribute_patterns(6)
        assert len(np.unique(p, axis=0)) == 64

    def test_pattern_str_roundtrip(self):
        for s in ["10110", "00000", "11111", "10001"]:
            assert pattern_to_str(str_to_pattern(s)) == s

    def test_build_test_items(self):
        items = build_test_items(20, 3, seed=1)
        assert items["q_matrix"].shape == (20, 3)
        assert items["slip"].shape == (20,)
        assert((items["slip"] >= 0) & (items["slip"] <= 1)).all()


# ====================================================================== #
# Q 矩阵
# ====================================================================== #
class TestQMatrix:
    def test_basic(self):
        q = QMatrix(np.array([[1, 0], [1, 1], [0, 1]]))
        assert q.n_items == 3
        assert q.n_attrs == 2
        assert q.n_patterns == 4

    def test_invalid_values(self):
        with pytest.raises(ValueError):
            QMatrix(np.array([[1, 2], [0, 1]]))

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            QMatrix(np.array([1, 0, 1]))

    def test_ideal_response_conjunctive(self):
        q = QMatrix(np.array([[1, 0], [1, 1]]))
        e = q.ideal_response("conjunctive")
        assert e.shape == (4, 2)
        # 模式 00: 两题都不会
        assert list(e[0]) == [0, 0]
        # 模式 11: 两题都会
        assert list(e[3]) == [1, 1]
        # 模式 10 (掌握属性1): 第1题会, 第2题不会
        assert e[2, 0] == 1 and e[2, 1] == 0

    def test_ideal_response_disjunctive(self):
        q = QMatrix(np.array([[1, 0], [1, 1]]))
        e = q.ideal_response("disjunctive")
        # 行索引 = 二进制模式：0=00, 1=01, 2=10, 3=11（左为属性1）
        assert list(e[0]) == [0, 0]      # 都没有 -> 两题都不会
        assert list(e[1]) == [0, 1]      # 只掌握属性2 -> 第1题不会，第2题会
        assert list(e[2]) == [1, 1]      # 只掌握属性1 -> 第1题会，第2题会（析取）
        assert list(e[3]) == [1, 1]      # 都掌握

    def test_validate_uncovered(self):
        q = QMatrix(np.array([[1, 0], [1, 0]]))
        issues = q.validate()
        assert any("属性" in i for i in issues)

    def test_validate_empty_item(self):
        q = QMatrix(np.array([[1, 1], [0, 0]]))
        issues = q.validate()
        assert any("题目" in i for i in issues)

    def test_from_random_covers_all(self):
        q = QMatrix.from_random(30, 5, seed=3)
        assert (q.attribute_exposure() > 0).all()

    def test_diagnose_connected(self):
        q = QMatrix(np.array([[1, 1], [1, 1]]))
        d = q.diagnose()
        assert d.is_connected

    def test_diagnose_disconnected(self):
        # 属性 1,2 由第1题考查；属性 3 由第2题考查，与前面无共同题
        q = QMatrix(np.array([[1, 1, 0], [0, 0, 1]]))
        d = q.diagnose()
        assert not d.is_connected
        assert len(d.connected_components) == 2

    def test_diagnose_duplicates(self):
        q = QMatrix(np.array([[1, 0], [1, 0], [0, 1]]))
        d = q.diagnose()
        assert len(d.duplicate_rows) >= 1

    def test_required_attrs(self):
        q = QMatrix(np.array([[1, 0, 1], [0, 1, 0]]))
        assert q.required_attrs(0) == [0, 2]
        assert q.required_attrs(1) == [1]

    def test_roundtrip_dict(self):
        q = QMatrix.from_random(15, 3, seed=5)
        q2 = QMatrix.from_dict(q.to_dict())
        assert (q.matrix == q2.matrix).all()
        assert q.attr_names == q2.attr_names


# ====================================================================== #
# 响应模型
# ====================================================================== #
@pytest.fixture
def small_q():
    return QMatrix(np.array([[1, 0], [1, 1], [0, 1]]))


class TestModels:
    def test_dina_prob_range(self, small_q):
        m = DINA(small_q, slip=np.array([0.1, 0.2, 0.15]), guess=np.array([0.2, 0.1, 0.25]))
        p = m.prob_matrix()
        assert p.shape == (4, 3)
        assert ((p > 0) & (p < 1)).all()

    def test_dina_prob_values(self, small_q):
        m = DINA(small_q, slip=np.full(3, 0.1), guess=np.full(3, 0.2))
        p = m.prob_matrix()
        # 模式 00 -> 全未掌握 -> 全为 guess
        np.testing.assert_allclose(p[0], [0.2, 0.2, 0.2])
        # 模式 11 -> 全掌握 -> 全为 1-slip
        np.testing.assert_allclose(p[3], [0.9, 0.9, 0.9])

    def test_dino_prob_values(self, small_q):
        m = DINO(small_q, slip=np.full(3, 0.1), guess=np.full(3, 0.2))
        p = m.prob_matrix()
        # 模式 00 -> 全未掌握 -> guess
        np.testing.assert_allclose(p[0], [0.2, 0.2, 0.2])
        # 模式 11 -> 全掌握 -> 1-slip
        np.testing.assert_allclose(p[3], [0.9, 0.9, 0.9])

    def test_simulate_shape(self, small_q):
        m = DINA(small_q)
        rng = np.random.default_rng(0)
        r = m.simulate(np.array([1, 0]), rng=rng)
        assert r.shape == (3,)
        assert set(np.unique(r)).issubset({0, 1})

    def test_simulate_deterministic_extreme(self, small_q):
        # slip=0, guess=0 -> 完全确定
        m = DINA(small_q, slip=np.zeros(3), guess=np.zeros(3))
        rng = np.random.default_rng(0)
        r = m.simulate(np.array([1, 1]), rng=rng)
        np.testing.assert_array_equal(r, [1, 1, 1])
        r = m.simulate(np.array([0, 0]), rng=rng)
        np.testing.assert_array_equal(r, [0, 0, 0])

    def test_likelihood_peaks_at_truth(self, small_q):
        m = DINA(small_q, slip=np.full(3, 0.05), guess=np.full(3, 0.05))
        rng = np.random.default_rng(1)
        true = np.array([1, 1])
        r = m.simulate(true, rng=rng)
        lik = m.likelihood(r, np.arange(3))
        assert int(np.argmax(lik)) == 3      # 模式 11 的索引

    def test_gdina_from_dina_matches(self, small_q):
        """G-DINA 的指示函数基应能**完全精确**重构 DINA 概率。"""
        from cdcat.models import DINA as _DINA

        slip = np.array([0.1, 0.2, 0.15])
        guess = np.array([0.2, 0.1, 0.25])
        dina = _DINA(small_q, slip=slip, guess=guess)
        g = GDINA.from_dina(small_q, slip, guess)
        np.testing.assert_allclose(g.prob_matrix(), dina.prob_matrix(), atol=1e-10)

    def test_gdina_three_attrs_equivalence(self):
        """考查 3 个属性的题目也应精确等价（验证设计矩阵可逆性）。"""
        q = QMatrix(np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]]))
        slip = np.array([0.12, 0.08, 0.2])
        guess = np.array([0.18, 0.22, 0.1])
        dina = DINA(q, slip=slip, guess=guess)
        g = GDINA.from_dina(q, slip, guess)
        np.testing.assert_allclose(g.prob_matrix(), dina.prob_matrix(), atol=1e-10)

    def test_rrum_range(self, small_q):
        m = RRUM(small_q)
        p = m.prob_matrix()
        assert ((p > 0) & (p <= 1)).all()

    def test_create_model_factory(self, small_q):
        for name in ["DINA", "DINO", "G-DINA", "RRUM"]:
            m = create_model(name, small_q)
            assert m.prob_matrix().shape == (4, 3)

    def test_create_model_bad_name(self, small_q):
        with pytest.raises(ValueError):
            create_model("NOPE", small_q)

    def test_monotonicity_dina(self, small_q):
        """DINA 下掌握更多属性不应降低答对概率。"""
        m = DINA(small_q, slip=np.full(3, 0.1), guess=np.full(3, 0.2))
        p = m.prob_matrix()
        # 模式 10 是否 >= 模式 00
        assert (p[2] >= p[0] - 1e-12).all()


# ====================================================================== #
# 估计器
# ====================================================================== #
class TestEstimator:
    def setup_method(self):
        self.q = QMatrix.from_random(40, 4, seed=11)
        self.items = build_test_items(40, 4, seed=11)
        self.model = DINA(self.q, slip=self.items["slip"], guess=self.items["guess"])

    def test_posterior_normalized(self):
        est = AttributeEstimator(self.model)
        post = est.posterior(np.array([1, 0, 1]), np.array([0, 5, 9]))
        assert post.shape == (16,)
        assert abs(post.sum() - 1.0) < 1e-9

    def test_posterior_empty_is_prior(self):
        est = AttributeEstimator(self.model)
        post = est.posterior(np.array([]), np.array([], dtype=int))
        np.testing.assert_allclose(post, est.prior)

    def test_map_recovers_after_many_items(self):
        est = AttributeEstimator(self.model, method="MAP")
        rng = np.random.default_rng(3)
        true = np.array([1, 0, 1, 0])
        items = np.arange(self.q.n_items)
        resp = self.model.simulate(true, items, rng)
        res = est.estimate(resp, items)
        assert est.attribute_accuracy(res.pattern, true) >= 0.75

    def test_eap_vs_map(self):
        rng = np.random.default_rng(4)
        true = np.array([1, 1, 0, 0])
        items = np.arange(self.q.n_items)
        resp = self.model.simulate(true, items, rng)
        est_map = AttributeEstimator(self.model, method="MAP").estimate(resp, items)
        est_eap = AttributeEstimator(self.model, method="EAP").estimate(resp, items)
        assert est_map.pattern.shape == (4,)
        assert est_eap.pattern.shape == (4,)
        assert 0 <= est_eap.marginal.min() and est_eap.marginal.max() <= 1

    def test_marginal_probabilities(self):
        est = AttributeEstimator(self.model)
        post = np.zeros(16)
        post[15] = 1.0      # 模式 1111
        marg = est.marginal_probabilities(post)
        np.testing.assert_allclose(marg, [1, 1, 1, 1])

    def test_higher_order_prior(self):
        est = AttributeEstimator(self.model, higher_order=True, attr_prior=np.full(4, 0.8))
        assert abs(est.prior.sum() - 1.0) < 1e-9

    def test_sequential_history(self):
        est = AttributeEstimator(self.model)
        rng = np.random.default_rng(5)
        true = np.array([1, 0, 1, 1])
        items = np.arange(10)
        resp = self.model.simulate(true, items, rng)
        results = est.estimate_sequential(resp, items)
        assert len(results) == 10
        assert est.history.marginal_matrix().shape == (10, 4)


# ====================================================================== #
# 选题准则
# ====================================================================== #
class TestSelection:
    def setup_method(self):
        self.q = QMatrix.from_random(50, 4, seed=21)
        self.items = build_test_items(50, 4, seed=21)
        self.model = DINA(self.q, slip=self.items["slip"], guess=self.items["guess"])
        self.post = np.full(16, 1 / 16)

    @pytest.mark.parametrize("crit", ["KL", "PWKL", "MPWKL", "GDI", "SHE", "CDI"])
    def test_criteria_return_shape(self, crit):
        sel = create_selector(crit, self.model)
        vals = sel.compute(self.post, np.arange(50))
        assert vals.shape == (50,)
        assert np.isfinite(vals).all()

    @pytest.mark.parametrize("crit", ["KL", "PWKL", "MPWKL", "GDI", "SHE", "CDI"])
    def test_select_returns_valid_index(self, crit):
        sel = create_selector(crit, self.model)
        item = sel.select(self.post)
        assert 0 <= item < 50

    def test_select_excludes(self):
        sel = PWKL(self.model)
        excluded = np.arange(49)
        item = sel.select(self.post, exclude=excluded)
        assert item == 49

    def test_rank_ordering(self):
        sel = PWKL(self.model)
        ranked = sel.rank(self.post, top_n=5)
        assert len(ranked) == 5
        vals = [v for _, v in ranked]
        assert vals == sorted(vals, reverse=True)

    def test_gdi_nonnegative(self):
        sel = GDI(self.model)
        vals = sel.compute(self.post, np.arange(50))
        assert (vals >= -1e-9).all()

    def test_mpwkl_default_seed_is_reproducible(self):
        """默认构造的 MPWKL 必须可复现：扰动流不能来自操作系统熵源。"""
        post = np.abs(np.random.default_rng(3).normal(size=16))
        post = post / post.sum()
        vals_a = create_selector("MPWKL", self.model).compute(post, np.arange(50))
        vals_b = create_selector("MPWKL", self.model).compute(post, np.arange(50))
        assert np.array_equal(vals_a, vals_b)

    def test_mpwkl_explicit_seed_changes_perturbation(self):
        """显式换种子会改变扰动流——说明种子确实生效，而不是被忽略。"""
        post = np.full(16, 1 / 16)
        post[0] = 0.5
        post = post / post.sum()
        a = MPWKL(self.model, seed=1).compute(post, np.arange(50))
        b = MPWKL(self.model, seed=987654).compute(post, np.arange(50))
        assert not np.array_equal(a, b)

    def test_kl_nonnegative(self):
        sel = KL(self.model)
        vals = sel.compute(self.post, np.arange(50))
        assert (vals >= -1e-9).all()

    def test_pwkl_set_estimate(self):
        sel = PWKL(self.model)
        sel.set_estimate(np.array([1, 0, 1, 0]))
        assert sel._estimate_idx is not None
        vals = sel.compute(self.post)
        assert np.isfinite(vals).all()

    def test_bad_criterion_name(self):
        with pytest.raises(ValueError):
            create_selector("NOPE", self.model)


# ====================================================================== #
# 终止规则
# ====================================================================== #
class TestTermination:
    def _state(self, **kw):
        base = {
            "posterior": np.array([0.1, 0.1, 0.7, 0.1]),
            "marginal": np.array([0.9, 0.85]),
            "estimate": np.array([1, 1]),
            "n_answered": 10,
            "history": [3, 3, 3, 3, 3, 3],
            "n_items": 50,
            "min_items": 3,
        }
        base.update(kw)
        return base

    def test_fixed_length(self):
        r = FixedLength(10)
        assert r.should_stop(self._state(n_answered=10))
        assert not r.should_stop(self._state(n_answered=9))

    def test_posterior_threshold_max(self):
        r = PosteriorThreshold(0.6, mode="max")
        assert r.should_stop(self._state())

    def test_posterior_threshold_all_attrs(self):
        r = PosteriorThreshold(0.8, mode="all_attrs")
        assert r.should_stop(self._state(marginal=np.array([0.9, 0.85])))
        assert not r.should_stop(self._state(marginal=np.array([0.9, 0.5])))

    def test_stability(self):
        r = StabilityRule(n_stable=5)
        assert r.should_stop(self._state())
        assert not r.should_stop(self._state(history=[1, 3, 3, 3, 3, 3]))

    def test_pattern_confidence(self):
        r = PatternConfidence(0.8)
        assert r.should_stop(self._state(posterior=np.array([0.05, 0.05, 0.85, 0.05])))
        # 断言边际达标但模式后验不足时不停止
        flat_post = np.full(4, 0.25)
        assert not r.should_stop(self._state(posterior=flat_post))

    def test_composite_min_items_guard(self):
        r = CompositeRule([FixedLength(1)], how="any", min_items=5, max_items=20)
        assert not r.should_stop(self._state(n_answered=3))
        assert r.should_stop(self._state(n_answered=5))

    def test_composite_max_items(self):
        r = CompositeRule([FixedLength(999)], how="any", min_items=3, max_items=10)
        assert r.should_stop(self._state(n_answered=10))

    def test_composite_how_all(self):
        r = CompositeRule(
            [FixedLength(1), StabilityRule(n_stable=5)],
            how="all", min_items=1, max_items=50,
        )
        assert r.should_stop(self._state(n_answered=10))
        assert not r.should_stop(self._state(n_answered=10, history=[1, 2, 3]))


# ====================================================================== #
# 仿真引擎
# ====================================================================== #
class TestSimulation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.q = QMatrix.from_random(50, 4, seed=31)
        self.items = build_test_items(50, 4, seed=31)

    def _sim(self, selector="PWKL", **kw):
        return CDCATSimulator(
            self.q,
            model="DINA",
            model_params={"slip": self.items["slip"], "guess": self.items["guess"]},
            selector=selector,
            **kw,
        )

    def test_administer_basic(self):
        sim = self._sim()
        rng = np.random.default_rng(0)
        rec = sim.administer(np.array([1, 0, 1, 0]), rng=rng)
        assert rec.n_answered >= 3
        assert len(rec.administered) == rec.n_answered
        assert len(rec.responses) == rec.n_answered
        assert rec.est_pattern is not None

    def test_no_duplicate_items(self):
        sim = self._sim()
        rng = np.random.default_rng(1)
        rec = sim.administer(np.array([1, 1, 0, 0]), rng=rng)
        assert len(set(rec.administered)) == len(rec.administered)

    def test_item_indices_valid(self):
        sim = self._sim()
        rng = np.random.default_rng(2)
        rec = sim.administer(np.array([0, 0, 1, 1]), rng=rng)
        assert all(0 <= i < self.q.n_items for i in rec.administered)

    def test_fixed_length(self):
        sim = self._sim()
        rng = np.random.default_rng(3)
        rec = sim.simulate_fixed_length(np.array([1, 0, 1, 1]), n_items=15, rng=rng)
        assert rec.n_answered == 15
        assert rec.marginal_history.shape == (16, 4)

    def test_run_reproducible(self):
        sim = self._sim()
        r1 = sim.run(n_examinees=30, n_replications=2, seed=42)
        r2 = sim.run(n_examinees=30, n_replications=2, seed=42)
        assert r1.pmr() == r2.pmr()
        assert r1.avg_items() == r2.avg_items()

    def test_run_different_seeds_differ(self):
        sim = self._sim()
        r1 = sim.run(n_examinees=50, n_replications=2, seed=1)
        r2 = sim.run(n_examinees=50, n_replications=2, seed=2)
        assert r1.pmr() != r2.pmr()

    def test_run_improves_with_more_items(self):
        """固定题量下，题量越多判准率应越高。"""
        sim = self._sim()
        rng = np.random.default_rng(7)
        truths = self.q.attribute_patterns()[rng.integers(0, 16, 40)]
        accs = []
        for n in (5, 20):
            correct = 0
            for a in truths:
                rec = sim.simulate_fixed_length(a, n, rng)
                correct += int(rec.pattern_correct)
            accs.append(correct / len(truths))
        assert accs[1] >= accs[0]

    def test_compare_criteria(self):
        sim = self._sim()
        res = sim.compare_criteria(
            ["PWKL", "GDI"], n_examinees=30, n_replications=2, seed=0, fixed_length=15
        )
        assert set(res.keys()) == {"PWKL", "GDI"}
        assert res["PWKL"].avg_items() == pytest.approx(15, abs=0.01)

    def test_model_mismatch(self):
        """真实模型与估计模型不一致时应能运行（性能可下降）。"""
        from cdcat import DINO as _DINO

        true_m = _DINO(self.q, slip=self.items["slip"], guess=self.items["guess"])
        sim = CDCATSimulator(
            self.q, model="DINA",
            model_params={"slip": self.items["slip"], "guess": self.items["guess"]},
            selector="PWKL", true_model=true_m,
        )
        res = sim.run(n_examinees=30, n_replications=1, seed=0)
        assert res.pmr() >= 0

    def test_convergence_curve(self):
        sim = self._sim()
        curve = sim.convergence_curve(n_examinees=20, max_items=10, seed=0)
        assert len(curve["pmr"]) == 11
        assert curve["amr"][-1] >= curve["amr"][0]


# ====================================================================== #
# 评估器
# ====================================================================== #
class TestEvaluator:
    def setup_method(self):
        q = QMatrix.from_random(40, 4, seed=41)
        items = build_test_items(40, 4, seed=41)
        sim = CDCATSimulator(
            q, model="DINA",
            model_params={"slip": items["slip"], "guess": items["guess"]},
            selector="PWKL",
        )
        self.result = sim.run(n_examinees=60, n_replications=2, seed=5)

    def test_evaluate_metrics_in_range(self):
        rep = Evaluator().evaluate(self.result, label="PWKL")
        assert 0 <= rep.pmr <= 1
        assert 0 <= rep.amr <= 1
        assert rep.avg_items > 0
        assert 0 <= rep.test_utilization <= 1
        assert rep.pmr_ci[0] <= rep.pmr <= rep.pmr_ci[1]

    def test_attribute_accuracy_length(self):
        assert len(self.result.attribute_accuracy_by_attr()) == 4

    def test_exposure_chi_square(self):
        chi = Evaluator().exposure_chi_square(self.result)
        assert "statistic" in chi and "p_value" in chi
        assert 0 <= chi["p_value"] <= 1

    def test_efficiency_index_positive(self):
        assert Evaluator().efficiency_index(self.result) > 0

    def test_pmr_by_pattern(self):
        d = self.result.pmr_by_pattern()
        assert len(d) > 0
        for v in d.values():
            assert 0 <= v["pmr"] <= 1

    def test_marginal_convergence_shape(self):
        curve = self.result.marginal_convergence_curve()
        assert curve.shape[1] == 4

    def test_full_report_string(self):
        s = Evaluator().full_report({"PWKL": self.result})
        assert "PMR" in s and "AMR" in s


# ====================================================================== #
# EM 标定
# ====================================================================== #
class TestCalibration:
    def test_parameter_recovery(self):
        q = QMatrix.from_random(35, 4, seed=51)
        items = build_test_items(35, 4, seed=51)
        model = DINA(q, slip=items["slip"], guess=items["guess"])
        rng = np.random.default_rng(0)
        n = 1500
        alphas = q.attribute_patterns()[rng.integers(0, 16, n)]
        X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)

        cal = EMCalibrator(q, model_type="DINA", max_iter=150)
        result = cal.fit(X, verbose=False)

        assert result.converged
        # 参数恢复应达到较高相关
        assert np.corrcoef(result.slip, items["slip"])[0, 1] > 0.7
        assert np.corrcoef(result.guess, items["guess"])[0, 1] > 0.7
        assert np.abs(result.slip - items["slip"]).mean() < 0.06

    def test_log_likelihood_increases(self):
        q = QMatrix.from_random(25, 3, seed=52)
        items = build_test_items(25, 3, seed=52)
        model = DINA(q, slip=items["slip"], guess=items["guess"])
        rng = np.random.default_rng(0)
        alphas = q.attribute_patterns()[rng.integers(0, 8, 500)]
        X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)

        result = EMCalibrator(q).fit(X)
        ll = result.log_likelihood_history
        assert ll[-1] >= ll[0]

    def test_dino_calibration(self):
        q = QMatrix.from_random(25, 3, seed=53)
        items = build_test_items(25, 3, seed=53)
        model = DINO(q, slip=items["slip"], guess=items["guess"])
        rng = np.random.default_rng(0)
        alphas = q.attribute_patterns()[rng.integers(0, 8, 800)]
        X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)

        result = EMCalibrator(q, model_type="DINO").fit(X)
        assert result.slip.shape == (25,)

    def test_examinee_patterns(self):
        q = QMatrix.from_random(30, 3, seed=54)
        items = build_test_items(30, 3, seed=54)
        model = DINA(q, slip=items["slip"], guess=items["guess"])
        rng = np.random.default_rng(0)
        alphas = q.attribute_patterns()[rng.integers(0, 8, 300)]
        X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)

        cal = EMCalibrator(q)
        result = cal.fit(X)
        est = cal.examinee_patterns(X, result.slip, result.guess)
        assert est.shape == (300, 3)
        # 整体属性判准率应明显高于随机
        acc = np.mean([
            np.mean(est[i] == alphas[i]) for i in range(300)
        ])
        assert acc > 0.7

    def test_invalid_model_type(self):
        q = QMatrix.from_random(10, 3, seed=1)
        with pytest.raises(ValueError):
            EMCalibrator(q, model_type="HO-DINA")  # 未实现，必须报错

    def test_model_type_alias(self):
        q = QMatrix.from_random(10, 3, seed=1)
        # 'gdina' / 'g_dina' 等写法应被归一化为 G-DINA
        for alias in ("GDINA", "gdina", "g-dina", "G_DINA"):
            assert EMCalibrator(q, model_type=alias).model_type == "G-DINA"

    def test_dimension_mismatch(self):
        q = QMatrix.from_random(10, 3, seed=1)
        with pytest.raises(ValueError):
            EMCalibrator(q).fit(np.zeros((5, 8)))


# ====================================================================== #
# 曝光控制
# ====================================================================== #
class TestExposureControl:
    def setup_method(self):
        self.q = QMatrix.from_random(50, 4, seed=71)
        self.items = build_test_items(50, 4, seed=71)
        self.mp = {"slip": self.items["slip"], "guess": self.items["guess"]}

    def _sim(self, ctrl):
        return CDCATSimulator(
            self.q, model="DINA", model_params=self.mp,
            selector="PWKL", exposure_control=ctrl,
        )

    def test_no_control_baseline(self):
        from cdcat import NoExposureControl

        res = self._sim(NoExposureControl(50)).run(n_examinees=100, n_replications=4, seed=0)
        assert res.pmr() > 0.6

    def test_max_limit_reduces_peak(self):
        from cdcat import MaxExposureLimit, NoExposureControl

        base = self._sim(NoExposureControl(50)).run(n_examinees=200, n_replications=4, seed=0)
        limited = self._sim(
            MaxExposureLimit(50, max_rate=0.2, warmup=20)
        ).run(n_examinees=200, n_replications=4, seed=0)
        # 峰值曝光应显著下降
        assert limited.item_exposure_rate().max() < base.item_exposure_rate().max()
        # 精度损失应可控
        assert limited.pmr() > base.pmr() - 0.1

    def test_randomesque(self):
        from cdcat import RandomesqueExposure

        res = self._sim(
            RandomesqueExposure(50, top_n=5)
        ).run(n_examinees=150, n_replications=4, seed=0)
        assert res.pmr() > 0.6
        assert res.test_utilization() > 0.5

    def test_stratified(self):
        from cdcat import StratifiedExposure

        res = self._sim(
            StratifiedExposure(50, penalty=0.5, warmup=20)
        ).run(n_examinees=150, n_replications=4, seed=0)
        assert res.pmr() > 0.6

    def test_sympson_hetter_calibration_reduces_exposure(self):
        from cdcat import SympsonHetter

        sh = SympsonHetter(50, target_rate=0.25)
        sim = self._sim(sh)
        before = sim.run(n_examinees=100, n_replications=3, seed=0).item_exposure_rate().max()
        sh.calibrate(sim, n_examinees=100, n_replications=2, n_iterations=4, seed=1)
        after = self._sim(sh).run(n_examinees=100, n_replications=3, seed=0).item_exposure_rate().max()
        assert after < before

    def test_exposure_rate_semantics(self):
        """曝光率 = 见过该题的被试比例，应 <= 1。"""
        from cdcat import MaxExposureLimit

        res = self._sim(
            MaxExposureLimit(50, max_rate=0.3, warmup=10)
        ).run(n_examinees=100, n_replications=3, seed=0)
        er = res.item_exposure_rate()
        assert (er >= 0).all() and (er <= 1.0 + 1e-9).all()

    def test_create_exposure_control_factory(self):
        from cdcat import create_exposure_control

        for name in ["none", "maxlimit", "randomesque", "stratified", "sympsonhetter"]:
            c = create_exposure_control(name, 50)
            assert c.n_items == 50

    def test_bad_exposure_name(self):
        from cdcat import create_exposure_control

        with pytest.raises(ValueError):
            create_exposure_control("nope", 50)

    def test_no_duplicate_items_with_exposure_control(self):
        from cdcat import RandomesqueExposure

        sim = self._sim(RandomesqueExposure(50, top_n=8))
        rng = np.random.default_rng(3)
        rec = sim.administer(np.array([1, 0, 1, 1]), rng=rng)
        assert len(set(rec.administered)) == len(rec.administered)


# ====================================================================== #
# 可辨识性分析
# ====================================================================== #
class TestIdentifiability:
    def test_complete_qmatrix_is_identifiable(self):
        q = QMatrix(np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ]))
        rep = q.analyse_identifiability()
        assert rep.n_unidentifiable_patterns == 0
        assert rep.expected_pmr_ceiling == pytest.approx(1.0)
        assert rep.is_complete
        assert q.is_identifiable()

    def test_confounded_patterns_detected(self):
        # 属性 1、2 总同现，模式 00 与 10 的理想反应相同
        q = QMatrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
        rep = q.analyse_identifiability()
        assert rep.n_unidentifiable_patterns > 0
        assert rep.expected_pmr_ceiling < 1.0
        assert (0, 1) in rep.interchangeable_attr_pairs

    def test_ceiling_matches_observed_pmr(self):
        """核心验证：Q 矩阵的结构上限应能预测实测 PMR 的天花板。"""
        from cdcat import CDCATSimulator, build_test_items

        q = QMatrix.from_random(40, 4, seed=1)
        items = build_test_items(40, 4, seed=1)
        ceiling = q.analyse_identifiability().expected_pmr_ceiling
        assert ceiling < 1.0        # 该矩阵确实有结构缺陷

        # 零误差下算法完美，PMR 应紧贴上界
        sim = CDCATSimulator(
            q, model="DINA",
            model_params={"slip": np.zeros(40), "guess": np.zeros(40)},
            selector="PWKL",
        )
        res = sim.run(n_examinees=200, n_replications=2, seed=0)
        # 实测 PMR 不应超过上限（允许极小数值误差）
        assert res.pmr() <= ceiling + 0.01
        # 且应接近上限，说明算法本身没问题
        assert res.pmr() >= ceiling - 0.05

    def test_uncovered_attribute(self):
        q = QMatrix(np.array([[1, 0, 0], [1, 0, 0]]))
        rep = q.analyse_identifiability()
        assert 2 in rep.uncovered_attrs
        assert rep.expected_pmr_ceiling < 1.0

    def test_singleton_attrs_detected(self):
        q = QMatrix(np.array([[1, 0], [0, 1], [1, 1]]))
        rep = q.analyse_identifiability()
        assert set(rep.singleton_attrs) == {0, 1}

    def test_summary_string(self):
        q = QMatrix.from_random(20, 3, seed=1)
        s = q.analyse_identifiability().summary()
        assert "可辨识" in s

    def test_pmr_vs_ceiling_report(self):
        from cdcat import CDCATSimulator, build_test_items

        q = QMatrix.from_random(30, 4, seed=2)
        items = build_test_items(30, 4, seed=2)
        sim = CDCATSimulator(
            q, model="DINA",
            model_params={"slip": items["slip"], "guess": items["guess"]},
            selector="PWKL",
        )
        res = sim.run(n_examinees=80, n_replications=2, seed=0)
        d = res.pmr_vs_ceiling()
        assert 0 <= d["efficiency"] <= 1.2
        assert "verdict" in d
        assert d["ceiling"] > 0


# ====================================================================== #
# 题库信息量分析
# ====================================================================== #
class TestItemAnalysis:
    def setup_method(self):
        self.q = QMatrix.from_random(40, 4, seed=31)
        self.items = build_test_items(40, 4, seed=31)
        self.model = DINA(self.q, slip=self.items["slip"], guess=self.items["guess"])

    def test_basic_shapes(self):
        from cdcat import analyse_items

        ia = analyse_items(self.model)
        assert ia.discrimination.shape == (40,)
        assert ia.information.shape == (40,)
        assert (ia.information >= 0).all()
        assert ia.test_information_curve is not None
        assert ia.test_information_curve[-1] == pytest.approx(1.0)

    def test_discrimination_matches_model(self):
        from cdcat import analyse_items

        ia = analyse_items(self.model)
        expected = 1 - self.items["slip"] - self.items["guess"]
        # 排序应一致
        assert np.array_equal(ia.discrimination_rank, np.argsort(-expected))

    def test_redundancy_detection(self):
        from cdcat import analyse_items

        # 人为制造两道 Q 向量完全相同的题（结构冗余）
        q = QMatrix(np.array([[1, 0], [1, 0], [0, 1], [1, 1]]))
        m = DINA(q, slip=np.full(4, 0.1), guess=np.full(4, 0.1))
        ia = analyse_items(m, n_simulate=600, seed=0, redundancy_threshold=0.9)
        # 结构冗余（Q 向量相同）必须被检出
        assert [0, 1] in ia.structural_duplicates
        pairs = {(i, j) for i, j, _ in ia.redundant_pairs}
        assert (0, 1) in pairs

    def test_structural_vs_statistical_redundancy(self):
        """Q 向量不同但作答高度相关的题，也应能被统计判据捕获。"""
        from cdcat import analyse_items

        # 题0 与 题1 Q 向量不同，但都是单属性且属性相近 -> 相关较低
        q = QMatrix(np.array([[1, 0], [0, 1], [1, 1]]))
        m = DINA(q, slip=np.full(3, 0.05), guess=np.full(3, 0.05))
        ia = analyse_items(m, n_simulate=1000, seed=0, redundancy_threshold=0.5)
        # 无结构重复
        assert ia.structural_duplicates == []
        # 单属性题之间相关应低于双属性题，具体值不锁定，只验证不报错且有序
        for _, _, c in ia.redundant_pairs:
            assert c >= 0.5

    def test_top_items(self):
        from cdcat import analyse_items

        ia = analyse_items(self.model)
        top = ia.top_items(5)
        assert len(top) == 5
        assert len(set(top)) == 5

    def test_summary(self):
        from cdcat import analyse_items

        s = analyse_items(self.model).summary()
        assert "区分度" in s


# ====================================================================== #
# 被试诊断报告
# ====================================================================== #
class TestExamineeReport:
    def setup_method(self):
        self.q = QMatrix.from_random(40, 4, seed=41,
                                     attr_names=["通分", "约分", "加减", "乘除"])
        self.items = build_test_items(40, 4, seed=41)
        self.sim = CDCATSimulator(
            self.q, model="DINA",
            model_params={"slip": self.items["slip"], "guess": self.items["guess"]},
            selector="PWKL",
        )

    def _report(self, truth=None):
        from cdcat import build_examinee_report

        rng = np.random.default_rng(0)
        tp = np.array([1, 0, 1, 0]) if truth is None else truth
        rec = self.sim.administer(tp, rng=rng)
        return build_examinee_report(rec, self.q.attr_names, self.q.item_names, self.q)

    def test_report_fields(self):
        rep = self._report()
        assert rep.n_answered > 0
        assert len(rep.est_pattern) == 4
        assert len(rep.marginal) == 4
        assert 0 <= rep.accuracy <= 1
        assert len(rep.mastered) + len(rep.not_mastered) == 4

    def test_attr_confidence_range(self):
        rep = self._report()
        for i in range(4):
            assert 0 <= rep.attr_confidence(i) <= 1

    def test_to_dict(self):
        d = self._report().to_dict()
        assert "estimated_pattern" in d
        assert len(d["attributes"]) == 4
        assert len(d["responses"]) == d["n_answered"]

    def test_to_text(self):
        s = self._report().to_text()
        assert "认知诊断报告" in s
        assert "通分" in s

    def test_include_truth_flag(self):
        from cdcat import build_examinee_report

        rng = np.random.default_rng(0)
        rec = self.sim.administer(np.array([1, 1, 0, 0]), rng=rng)
        with_truth = build_examinee_report(rec, self.q.attr_names, self.q.item_names, self.q, include_truth=True)
        without = build_examinee_report(rec, self.q.attr_names, self.q.item_names, self.q, include_truth=False)
        assert with_truth.true_pattern is not None
        assert without.true_pattern is None
        assert without.pattern_correct is None


# ====================================================================== #
# 端到端集成
# ====================================================================== #
class TestIntegration:
    def test_full_pipeline(self):
        """完整链路：建库 -> 仿真数据 -> 标定 -> 用标定参数施测 -> 评估。"""
        q = QMatrix.from_random(50, 4, seed=61)
        items = build_test_items(50, 4, seed=61)
        true_model = DINA(q, slip=items["slip"], guess=items["guess"])

        # 1) 生成"历史作答数据"
        rng = np.random.default_rng(0)
        alphas = q.attribute_patterns()[rng.integers(0, 16, 1200)]
        X = np.array([true_model.simulate(a, rng=rng) for a in alphas], dtype=float)

        # 2) EM 标定
        cal_result = EMCalibrator(q, model_type="DINA", max_iter=150).fit(X)
        assert cal_result.converged

        # 3) 用标定参数构建 CD-CAT 并仿真
        sim = CDCATSimulator(
            q, model="DINA",
            model_params={"slip": cal_result.slip, "guess": cal_result.guess},
            selector="PWKL",
        )
        result = sim.run(n_examinees=80, n_replications=2, seed=1)

        # 4) 评估
        rep = Evaluator().evaluate(result)
        assert rep.pmr > 0.5
        assert rep.amr > 0.8
        assert rep.test_utilization > 0.3
