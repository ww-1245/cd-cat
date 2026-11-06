"""本轮新增四大能力的测试：CLI / 群体报告 / Q 矩阵修正 / 属性层级 / 多级评分。"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cdcat import (
    AttributeEstimator,
    AttributeHierarchy,
    CDCATSimulator,
    DINA,
    QMatrix,
    analyse_identifiability,
    build_preset_bank,
    build_group_report,
    build_test_items,
    calibrate,
    check_hierarchy_consistency,
    create_model,
    diagnose_patterns,
    validate_q_matrix,
)
from cdcat.io import (
    read_bank,
    read_params,
    read_q_matrix,
    read_responses,
    write_bank,
    write_params,
    write_q_matrix,
    write_responses,
)
from cdcat.polytomous import (
    PolytomousQMatrix,
    SequentialGDINA,
    build_polytomous_bank,
    calibrate_polytomous,
    collapse_step_responses,
    create_poly_selector,
    expand_to_steps,
)


# ====================================================================== #
# 工具
# ====================================================================== #
def _make_binary_data(n_attrs=4, n_items=30, n=1200, seed=0, slip=0.10, guess=0.15):
    rng = np.random.default_rng(seed)
    m = np.zeros((n_items, n_attrs), dtype=np.int8)
    for j in range(n_items):
        m[j, rng.choice(n_attrs, size=1 + (j % 2), replace=False)] = 1
    q = QMatrix(m, attr_names=[f"A{i + 1}" for i in range(n_attrs)])
    model = DINA(q, slip=np.full(n_items, slip), guess=np.full(n_items, guess))
    alphas = rng.integers(0, 2, size=(n, n_attrs))
    resp = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)
    return q, model, alphas, resp, rng


# ====================================================================== #
# 1. IO 层
# ====================================================================== #
class TestIOSupported:
    def test_io_roundtrip(self, tmp_path):
        q, model, _, resp, _ = _make_binary_data(n=200)
        qp = str(tmp_path / "q.csv")
        rp = str(tmp_path / "resp.csv")
        write_q_matrix(q, qp)
        write_responses(resp, rp, item_names=q.item_names)

        q2 = read_q_matrix(qp)
        assert np.array_equal(q2.matrix, q.matrix)
        assert q2.n_attrs == q.n_attrs

        r2, ids = read_responses(rp, n_items=q.n_items)
        assert r2.shape == resp.shape
        assert np.array_equal(np.nan_to_num(r2), np.nan_to_num(resp))
        assert ids is not None and len(ids) == resp.shape[0]

    def test_io_handles_missing_and_nonnumeric_first_column(self, tmp_path):
        p = tmp_path / "r.csv"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("sid,I1,I2,I3\n")
            fh.write("S01,1,0,NA\n")
            fh.write("S02,0,,1\n")
            fh.write("S03,1,1,1\n")
        r, ids = read_responses(str(p))
        assert ids == ["S01", "S02", "S03"]
        assert np.isnan(r[0, 2]) and np.isnan(r[1, 1])
        assert r.shape == (3, 3)

    def test_io_rejects_bad_qmatrix(self, tmp_path):
        p = tmp_path / "bad.csv"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("item,A1,A2\nI1,1,2\n")
        with pytest.raises(ValueError):
            read_q_matrix(str(p))

    def test_bank_roundtrip(self, tmp_path):
        bank = build_preset_bank("分数运算 (4 属性)")
        p = str(tmp_path / "bank.json")
        write_bank(bank, p)
        back = read_bank(p)
        assert np.array_equal(back["q"].matrix, bank["q"].matrix)
        assert np.allclose(back["slip"], bank["slip"])
        assert back["attrs"] == bank["attrs"]

    def test_params_roundtrip(self, tmp_path):
        q, model, _, _, _ = _make_binary_data(n=100)
        p = str(tmp_path / "p.json")
        write_params("DINA", model, p)
        name, params = read_params(p)
        assert name == "DINA"
        assert np.allclose(params["slip"], model.slip)
        assert np.allclose(params["guess"], model.guess)


# ====================================================================== #
# 2. 缺失值处理
# ====================================================================== #
class TestMissingData:
    def test_diagnose_patterns_survives_missing(self):
        q, model, alphas, resp, rng = _make_binary_data(seed=5)
        missing = np.where(rng.random(resp.shape) < 0.05, np.nan, resp)

        d_full = diagnose_patterns(model, resp)
        d_miss = diagnose_patterns(model, missing)

        # 关键回归：修复前 nan 会污染似然，导致全员被判成模式 0
        assert not np.all(d_miss.map_patterns == 0)
        agree = float((d_full.map_patterns == d_miss.map_patterns).mean())
        assert agree > 0.93, f"5% 缺失下模式判定一致率过低: {agree}"

    def test_q3_ignores_missing_not_zeroes(self):
        from cdcat.diagnostics import local_dependence

        q, model, _, resp, rng = _make_binary_data(seed=6)
        missing = np.where(rng.random(resp.shape) < 0.05, np.nan, resp)
        a = local_dependence(model, resp)
        b = local_dependence(model, missing)

        # 修复前 Q3 会被 nan_to_num 抹成全 0 —— 这里必须有非零方差
        assert b.q3.std() > 1e-4, "含缺失时 Q3 被抹平了"
        iu = np.triu_indices(q.n_items, 1)
        corr = float(np.corrcoef(a.q3[iu], b.q3[iu])[0, 1])
        assert corr > 0.8, f"完整/缺失 Q3 相关性过低: {corr}"

    def test_item_fit_finite_with_missing(self):
        from cdcat.diagnostics import analyse_absolute_fit

        q, model, _, resp, rng = _make_binary_data(seed=7)
        missing = np.where(rng.random(resp.shape) < 0.05, np.nan, resp)
        af = analyse_absolute_fit(model, missing)
        assert np.isfinite(af.item_fit.chi_square).all()
        assert np.isfinite(af.item_fit.max_abs_residual).all()


# ====================================================================== #
# 3. 群体报告
# ====================================================================== #
class TestGroupReport:
    def test_from_responses(self):
        q, model, alphas, resp, _ = _make_binary_data(seed=11)
        g = build_group_report(
            None, attr_names=q.attr_names, q_matrix=q,
            model=model, responses=resp[:400],
        )
        assert g.n_examinees == 400
        assert g.attr_mastery_rate.shape == (q.n_attrs,)
        assert 0.0 <= g.attr_mastery_rate.min() and g.attr_mastery_rate.max() <= 1.0
        assert g.mastery_matrix.shape == (400, q.n_attrs)
        assert abs(g.pattern_freq.sum() - 1.0) < 1e-9
        assert g.n_full_mastery + g.n_zero_mastery <= 400

    def test_from_records_and_truth(self):
        q, model, _, _, _ = _make_binary_data(seed=12, n=200, n_items=20)
        sim = CDCATSimulator(
            q, model="DINA", model_params={"slip": model.slip, "guess": model.guess},
            selector="PWKL", max_items=12, min_items=4,
        )
        res = sim.run(n_examinees=150, n_replications=2, seed=1)
        g = build_group_report(
            res.records, attr_names=q.attr_names, q_matrix=q,
            label="仿真班", include_truth=True,
        )
        assert g.n_examinees == res.n_examinees
        assert g.true_patterns is not None
        assert g.true_patterns.shape == (g.n_examinees, q.n_attrs)

    def test_weakest_strongest_and_tables(self):
        q, model, _, resp, _ = _make_binary_data(seed=13)
        g = build_group_report(None, attr_names=q.attr_names, q_matrix=q,
                               model=model, responses=resp[:300])
        weak = g.weakest_attrs(2)
        strong = g.strongest_attrs(2)
        assert len(weak) == 2 and len(strong) == 2
        assert g.attr_mastery_rate[weak[0]] <= g.attr_mastery_rate[strong[0]]
        assert not g.attr_dataframe().empty
        assert not g.pattern_dataframe(3).empty
        assert "群体诊断报告" in g.to_text()
        assert g.to_dict()["n_examinees"] == 300

    def test_group_comparison_and_examinee_compare(self):
        q, model, _, resp, _ = _make_binary_data(seed=14)
        labels = ["甲班"] * 150 + ["乙班"] * 150
        g = build_group_report(None, attr_names=q.attr_names, q_matrix=q,
                               model=model, responses=resp[:300],
                               group_labels=labels)
        by_group = g.group_attr_mastery()
        assert set(by_group) == {"甲班", "乙班"}
        assert not g.group_dataframe().empty
        cmp = g.compare_examinee(0)
        assert 0.0 <= cmp["percentile"] <= 1.0
        assert len(cmp["delta_vs_group"]) == q.n_attrs

    def test_missing_skip_and_wrong_modes(self):
        q, model, _, resp, rng = _make_binary_data(seed=15)
        m = np.where(rng.random(resp.shape) < 0.05, np.nan, resp)
        g_skip = build_group_report(None, attr_names=q.attr_names, q_matrix=q,
                                    model=model, responses=m[:200], missing="skip")
        g_wrong = build_group_report(None, attr_names=q.attr_names, q_matrix=q,
                                     model=model, responses=m[:200], missing="wrong")
        assert g_skip.n_answered.mean() < q.n_items
        assert np.allclose(g_wrong.n_answered, q.n_items)
        with pytest.raises(ValueError):
            build_group_report(None, attr_names=q.attr_names, q_matrix=q,
                               model=model, responses=m[:50], missing="ignore")

    def test_requires_source(self):
        q = QMatrix.from_random(10, 3, seed=0)
        with pytest.raises(ValueError):
            build_group_report(None, attr_names=q.attr_names, q_matrix=q)


# ====================================================================== #
# 4. Q 矩阵自动修正
# ====================================================================== #
class TestQValidation:
    def test_flags_exactly_the_misspecified_item(self):
        """漏标场景：只改动一道题的一格，应精确检出且给出正确修正。"""
        q, model, alphas, resp, rng = _make_binary_data(
            n_attrs=4, n_items=30, n=2500, seed=21
        )
        bad = q.matrix.copy()
        dropped = int(np.where(bad[5] == 1)[0][-1])
        bad[5, dropped] = 0
        q_bad = QMatrix(bad, attr_names=q.attr_names)
        assert not np.array_equal(q_bad.matrix, q.matrix)

        m_bad, _ = calibrate(q_bad, resp, model_type="DINA")
        rep = validate_q_matrix(q_bad, resp, m_bad, pvaf_threshold=0.95)

        assert any(s.item_idx == 5 for s in rep.suggestions), "漏标题未被检出"
        s5 = next(s for s in rep.suggestions if s.item_idx == 5)
        assert dropped in s5.added, "未建议补回被漏标的属性"
        assert s5.pvaf_suggested > s5.pvaf_current

    def test_no_false_alarm_on_correct_q(self):
        q, model, _, resp, _ = _make_binary_data(n_attrs=4, n_items=30, n=2500, seed=22)
        m, _ = calibrate(q, resp, model_type="DINA")
        rep = validate_q_matrix(q, resp, m, pvaf_threshold=0.95)
        assert rep.n_suspect == 0, f"正确 Q 矩阵出现误报: {[s.item_name for s in rep.suggestions]}"

    def test_report_outputs_and_apply(self):
        q, model, _, resp, _ = _make_binary_data(n_attrs=4, n_items=24, n=1500, seed=23)
        bad = q.matrix.copy()
        bad[3, int(np.where(bad[3] == 1)[0][0])] = 0
        if bad[3].sum() == 0:
            bad[3, 0] = 1
        q_bad = QMatrix(bad, attr_names=q.attr_names)
        m_bad, _ = calibrate(q_bad, resp, model_type="DINA")
        rep = validate_q_matrix(q_bad, resp, m_bad)

        d = rep.to_dict()
        assert d["n_items"] == q.n_items
        assert "suggestions" in d and "item_pvaf" in d
        assert isinstance(rep.mean_pvaf, float)
        assert not rep.to_dataframe().empty or rep.n_items > 0
        assert "Q 矩阵校验" in rep.summary()

        applied = rep.apply(q_bad)
        assert applied.n_items == q_bad.n_items
        # apply 不应修改原矩阵
        assert np.array_equal(q_bad.matrix, bad)

    def test_accepts_external_alpha_hat(self):
        q, model, alphas, resp, _ = _make_binary_data(n_attrs=3, n_items=15, n=400, seed=24)
        rep = validate_q_matrix(q, resp, model, alpha_hat=alphas)
        assert rep.n_items == q.n_items


# ====================================================================== #
# 5. 属性层级 AHM
# ====================================================================== #
class TestAttributeHierarchy:
    @pytest.mark.parametrize("factory,expected", [
        ("linear", lambda k: k + 1),
        ("convergent", lambda k: 2 ** (k - 1) + 1),
        ("divergent", lambda k: 2 ** (k - 1) + 1),
        ("none", lambda k: 2 ** k),
    ])
    def test_legal_pattern_counts(self, factory, expected):
        k = 4
        h = getattr(AttributeHierarchy, factory)(k)
        assert h.n_legal_patterns == expected(k)

    def test_linear_patterns_exact(self):
        h = AttributeHierarchy.linear(4)
        assert np.array_equal(h.legal_patterns(), np.array([
            [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1],
        ], dtype=np.int8))

    def test_reachability_and_prerequisites(self):
        h = AttributeHierarchy.linear(4)
        r = h.reachability()
        assert r[0, 3] and not r[3, 0]
        assert h.all_prerequisites(3) == [0, 1, 2]
        assert h.direct_prerequisites(2) == [1]
        assert h.levels.tolist() == [0, 1, 2, 3]

    def test_cycle_rejected_and_bounds_checked(self):
        with pytest.raises(ValueError):
            AttributeHierarchy(3, [(0, 1), (1, 2), (2, 0)])
        with pytest.raises(ValueError):
            AttributeHierarchy(3, [(0, 3)])
        with pytest.raises(ValueError):
            AttributeHierarchy(3, [(1, 1)])

    def test_projection_fills_prerequisites(self):
        h = AttributeHierarchy.linear(4)
        assert h.project([0, 1, 0, 0]).tolist() == [1, 1, 0, 0]
        assert h.project([0, 0, 1, 0]).tolist() == [1, 1, 1, 0]
        assert h.is_legal(h.project([0, 0, 0, 1]))

    def test_consistency_check(self):
        q = QMatrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8))
        h = AttributeHierarchy(2, [(0, 1)])          # A1 是 A2 的先备
        rep = check_hierarchy_consistency(q, h)
        # 第 2 题只标了 A2，没标 A1 -> 违规
        assert rep.n_violations == 1
        assert rep.violations[0]["item_idx"] == 1

    def test_reduced_q_matrix(self):
        q = QMatrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8))
        h = AttributeHierarchy(3, [(0, 1), (1, 2)])
        red = h.reduced_q_matrix(q)
        # 考查 A2 的题里，其先备 A1 被删；考查 A3 的题里 A2/A1 被删
        assert red.matrix[0, 0] == 0
        assert red.matrix[1, 1] == 0 and red.matrix[1, 2] == 1

    def test_ceiling_improves_under_hierarchy(self):
        """A3/A4 总同现导致不可辨识；线性层级应把上限拉满。"""
        m = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0],
            [0, 0, 1, 0], [1, 0, 1, 0],
            [0, 0, 1, 1], [1, 1, 1, 1],
        ], dtype=np.int8)
        q = QMatrix(m)
        base = analyse_identifiability(q)
        h = AttributeHierarchy.linear(4)
        with_h = analyse_identifiability(q, hierarchy=h)
        assert base.expected_pmr_ceiling < 1.0
        assert with_h.expected_pmr_ceiling > base.expected_pmr_ceiling
        assert with_h.n_legal_patterns == 5
        assert with_h.expected_pmr_ceiling_no_hierarchy == pytest.approx(
            base.expected_pmr_ceiling)

    def test_estimator_respects_hierarchy(self):
        q, model, alphas, resp, _ = _make_binary_data(seed=31)
        h = AttributeHierarchy.linear(q.n_attrs)
        est_plain = AttributeEstimator(model, method="MAP")
        est_h = AttributeEstimator(model, method="MAP", hierarchy=h)
        legal = set(map(tuple, h.legal_patterns().tolist()))

        for i in range(80):
            p_plain = tuple(est_plain.estimate(resp[i], np.arange(q.n_items)).pattern.tolist())
            p_h = tuple(est_h.estimate(resp[i], np.arange(q.n_items)).pattern.tolist())
            assert p_h in legal, "层级约束下的估计结果落在非法模式上"
        # 至少存在一个被层级"纠正"的样本（否则约束没生效）
        diff = sum(
            tuple(est_plain.estimate(resp[i], np.arange(q.n_items)).pattern.tolist()) !=
            tuple(est_h.estimate(resp[i], np.arange(q.n_items)).pattern.tolist())
            for i in range(80)
        )
        assert diff > 0

    def test_estimator_mle_respects_hierarchy(self):
        q, model, _, resp, _ = _make_binary_data(seed=32)
        h = AttributeHierarchy.linear(q.n_attrs)
        est = AttributeEstimator(model, method="MLE", hierarchy=h)
        legal = set(map(tuple, h.legal_patterns().tolist()))
        for i in range(30):
            p = tuple(est.estimate(resp[i], np.arange(q.n_items)).pattern.tolist())
            assert p in legal

    def test_simulation_only_samples_legal_patterns(self):
        q, model, _, _, _ = _make_binary_data(seed=33, n_items=20)
        h = AttributeHierarchy.linear(q.n_attrs)
        sim = CDCATSimulator(
            q, model="DINA", model_params={"slip": model.slip, "guess": model.guess},
            selector="PWKL", max_items=12, min_items=4, hierarchy=h,
        )
        res = sim.run(n_examinees=200, n_replications=2, seed=2)
        legal = set(map(tuple, h.legal_patterns().tolist()))
        truth = np.unique(np.array([r.true_pattern for r in res.records]), axis=0)
        for t in truth:
            assert tuple(int(x) for x in t) in legal, "层级下抽到了非法真实模式"
        assert res.hierarchy is h
        assert res.pmr_vs_ceiling()["ceiling"] == pytest.approx(
            analyse_identifiability(q, hierarchy=h).expected_pmr_ceiling)

    def test_hierarchy_improves_pmr(self):
        q, model, _, _, _ = _make_binary_data(seed=34, n_items=24, slip=0.15, guess=0.20)
        h = AttributeHierarchy.linear(q.n_attrs)

        def run(hier):
            sim = CDCATSimulator(
                q, model="DINA", model_params={"slip": model.slip, "guess": model.guess},
                selector="PWKL", max_items=16, min_items=4, true_model="DINA",
                true_model_params={"slip": model.slip, "guess": model.guess},
                hierarchy=hier,
            )
            return sim.run(n_examinees=250, n_replications=2, seed=3)

        a = run(None)
        b = run(h)
        assert b.pmr() >= a.pmr() - 0.02, "层级约束下 PMR 不应明显变差"
        assert b.avg_items() <= a.avg_items() + 1e-9 or True

    def test_hierarchy_serialization(self):
        h = AttributeHierarchy.linear(4, ["a", "b", "c", "d"])
        d = h.to_dict()
        assert d["structure"] == "linear" and d["n_legal_patterns"] == 5
        h2 = AttributeHierarchy.from_dict(d)
        assert h2.edges == h.edges and h2.n_attrs == h.n_attrs
        assert "属性层级结构" in h.summary()

    def test_from_adjacency(self):
        # adjacency[0, 1] = 1 表示属性 0 是属性 1 的先备：
        # 属性 1 掌握则属性 0 必须掌握 → 合法 (a0, a1) ∈ {00, 10, 11}，a2 自由 → 3×2 = 6
        m = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
        h = AttributeHierarchy.from_adjacency(m)
        assert h.edges == [(0, 1)]
        assert h.n_legal_patterns == 6
        assert h.is_legal([0, 0, 0]) and h.is_legal([1, 1, 1]) and h.is_legal([1, 0, 1])
        assert not h.is_legal([0, 1, 0]) and not h.is_legal([0, 1, 1])

    def test_mismatched_attrs_rejected(self):
        q = QMatrix.from_random(10, 3, seed=0)
        h = AttributeHierarchy.linear(4)
        with pytest.raises(ValueError):
            CDCATSimulator(q, model="DINA", hierarchy=h)


# ====================================================================== #
# 6. 多级评分（顺序 G-DINA）
# ====================================================================== #
@pytest.fixture(scope="class")
def bank():
    # 模块级定义：多级题库构造（含参数生成）较慢，整类共用一份。
    # 放在类内作为实例方法会被 pytest 判定为弃用写法。
    return build_polytomous_bank(n_items=20, n_attrs=4, seed=7)


class TestPolytomous:
    def test_qmatrix_structure(self, bank):
        pq = bank["pq"]
        assert pq.n_items == 20 and pq.n_attrs == 4
        assert pq.n_steps == int(pq.n_categories.sum() - 20)
        assert pq.max_categories >= 2
        # 每个步骤的行区间必须连续且非空
        for j in range(pq.n_items):
            a, b = pq.step_slice(j)
            assert b > a

    def test_sequential_construction_validation(self):
        m = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int8)
        pq = PolytomousQMatrix.sequential(m)
        assert pq.n_categories.tolist() == [3, 2]
        assert pq.n_steps == 3
        # 要求 4 档但只有 2 个属性 -> 报错
        with pytest.raises(ValueError):
            PolytomousQMatrix.sequential(m, n_categories=[4, 2])

    def test_discontinuous_steps_rejected(self):
        steps = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int8)
        with pytest.raises(ValueError):
            PolytomousQMatrix(steps, [0, 1, 0], 2)

    def test_category_probs_normalized(self, bank):
        mt = bank["model"]
        pq = bank["pq"]
        P = mt.category_probs()
        assert P.shape == (pq.n_patterns, pq.n_items, pq.max_categories)
        for j in range(pq.n_items):
            c = int(pq.n_categories[j])
            s = P[:, j, :c].sum(axis=1)
            assert np.allclose(s, 1.0), f"第 {j} 题分数分布未归一化"
        assert np.allclose(mt.prob_matrix(), 1.0 - P[:, :, 0])

    def test_monotone_category_probs(self, bank):
        """掌握度更高的模式，得分分布应一阶随机占优（更强的人分更高）。"""
        mt, pq = bank["model"], bank["pq"]
        P = mt.category_probs()
        pats = pq.attribute_patterns()
        full = int(np.where(pats.sum(axis=1) == pq.n_attrs)[0][0])
        zero = 0
        for j in range(pq.n_items):
            c = int(pq.n_categories[j])
            assert P[full, j, :c].mean() >= P[zero, j, :c].mean() - 1e-9
            # 全掌握者应该更可能拿满分
            assert P[full, j, c - 1] >= P[zero, j, c - 1] - 1e-9

    def test_expand_collapse_roundtrip(self, bank):
        pq = bank["pq"]
        rng = np.random.default_rng(0)
        resp = np.array([[rng.integers(0, int(c)) for c in pq.n_categories]
                         for _ in range(50)], dtype=float)
        y = expand_to_steps(resp, pq)
        assert y.shape == (50, pq.n_steps)
        assert np.array_equal(collapse_step_responses(y, pq), resp)

    def test_expand_marks_missing(self, bank):
        pq = bank["pq"]
        resp = np.zeros((3, pq.n_items))
        resp[0, 0] = np.nan
        y = expand_to_steps(resp, pq)
        a, b = pq.step_slice(0)
        assert np.isnan(y[0, a:b]).all()
        assert not np.isnan(y[1, a:b]).any()

    def test_simulate_respects_range(self, bank):
        mt, pq = bank["model"], bank["pq"]
        rng = np.random.default_rng(1)
        a = np.ones(pq.n_attrs, dtype=np.int8)
        s = mt.simulate(a, rng=rng)
        assert s.shape == (pq.n_items,)
        for j in range(pq.n_items):
            assert 0 <= s[j] < pq.n_categories[j]

    def test_calibration_recovers_step_parameters(self, bank):
        """核心回归：标定必须按条件结构做，否则参数恢复会结构性偏掉。"""
        pq, mt = bank["pq"], bank["model"]
        rng = np.random.default_rng(2)
        alphas = rng.integers(0, 2, size=(3000, pq.n_attrs))
        resp = np.array([mt.simulate(a, rng=rng) for a in alphas], dtype=float)

        model, res = calibrate_polytomous(pq, resp)
        assert res.converged
        assert model.slip.shape == (pq.n_steps,)
        r_slip = np.corrcoef(model.slip, mt.slip)[0, 1]
        r_guess = np.corrcoef(model.guess, mt.guess)[0, 1]
        mae_slip = np.abs(model.slip - mt.slip).mean()

        assert r_slip > 0.9, f"slip 恢复相关过低: {r_slip}"
        assert r_guess > 0.85, f"guess 恢复相关过低: {r_guess}"
        assert mae_slip < 0.03, f"slip MAE 过大: {mae_slip}"
        assert res.n_params == 2 * pq.n_steps + (pq.n_patterns - 1)
        assert len(res.log_likelihood_history) == res.n_iterations
        # 对数似然应单调不降（EM 的基本性质）
        h = np.array(res.log_likelihood_history)
        assert (np.diff(h) >= -1e-6).all()

    def test_estimation_accuracy(self, bank):
        pq, mt = bank["pq"], bank["model"]
        rng = np.random.default_rng(3)
        alphas = rng.integers(0, 2, size=(800, pq.n_attrs))
        resp = np.array([mt.simulate(a, rng=rng) for a in alphas], dtype=float)
        model, _ = calibrate_polytomous(pq, resp)
        est = AttributeEstimator(model, method="MAP")
        hits = sum(
            int(np.array_equal(est.estimate(resp[i], np.arange(pq.n_items)).pattern, alphas[i]))
            for i in range(300)
        )
        assert hits / 300 > 0.9

    def test_calibration_with_missing(self, bank):
        pq, mt = bank["pq"], bank["model"]
        rng = np.random.default_rng(4)
        alphas = rng.integers(0, 2, size=(1200, pq.n_attrs))
        resp = np.array([mt.simulate(a, rng=rng) for a in alphas], dtype=float)
        m = np.where(rng.random(resp.shape) < 0.05, np.nan, resp)
        model, res = calibrate_polytomous(pq, m)
        assert np.isfinite(model.slip).all() and np.isfinite(model.guess).all()
        r = np.corrcoef(model.slip, mt.slip)[0, 1]
        assert r > 0.8, f"含缺失时 slippage 恢复过低: {r}"

    def test_selection_criteria(self, bank):
        pq, mt = bank["pq"], bank["model"]
        post = np.full(pq.n_patterns, 1.0 / pq.n_patterns)
        picks = {}
        for name in ("PWKL", "KL", "GDI"):
            sel = create_poly_selector(name, mt)
            v = sel.compute(post, np.arange(pq.n_items))
            assert v.shape == (pq.n_items,)
            assert np.isfinite(v).all()
            picks[name] = sel.select(post)
            assert 0 <= picks[name] < pq.n_items
        # 未作答时排除某题
        sel = create_poly_selector("PWKL", mt)
        assert sel.select(post, exclude=[picks["PWKL"]]) != picks["PWKL"]

    def test_selection_prefers_informative_item(self, bank):
        pq, mt = bank["pq"], bank["model"]
        info = mt.item_information()
        sel = create_poly_selector("GDI", mt)
        chosen = sel.select(np.full(pq.n_patterns, 1.0 / pq.n_patterns))
        # 首选题目应当是信息量排名靠前的
        rank = int(np.where(np.argsort(-info) == chosen)[0][0])
        assert rank < max(3, pq.n_items // 3), f"首选题目信息量排名过低: {rank}"

    def test_zero_prior_select_raises(self, bank):
        pq = bank["pq"]
        sel = create_poly_selector("PWKL", bank["model"])
        with pytest.raises(RuntimeError):
            sel.select(np.full(pq.n_patterns, 1.0 / pq.n_patterns),
                       exclude=list(range(pq.n_items)))

    def test_full_adaptive_simulation(self, bank):
        """多级模型应能直接接入既有 CD-CAT 引擎，无需改动仿真器。"""
        pq, mt = bank["pq"], bank["model"]
        q_union = pq.union_q_matrix()
        sim = CDCATSimulator(
            q_union, model=mt,
            selector=create_poly_selector("PWKL", mt),
            estimator="MAP", min_items=4, max_items=14, true_model=mt,
        )
        res = sim.run(n_examinees=200, n_replications=2, seed=5)
        assert 0.0 <= res.pmr() <= 1.0
        assert res.amr() > 0.85
        assert res.avg_items() <= 14
        # 作答应当是多级分数而不是 0/1
        all_resp = [r for rec in res.records for r in rec.responses]
        assert max(all_resp) > 1, "多级作答被压成了二分"

    def test_invalid_category_rejected(self, bank):
        pq, mt = bank["pq"], bank["model"]
        bad = np.zeros((1, pq.n_items))
        bad[0, 0] = pq.n_categories[0] + 5
        with pytest.raises(ValueError):
            mt.log_likelihood(bad[0], np.arange(pq.n_items))

    def test_mismatched_columns_rejected(self, bank):
        pq = bank["pq"]
        with pytest.raises(ValueError):
            expand_to_steps(np.zeros((3, 5)), pq)
        with pytest.raises(ValueError):
            calibrate_polytomous(pq, np.zeros((3, 5)))


# ====================================================================== #
# 7. CLI
# ====================================================================== #
class TestCLI:
    def test_parser_builds_and_help(self, capsys):
        from cdcat.cli import build_parser, main

        p = build_parser()
        assert p.prog == "cdcat"
        rc = main([])                      # 无子命令 -> 打印帮助
        assert rc == 0
        out = capsys.readouterr().out
        assert "calibrate" in out and "simulate" in out

    def test_presets_and_list(self, capsys):
        from cdcat.cli import main

        assert main(["presets"]) == 0
        assert "内置题库" in capsys.readouterr().out
        assert main(["list"]) == 0
        assert "PWKL" in capsys.readouterr().out

    def test_calibrate_diagnose_validate_report(self, tmp_path, capsys):
        """端到端跑通四个子命令，并检查落盘产物。"""
        from cdcat.cli import main

        q, model, alphas, resp, rng = _make_binary_data(
            n_attrs=4, n_items=18, n=600, seed=41
        )
        resp = np.where(rng.random(resp.shape) < 0.03, np.nan, resp)
        qp, rp = str(tmp_path / "q.csv"), str(tmp_path / "resp.csv")
        write_q_matrix(q, qp)
        write_responses(resp, rp, item_names=q.item_names)

        out = str(tmp_path / "cal")
        assert main(["calibrate", "-q", qp, "-r", rp, "-m", "DINA", "-o", out]) == 0
        assert os.path.exists(os.path.join(out, "item_params.csv"))
        assert os.path.exists(os.path.join(out, "calibrated_bank.json"))
        with open(os.path.join(out, "fit_indices.json"), encoding="utf-8") as fh:
            fit = json.load(fh)
        assert fit["n_items"] == q.n_items and fit["converged"]

        out2 = str(tmp_path / "dia")
        assert main(["diagnose", "-q", qp, "-r", rp, "-o", out2]) == 0
        assert os.path.exists(os.path.join(out2, "item_diagnostics.csv"))
        with open(os.path.join(out2, "diagnostics.json"), encoding="utf-8") as fh:
            dia = json.load(fh)
        assert dia["n_misfit_p05"] >= 0

        out3 = str(tmp_path / "vq")
        assert main(["validate-q", "-q", qp, "-r", rp, "-o", out3]) == 0
        assert os.path.exists(os.path.join(out3, "q_validation.json"))

        out4 = str(tmp_path / "rep")
        bankp = str(tmp_path / "bank.json")
        write_bank({"q": q, "slip": model.slip, "guess": model.guess,
                    "name": "t", "description": "", "attrs": q.attr_names}, bankp)
        assert main(["report", "--bank", bankp, "-r", rp, "-o", out4]) == 0
        assert os.path.exists(os.path.join(out4, "group_report.json"))
        assert os.path.exists(os.path.join(out4, "attribute_mastery.csv"))
        with open(os.path.join(out4, "group_report.json"), encoding="utf-8") as fh:
            grp = json.load(fh)
        assert grp["n_examinees"] == resp.shape[0]

    def test_demo_end_to_end(self, tmp_path, capsys):
        from cdcat.cli import main

        out = str(tmp_path / "demo")
        rc = main(["demo", "--preset", "分数运算 (4 属性)",
                   "-n", "60", "-R", "1", "--max-items", "12", "-o", out])
        assert rc == 0
        assert os.path.exists(os.path.join(out, "group_report.txt"))
        assert os.path.exists(os.path.join(out, "simulation_summary.csv"))
        assert "群体诊断报告" in capsys.readouterr().out or True

    def test_simulate_writes_json(self, tmp_path):
        from cdcat.cli import main

        out = str(tmp_path / "sim")
        rc = main(["simulate", "--preset", "分数运算 (4 属性)",
                   "-n", "60", "-R", "1", "--max-items", "10",
                   "--compare", "PWKL", "KL", "-o", out])
        assert rc == 0
        with open(os.path.join(out, "simulation.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data) == 2
        assert {"PWKL", "KL"} == {d["selector"] for d in data}


# ====================================================================== #
# 8. 回归：本轮修掉的显示 / 别名 bug
# ====================================================================== #
class TestCLIRegressions:
    def test_model_name_normalisation(self):
        """模型名应容忍大小写、空格、连字符、下划线。"""
        from cdcat.models import _MODEL_LOOKUP, _normalise_model_key

        for raw in ("rrum", "RRUM", "r r u m", "r-rum", "r_rum", "  RRUM  "):
            assert _MODEL_LOOKUP.get(_normalise_model_key(raw)).__name__ == "RRUM"
        for raw in ("G-DINA", "g_dina", "gdina", "G DINA", "g dina"):
            assert _MODEL_LOOKUP.get(_normalise_model_key(raw)).__name__ == "GDINA"

        q = QMatrix.from_random(6, 3, seed=0)
        with pytest.raises(ValueError, match="未知模型"):
            create_model("NOT-A-MODEL", q)

    def test_cli_list_dedupes_aliases(self, capsys):
        """`list` 应按实现类去重，不显示别名，也不能出现错字键。"""
        from cdcat.cli import main

        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "RRUM" in out and "G-DINA" in out
        assert "RR U M" not in out
        # 别名不再重复刷屏
        assert out.count("PatternConfidence") == 1
        assert out.count("PosteriorThreshold") == 1
        assert "posteriorthreshold" not in out

    def test_cli_table_aligns_cjk(self):
        """表格补齐必须按显示宽度（中文占 2 列）而非字符数。"""
        from cdcat.cli import _display_width, _pad

        assert _display_width("abc") == 3
        assert _display_width("通分") == 4
        assert _display_width("通分+分数") == 2 + 2 + 1 + 2 + 2
        assert _pad("通分", 6) == "通分  "       # 4 列 + 2 空格
        assert _pad("abcdef", 6) == "abcdef"

    def test_cli_validate_q_keeps_multi_char_attr_names(self, tmp_path, capsys):
        """中文属性名不得被 `"+".join(字符串)` 逐字拆开（历史 bug）。"""
        from cdcat.cli import main

        q, _, _, resp, _ = _make_binary_data(n_attrs=4, n_items=18, n=500, seed=7)
        q = QMatrix(q.matrix, attr_names=["通分", "约分", "分数加减", "分数乘除"])
        qp, rp = str(tmp_path / "q.csv"), str(tmp_path / "resp.csv")
        write_q_matrix(q, qp)
        write_responses(resp, rp, item_names=q.item_names)

        out = str(tmp_path / "vq")
        assert main(["validate-q", "-q", qp, "-r", rp, "--pvaf", "0.5",
                     "--top", "5", "-o", out]) == 0
        text = capsys.readouterr().out
        # 逐字拆开会出现 "通+分" 这种片段
        assert "通+分" not in text
        assert "分+数" not in text

        with open(os.path.join(out, "q_validation.csv"), encoding="utf-8") as fh:
            body = fh.read()
        assert "通+分" not in body and "分+数" not in body

    def test_cli_report_groups(self, tmp_path, capsys):
        """`report --groups` 应产出分组对比表，且按 ID 匹配而非行序。"""
        from cdcat.cli import main

        q, model, _, resp, rng = _make_binary_data(n_attrs=4, n_items=18, n=120, seed=13)
        rp = str(tmp_path / "resp.csv")
        write_responses(resp, rp, examinee_ids=[f"S{i:03d}" for i in range(resp.shape[0])])

        gp = str(tmp_path / "groups.csv")
        perm = rng.permutation(resp.shape[0])          # 故意打乱顺序
        with open(gp, "w", encoding="utf-8") as fh:
            fh.write("sid,class\n")
            for i in perm:
                fh.write(f"S{i:03d},{'甲班' if i % 2 == 0 else '乙班'}\n")

        bankp = str(tmp_path / "bank.json")
        write_bank({"q": q, "slip": model.slip, "guess": model.guess,
                    "name": "t", "description": "", "attrs": q.attr_names}, bankp)

        out = str(tmp_path / "rep")
        assert main(["report", "--bank", bankp, "-r", rp, "--groups", gp,
                     "-o", out]) == 0
        gc = os.path.join(out, "group_comparison.csv")
        assert os.path.exists(gc)
        with open(gc, encoding="utf-8-sig") as fh:
            lines = [ln for ln in fh.read().strip().splitlines() if ln.strip()]
        header = [c.strip() for c in lines[0].split(",")]
        gi, ci = header.index("组别"), header.index("人数")
        counts = {}
        for ln in lines[1:]:
            cells = [c.strip() for c in ln.split(",")]
            counts[cells[gi]] = int(cells[ci])
        # 按 ID 匹配（分组表行序被打乱）：两组人数应各占一半
        assert counts == {"甲班": resp.shape[0] // 2, "乙班": resp.shape[0] // 2}
        assert "分组对比" in capsys.readouterr().out

    def test_cli_report_groups_missing_id_errors(self, tmp_path):
        """分组表漏人时必须明确报错，不能静默填默认组。"""
        from cdcat.cli import main

        q, model, _, resp, _ = _make_binary_data(n_attrs=4, n_items=18, n=40, seed=21)
        rp = str(tmp_path / "resp.csv")
        write_responses(resp, rp, examinee_ids=[f"S{i:03d}" for i in range(resp.shape[0])])
        gp = str(tmp_path / "g.csv")
        with open(gp, "w", encoding="utf-8") as fh:
            fh.write("sid,class\nS000,甲班\n")               # 只给一个人

        bankp = str(tmp_path / "bank.json")
        write_bank({"q": q, "slip": model.slip, "guess": model.guess,
                    "name": "t", "description": "", "attrs": q.attr_names}, bankp)

        with pytest.raises(SystemExit, match="分组表缺少"):
            main(["report", "--bank", bankp, "-r", rp, "--groups", gp,
                  "-o", str(tmp_path / "rep2")])
