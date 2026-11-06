"""非参数分类与 MIV 选题准则的测试。

覆盖三块：

1. 距离矩阵——与逐题暴力计算逐元素对齐（这是整个模块的地基，必须先钉死）；
2. 分类器——形状、伪后验归一化、缺失值、层级约束、与真实模式比对的精度；
3. 流水线 / CLI / MIV 的接入，确保"能用"而不只是"能算"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdcat import DINA, QMatrix, build_test_items  # noqa: E402
from cdcat.cli import main  # noqa: E402
from cdcat.estimation.nonparametric import (  # noqa: E402
    NonparametricResult,
    classify_nonparametric,
    distance_matrix,
    item_weights,
)
from cdcat.io import write_q_matrix, write_responses  # noqa: E402
from cdcat.pipeline import run_cdm_analysis  # noqa: E402


# ---------------------------------------------------------------------- #
def _data(n_items: int = 20, n: int = 300, seed: int = 3, n_attrs: int = 4):
    """造一套已知真值的 DINA 数据。"""
    q = QMatrix.from_random(
        n_items=n_items, n_attrs=n_attrs,
        min_attrs_per_item=1, max_attrs_per_item=2, seed=seed,
        attr_names=[f"A{i + 1}" for i in range(n_attrs)],
    )
    items = build_test_items(
        n_items, n_attrs, min_attrs_per_item=1, max_attrs_per_item=2, seed=seed + 1
    )
    model = DINA(q, slip=items["slip"], guess=items["guess"])
    rng = np.random.default_rng(seed)
    alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, n)]
    resp = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)
    return q, model, alphas, resp


# ====================================================================== #
# 权重
# ====================================================================== #
class TestItemWeights:
    def test_hamming_is_all_ones(self):
        _, _, _, resp = _data()
        w = item_weights(resp, "hamming")
        assert w.shape == (resp.shape[1],)
        assert np.allclose(w, 1.0)

    def test_gdd_normalised_to_unit_mean(self):
        _, _, _, resp = _data()
        w = item_weights(resp, "gdd")
        assert np.all(w > 0)
        assert w.mean() == pytest.approx(1.0)
        # 权重只取决于通过率，极端题会被截断而不是发散
        assert np.isfinite(w).all()

    def test_unknown_method_raises(self):
        _, _, _, resp = _data()
        with pytest.raises(ValueError, match="未知非参数方法"):
            item_weights(resp, "nope")


# ====================================================================== #
# 距离矩阵
# ====================================================================== #
class TestDistanceMatrix:
    @pytest.mark.parametrize("method", ["hamming", "gdd"])
    def test_matches_bruteforce(self, method):
        q, _, _, resp = _data(n=60)
        d = distance_matrix(q, resp, method=method)
        w = item_weights(resp, method)
        e = q.ideal_response()
        brute = np.abs(resp[:, None, :] - e[None, :, :]) @ w  # (n, S)
        assert d.shape == (resp.shape[0], q.n_patterns)
        assert np.allclose(d, brute)

    def test_missing_items_contribute_nothing(self):
        """未作答的题对距离的贡献必须恰为 0（不是按 0 分算）。"""
        q, _, _, resp = _data(n=40)
        j = 3
        x_nan = resp.copy()
        x_nan[0, j] = np.nan
        w = np.ones(q.n_items)
        d_nan = distance_matrix(q, x_nan, method="hamming", weights=w)

        e = q.ideal_response()
        full = np.abs(resp[0] - e) @ w                  # (S,) 含第 j 题
        drop = np.abs(resp[0, j] - e[:, j]) * w[j]      # 第 j 题的贡献
        assert np.allclose(d_nan[0], full - drop)
        # 该题贡献确实非零，否则这条测试是空转
        assert np.any(drop > 0)

    def test_hierarchy_marks_illegal_patterns_infinite(self):
        from cdcat.qmatrix import AttributeHierarchy

        q, _, _, resp = _data(n=30)
        h = AttributeHierarchy.linear(q.n_attrs)
        d = distance_matrix(q, resp, method="hamming", hierarchy=h)
        illegal_per_row = np.isinf(d).sum(axis=1)
        assert np.all(illegal_per_row == q.n_patterns - h.n_legal_patterns)
        # 合法模式上的距离仍是有限值
        assert np.isfinite(d[:, h.legal_mask()]).all()

    def test_weight_length_mismatch_raises(self):
        q, _, _, resp = _data(n=10)
        with pytest.raises(ValueError, match="权重长度"):
            distance_matrix(q, resp, weights=np.ones(q.n_items + 1))

    def test_response_width_mismatch_raises(self):
        q, _, _, resp = _data(n=10)
        with pytest.raises(ValueError, match="列数"):
            distance_matrix(q, resp[:, :-1])


# ====================================================================== #
# 分类器
# ====================================================================== #
class TestClassifier:
    def test_shapes_and_posterior_normalised(self):
        q, _, _, resp = _data()
        res = classify_nonparametric(q, resp, method="gdd", attr_names=q.attr_names)
        assert isinstance(res, NonparametricResult)
        assert res.patterns.shape == (resp.shape[0], q.n_attrs)
        assert res.posterior.shape == (resp.shape[0], q.n_patterns)
        assert np.allclose(res.posterior.sum(axis=1), 1.0)
        assert np.all((res.confidence > 0) & (res.confidence <= 1.0))
        assert np.all(res.pattern_indices >= 0)
        assert np.all(res.pattern_indices < q.n_patterns)
        assert res.attr_names == list(q.attr_names)

    @pytest.mark.parametrize("method", ["hamming", "gdd"])
    def test_accuracy_beats_chance(self, method):
        q, _, alphas, resp = _data(n=400)
        res = classify_nonparametric(q, resp, method=method)
        acc = res.accuracy(alphas)
        # 1/2^K 是随机猜测水平；真实数据上距离判别应显著更高
        assert acc["pmr"] > 1.5 / q.n_patterns
        assert acc["amr"] > 0.8
        assert len(acc["per_attr"]) == q.n_attrs

    def test_accuracy_shape_mismatch_raises(self):
        q, _, alphas, resp = _data(n=20)
        res = classify_nonparametric(q, resp)
        with pytest.raises(ValueError, match="不符"):
            res.accuracy(alphas[:, :-1])

    def test_hierarchy_constrains_predicted_patterns(self):
        from cdcat.qmatrix import AttributeHierarchy

        q, _, _, resp = _data(n=60)
        h = AttributeHierarchy.linear(q.n_attrs)
        res = classify_nonparametric(q, resp, method="hamming", hierarchy=h)
        for pattern in res.patterns:
            assert h.is_legal(pattern)
        assert "linear" in res.hierarchy

    def test_all_missing_row_does_not_crash(self):
        q, _, _, resp = _data(n=20)
        resp = resp.copy()
        resp[0, :] = np.nan
        res = classify_nonparametric(q, resp)
        # 一题未答 -> 所有距离相等 -> 后验退化为均匀，但不报错
        assert np.allclose(res.posterior[0], 1.0 / q.n_patterns)
        assert res.patterns.shape[0] == 20

    def test_temperature_controls_sharpness(self):
        q, _, _, resp = _data(n=50)
        sharp = classify_nonparametric(q, resp, temperature=0.2)
        soft = classify_nonparametric(q, resp, temperature=5.0)
        assert sharp.mean_confidence() > soft.mean_confidence()

    def test_invalid_arguments_raise(self):
        q, _, _, resp = _data(n=5)
        with pytest.raises(ValueError, match="未知非参数方法"):
            classify_nonparametric(q, resp, method="magic")
        with pytest.raises(ValueError, match="temperature"):
            classify_nonparametric(q, resp, temperature=0)

    def test_distance_gap_non_negative(self):
        q, _, _, resp = _data(n=40)
        res = classify_nonparametric(q, resp)
        gaps = res.distance_gap()
        assert gaps.shape == (resp.shape[0],)
        assert np.all(gaps >= -1e-12)

    def test_prior_only_breaks_ties(self):
        q, _, _, resp = _data(n=30)
        prior = np.full(q.n_patterns, 1.0 / q.n_patterns)
        a = classify_nonparametric(q, resp, method="hamming")
        b = classify_nonparametric(q, resp, method="hamming", prior=prior)
        assert np.array_equal(a.patterns, b.patterns)


# ====================================================================== #
# 与诊断对象的桥接
# ====================================================================== #
class TestDiagnosisBridge:
    def test_to_pattern_diagnosis_is_consistent(self):
        from cdcat.diagnostics import PatternDiagnosis

        q, _, _, resp = _data(n=80)
        res = classify_nonparametric(q, resp)
        diag = res.to_pattern_diagnosis()
        assert isinstance(diag, PatternDiagnosis)
        assert diag.n_examinees == res.n_examinees
        assert diag.n_attrs == q.n_attrs
        assert np.array_equal(diag.map_patterns, res.patterns)
        assert np.allclose(diag.eap_marginal, res.marginal)
        assert np.allclose(diag.map_confidence, res.confidence)
        # 现有组件可直接复用
        assert diag.mastery_proportion().shape == (q.n_attrs,)
        assert len(diag.pattern_distribution()) >= 1


# ====================================================================== #
# MIV 选题准则
# ====================================================================== #
class TestMIV:
    def test_values_bounded_and_finite(self):
        from cdcat.selection import create_selector

        q, model, _, _ = _data(n_items=15, n=10)
        sel = create_selector("MIV", model)
        post = np.full(q.n_patterns, 1.0 / q.n_patterns)
        vals = sel.compute(post)
        assert vals.shape == (q.n_items,)
        assert np.all(vals >= 0) and np.all(vals <= 0.25 + 1e-12)

    def test_needs_no_item_parameters(self):
        """同一 Q 矩阵下，题目参数怎么变，MIV 的值都不该变。"""
        from cdcat.selection import MIV

        q, _, _, _ = _data(n_items=12, n=10)
        m1 = DINA(q, slip=np.full(12, 0.05), guess=np.full(12, 0.05))
        m2 = DINA(q, slip=np.full(12, 0.30), guess=np.full(12, 0.30))
        post = np.linspace(0.01, 0.99, q.n_patterns)
        post = post / post.sum()
        assert np.allclose(MIV(m1).compute(post), MIV(m2).compute(post))

    def test_respects_candidate_subset(self):
        from cdcat.selection import create_selector

        q, model, _, _ = _data(n_items=15, n=10)
        sel = create_selector("MIV", model)
        post = np.full(q.n_patterns, 1.0 / q.n_patterns)
        cand = np.array([2, 5, 7])
        assert sel.compute(post, cand).shape == (3,)
        pick = sel.select(post, cand)
        assert pick in cand.tolist()

    def test_registered(self):
        from cdcat.selection import SELECTION_REGISTRY, create_selector

        assert "MIV" in SELECTION_REGISTRY
        q, model, _, _ = _data(n_items=8, n=5)
        assert create_selector("miv", model).name == "MIV"


# ====================================================================== #
# 流水线
# ====================================================================== #
class TestPipeline:
    def test_nonparametric_block_filled(self):
        q, _, alphas, resp = _data(n=300)
        analysis = run_cdm_analysis(
            q, resp, models=("DINA",), true_patterns=alphas,
            absolute_fit=False, nonparametric=("gdd", "hamming"),
        )
        assert set(analysis.nonparametric) == {"gdd", "hamming"}
        table = analysis.nonparametric_table()
        assert list(table["方法"]) == ["gdd", "hamming"]
        assert (table["参数数"] == 0).all()
        assert table["PMR"].notna().all()
        # 摘要里能看到非参数一段
        assert "非参数分类基线" in analysis.summary()

    def test_nonparametric_without_truth_leaves_pmr_none(self):
        q, _, _, resp = _data(n=100)
        analysis = run_cdm_analysis(
            q, resp, models=("DINA",), absolute_fit=False,
            nonparametric=("gdd",),
        )
        assert analysis.nonparametric["gdd"].pmr is None
        table = analysis.nonparametric_table()
        assert table["PMR"].isna().all()
        # 无真值时摘要也不应崩溃
        assert "非参数分类基线" in analysis.summary()

    def test_unknown_method_raises(self):
        q, _, _, resp = _data(n=50)
        with pytest.raises(ValueError, match="未知非参数方法"):
            run_cdm_analysis(
                q, resp, models=("DINA",), absolute_fit=False,
                nonparametric=("nope",),
            )

    def test_default_off(self):
        q, _, _, resp = _data(n=50)
        analysis = run_cdm_analysis(q, resp, models=("DINA",), absolute_fit=False)
        assert analysis.nonparametric == {}


# ====================================================================== #
# CLI
# ====================================================================== #
class TestCLI:
    def _fixture(self, tmp_path, n: int = 200):
        q, _, alphas, resp = _data(n=n)
        qp = str(tmp_path / "q.csv")
        rp = str(tmp_path / "resp.csv")
        tp = str(tmp_path / "truth.csv")
        write_q_matrix(q, qp)
        write_responses(resp, rp, item_names=q.item_names,
                        examinee_ids=[f"E{i + 1}" for i in range(n)])
        write_responses(alphas.astype(float), tp,
                        item_names=[f"A{i + 1}" for i in range(q.n_attrs)],
                        examinee_ids=[f"E{i + 1}" for i in range(n)])
        return q, {"q": qp, "resp": rp, "truth": tp}

    def test_cli_writes_outputs(self, tmp_path, capsys):
        _, p = self._fixture(tmp_path)
        out = str(tmp_path / "np")
        rc = main([
            "nonparam", "-q", p["q"], "-r", p["resp"], "-o", out,
            "--truth", p["truth"], "--json",
        ])
        assert rc == 0
        text = capsys.readouterr().out
        assert "非参数分类" in text and "PMR" in text

        for name in ("nonparametric.csv", "nonparametric.json",
                     "patterns_gdd.csv", "patterns_hamming.csv"):
            assert (Path(out) / name).exists(), name

        import json

        payload = json.loads((Path(out) / "nonparametric.json").read_text("utf-8"))
        assert set(payload) == {"gdd", "hamming"}
        assert payload["gdd"]["n_params"] == 0
        assert 0.0 <= payload["gdd"]["pmr"] <= 1.0

    def test_cli_hierarchy_option(self, tmp_path, capsys):
        _, p = self._fixture(tmp_path, n=120)
        out = str(tmp_path / "nph")
        rc = main([
            "nonparam", "-q", p["q"], "-r", p["resp"], "-o", out,
            "--methods", "hamming", "--hierarchy", "linear",
        ])
        assert rc == 0
        assert "层级约束：linear" in capsys.readouterr().out

    def test_cli_rejects_unknown_method(self, tmp_path):
        _, p = self._fixture(tmp_path, n=50)
        with pytest.raises(SystemExit):
            main([
                "nonparam", "-q", p["q"], "-r", p["resp"],
                "-o", str(tmp_path / "x"), "--methods", "magic",
            ])
