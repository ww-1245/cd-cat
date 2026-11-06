"""CDM 参数标定 / 诊断 / 模型比较 测试套件。

覆盖本轮新增能力：

* :mod:`cdcat.calibration` —— DINA / DINO / G-DINA / RRUM 的 MMLE-EM 估计
* :mod:`cdcat.diagnostics` —— 模型拟合指标、被试诊断、模型比较、LRT
* :mod:`cdcat.pipeline`    —— 端到端 CDM 分析流水线
* :mod:`demo.cdm_charts`   —— 交互式可视化

运行::

    pytest tests/test_cdm_analysis.py -v
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
    EMCalibrator,
    CalibrationResult,
    calibrate,
    FitIndices,
    compute_fit_indices,
    saturated_log_likelihood,
    PatternDiagnosis,
    diagnose_patterns,
    ModelComparison,
    compare_models,
    likelihood_ratio_test,
    parameter_recovery,
    run_cdm_analysis,
)
from cdcat.diagnostics import _count_free_params
from cdcat.utils import build_test_items


# ====================================================================== #
# 通用夹具
# ====================================================================== #
N_EXAMINEES = 1200


def make_data(q, model_name="DINA", n=N_EXAMINEES, seed=7):
    """按指定模型生成作答矩阵，返回 (model, X, alphas, slip, guess)。"""
    J, K = q.n_items, q.n_attrs
    items = build_test_items(J, K, seed=seed)
    slip, guess = items["slip"], items["guess"]

    if model_name == "DINA":
        model = DINA(q, slip=slip, guess=guess)
    elif model_name == "DINO":
        model = DINO(q, slip=slip, guess=guess)
    elif model_name == "G-DINA":
        model = GDINA.from_dina(q, slip, guess)
    elif model_name == "RRUM":
        # r_star 是"每属性一档"的折损系数，需按题给列表
        r_star = [
            np.full(max(int(q.matrix[i].sum()), 1), float(guess[i]))
            for i in range(q.n_items)
        ]
        model = RRUM(q, pi_star=1.0 - slip, r_star=r_star)
    else:
        raise ValueError(model_name)

    rng = np.random.default_rng(seed)
    S = q.attribute_patterns().shape[0]
    alphas = q.attribute_patterns()[rng.integers(0, S, n)]
    X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)
    return model, X, alphas, slip, guess


def fit_with(q, X, model_name, max_iter=250):
    """标定并返回 (CalibrationResult, FitIndices)。"""
    res = EMCalibrator(q, model_type=model_name, max_iter=max_iter).fit(X, verbose=False)
    fit = compute_fit_indices(
        X, res.log_likelihood, res.to_model(q), model_name=model_name
    )
    return res, fit


@pytest.fixture(scope="module")
def dina_data():
    q = QMatrix.from_random(30, 4, seed=101)
    return q, make_data(q, "DINA", seed=101)


@pytest.fixture(scope="module")
def gdina_data():
    q = QMatrix.from_random(20, 3, seed=102)
    return q, make_data(q, "G-DINA", seed=102)


@pytest.fixture(scope="module")
def rrum_data():
    q = QMatrix.from_random(20, 3, seed=103)
    return q, make_data(q, "RRUM", seed=103)


@pytest.fixture(scope="module")
def dina_analysis(dina_data):
    q, (_, X, alphas, slip, guess) = dina_data
    return run_cdm_analysis(
        q, X,
        models=("DINA", "DINO", "G-DINA"),
        true_slip=slip, true_guess=guess, true_patterns=alphas,
        max_iter=250,
    )


# ====================================================================== #
# 1. G-DINA EM 估计
# ====================================================================== #
class TestGDINACalibration:
    def test_converges(self, gdina_data):
        q, (_, X, _, _, _) = gdina_data
        cal = EMCalibrator(q, model_type="G-DINA", max_iter=200)
        res = cal.fit(X, verbose=False)

        assert isinstance(res, CalibrationResult)
        assert res.model_name == "G-DINA"
        assert res.converged
        assert res.n_iterations <= 200

    def test_log_likelihood_monotone(self, gdina_data):
        """EM 的观测似然必须单调不减（数值容差内）。"""
        q, (_, X, _, _, _) = gdina_data
        cal = EMCalibrator(q, model_type="G-DINA", max_iter=120)
        res = cal.fit(X, verbose=False)

        hist = np.asarray(res.log_likelihood_history, dtype=float)
        assert hist.size > 2
        diffs = np.diff(hist)
        assert diffs.min() > -1e-3 * abs(hist[0])

    def test_prob_matrix_recovers_truth(self, gdina_data):
        """G-DINA 恢复各潜在组的正确作答概率（核心目标）。"""
        q, (model, X, _, _, _) = gdina_data
        res, _ = fit_with(q, X, "G-DINA", max_iter=300)
        est = res.to_model(q).prob_matrix()   # (S, J)

        true = model.prob_matrix()            # (S, J)
        assert est.shape == true.shape
        mae = np.abs(est - true).mean()
        assert mae < 0.06, f"G-DINA 概率矩阵 MAE 过大: {mae:.4f}"

    def test_pattern_recovery(self, gdina_data):
        q, (_, X, alphas, _, _) = gdina_data
        res, _ = fit_with(q, X, "G-DINA", max_iter=300)
        diag = diagnose_patterns(res.to_model(q), X)

        pmr = np.mean(np.all(diag.map_patterns == alphas, axis=1))
        amr = np.mean(diag.map_patterns == alphas)
        assert pmr > 0.70, f"G-DINA PMR 过低: {pmr:.4f}"
        assert amr > 0.88, f"G-DINA AMR 过低: {amr:.4f}"

    def test_n_params_gdina(self):
        """G-DINA 每题的参数数应为 2^r_j。"""
        q = QMatrix.from_random(10, 3, seed=1)
        model = GDINA.from_dina(q, np.full(q.n_items, 0.2), np.full(q.n_items, 0.2))
        expected = int(sum(2 ** len(q.required_attrs(j)) for j in range(q.n_items)))
        assert _count_free_params(model) == expected


# ====================================================================== #
# 2. RRUM EM 估计
# ====================================================================== #
class TestRRUMCalibration:
    def test_converges(self, rrum_data):
        q, (_, X, _, _, _) = rrum_data
        cal = EMCalibrator(q, model_type="RRUM", max_iter=250)
        res = cal.fit(X, verbose=False)
        assert res.converged
        assert res.pi_star is not None and res.r_star is not None
        assert res.pi_star.shape == (q.n_items,)

    def test_pi_star_recovery(self, rrum_data):
        """π*（全掌握组正确率）应被较准确地恢复。"""
        q, (model, X, _, _, _) = rrum_data
        cal = EMCalibrator(q, model_type="RRUM", max_iter=300)
        res = cal.fit(X, verbose=False)

        mae = np.abs(res.pi_star - model.pi_star).mean()
        assert mae < 0.06, f"RRUM π* MAE 过大: {mae:.4f}"

    def test_r_star_in_unit_interval(self, rrum_data):
        q, (_, X, _, _, _) = rrum_data
        cal = EMCalibrator(q, model_type="RRUM", max_iter=250)
        res = cal.fit(X, verbose=False)
        # r* 是"缺一属性"的折损系数，须落在 (0, 1]
        assert isinstance(res.r_star, list)
        assert len(res.r_star) == q.n_items
        for i, r in enumerate(res.r_star):
            expected = max(int(q.matrix[i].sum()), 1)
            assert r.shape == (expected,)
            assert (r > 0).all() and (r <= 1.0 + 1e-6).all()

    def test_r_star_recovery(self, rrum_data):
        """r* 折损系数应被较准确地恢复。"""
        q, (model, X, _, _, _) = rrum_data
        cal = EMCalibrator(q, model_type="RRUM", max_iter=300)
        res = cal.fit(X, verbose=False)

        est = np.concatenate(res.r_star)
        true = np.concatenate(model.r_star)
        assert np.abs(est - true).mean() < 0.08

    def test_rrum_beats_dina_on_rrum_data(self, rrum_data):
        """在真实 RRUM 数据上，RRUM 的拟合应优于 DINA（模型选择正确性）。"""
        q, (_, X, _, _, _) = rrum_data
        _, f_rrum = fit_with(q, X, "RRUM")
        _, f_dina = fit_with(q, X, "DINA")
        assert f_rrum.aic < f_dina.aic, (
            f"RRUM AIC={f_rrum.aic:.1f} 未优于 DINA AIC={f_dina.aic:.1f}"
        )

    def test_n_params_rrum(self):
        q = QMatrix.from_random(10, 3, seed=1)
        r_star = [
            np.full(max(int(q.matrix[i].sum()), 1), 0.3) for i in range(q.n_items)
        ]
        model = RRUM(q, pi_star=np.full(q.n_items, 0.8), r_star=r_star)
        expected = int(sum(1 + len(q.required_attrs(j)) for j in range(q.n_items)))
        assert _count_free_params(model) == expected


# ====================================================================== #
# 3. DINA / DINO 回归保护
# ====================================================================== #
class TestDINARegression:
    def test_dina_parameter_recovery(self, dina_data):
        q, (_, X, _, slip, guess) = dina_data
        res = EMCalibrator(q, model_type="DINA", max_iter=300).fit(X, verbose=False)
        assert np.abs(res.slip - slip).mean() < 0.05
        assert np.abs(res.guess - guess).mean() < 0.05
        assert np.corrcoef(res.slip, slip)[0, 1] > 0.85

    def test_dina_best_model_on_dina_data(self, dina_data):
        """DINA 数据上，DINA 的 AIC/BIC 应优于 DINO 与 G-DINA。"""
        q, (_, X, _, _, _) = dina_data
        fits = {m: fit_with(q, X, m)[1] for m in ("DINA", "DINO", "G-DINA")}
        assert fits["DINA"].bic == min(f.bic for f in fits.values())
        assert fits["DINA"].aic <= fits["G-DINA"].aic + 1e-6

    def test_calibration_result_to_model(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res = EMCalibrator(q, model_type="DINA", max_iter=200).fit(X)
        model = res.to_model(q)
        # prob_matrix 形状为 (n_patterns, n_items)
        assert model.prob_matrix().shape == (q.attribute_patterns().shape[0], q.n_items)

    def test_summary_is_str(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res = EMCalibrator(q, model_type="DINA", max_iter=150).fit(X)
        s = res.summary()
        assert isinstance(s, str) and "DINA" in s

    def test_deviance_property(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res = EMCalibrator(q, model_type="DINA", max_iter=150).fit(X)
        assert res.deviance == pytest.approx(-2.0 * res.log_likelihood)

    def test_convenience_calibrate(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        model, res = calibrate(q, X, model_type="DINA", max_iter=150)
        assert isinstance(res, CalibrationResult)
        assert res.model_name == "DINA"
        assert model.prob_matrix().shape == (q.attribute_patterns().shape[0], q.n_items)


# ====================================================================== #
# 4. 拟合指标
# ====================================================================== #
class TestFitIndices:
    def test_fields_present(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, fit = fit_with(q, X, "DINA")

        assert isinstance(fit, FitIndices)
        assert fit.aic == pytest.approx(-2 * fit.log_likelihood + 2 * fit.n_params)
        assert fit.bic == pytest.approx(
            -2 * fit.log_likelihood + fit.n_params * np.log(fit.n_examinees)
        )
        assert fit.n_examinees == X.shape[0]
        assert fit.n_items == q.n_items
        assert fit.n_observations == X.shape[0] * q.n_items
        assert fit.deviance == pytest.approx(-2 * fit.log_likelihood)
        assert fit.model_name == "DINA"

    def test_bic_penalises_more_than_aic(self, dina_data):
        """BIC 惩罚 = k·ln(n)，n=e²≈7.4 时两者相同；n=1200 时必然更强。"""
        q, (_, X, _, _, _) = dina_data
        _, fit = fit_with(q, X, "DINA")
        assert fit.bic > fit.aic

    def test_aicc_geq_aic(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        _, fit = fit_with(q, X, "DINA")
        assert fit.aicc >= fit.aic - 1e-9

    def test_saturated_ll_upper_bound(self, dina_data):
        """饱和模型对数似然必须 ≥ 任何受约束模型的对数似然。"""
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        sat = saturated_log_likelihood(X)
        assert sat >= res.log_likelihood - 1e-6

    def test_relative_fit_in_range(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        _, fit = fit_with(q, X, "DINA")
        assert fit.cfi <= 1.0 + 1e-6
        assert fit.rmsea >= 0.0
        assert fit.tli <= 1.0 + 1e-6

    def test_to_dict_float_and_optional(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        _, fit = fit_with(q, X, "DINA")
        d = fit.to_dict()
        assert "aic" in d and "bic" in d and "log_likelihood" in d
        assert "model" in d
        # pmr/amr 未设置时不应出现在字典里
        assert "pmr" not in d

        fit.pmr = 0.9
        fit.amr = 0.97
        d2 = fit.to_dict()
        assert d2["pmr"] == pytest.approx(0.9)
        assert d2["amr"] == pytest.approx(0.97)

    def test_summary_str(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        _, fit = fit_with(q, X, "DINA")
        assert isinstance(fit.summary(), str)


# ====================================================================== #
# 5. 被试诊断
# ====================================================================== #
class TestDiagnosis:
    def test_shapes_and_ranges(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)

        S = q.attribute_patterns().shape[0]
        assert isinstance(diag, PatternDiagnosis)
        assert diag.posterior.shape == (X.shape[0], S)
        assert diag.map_patterns.shape == (X.shape[0], q.n_attrs)
        assert diag.eap_marginal.shape == (X.shape[0], q.n_attrs)
        assert diag.attr_reliability.shape == (X.shape[0], q.n_attrs)

        assert np.allclose(diag.posterior.sum(axis=1), 1.0, atol=1e-6)
        assert diag.attr_reliability.min() >= -1e-9
        assert diag.attr_reliability.max() <= 1.0 + 1e-9

    def test_map_matches_posterior_argmax(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)

        patterns = q.attribute_patterns().astype(np.int8)
        expected = patterns[np.argmax(diag.posterior, axis=1)]
        assert np.array_equal(diag.map_patterns, expected)
        assert np.array_equal(diag.pattern_indices, diag.posterior.argmax(axis=1))

    def test_map_confidence_matches(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)
        n = X.shape[0]
        expected = diag.posterior[np.arange(n), diag.pattern_indices]
        assert np.allclose(diag.map_confidence, expected)
        assert diag.map_confidence.min() >= 0.0
        assert diag.map_confidence.max() <= 1.0 + 1e-9

    def test_eap_thresholding_consistency(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)
        assert np.array_equal(diag.eap_patterns, (diag.eap_marginal >= 0.5))

    def test_accuracy_reasonable(self, dina_data):
        q, (_, X, alphas, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)
        amr = np.mean(diag.map_patterns == alphas)
        assert amr > 0.88, f"MAP 属性判准率偏低: {amr:.4f}"

    def test_mastery_counts_and_proportion(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)

        counts = diag.mastered_counts()
        assert counts.shape == (q.n_attrs,)
        assert counts.sum() == diag.map_patterns.sum()

        prop = diag.mastery_proportion()
        assert np.allclose(prop, counts / X.shape[0])
        assert diag.n_attrs == q.n_attrs
        assert diag.n_examinees == X.shape[0]

    def test_low_reliability_attrs(self, dina_data):
        """low_reliability_attrs 返回每个被试中低于阈值的属性索引列表。"""
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)

        thr = 0.6
        low = diag.low_reliability_attrs(threshold=thr)
        assert isinstance(low, list)
        assert len(low) == X.shape[0]
        # 逐被试核对
        for i in (0, 1, 17, X.shape[0] - 1):
            expected = np.where(diag.attr_reliability[i] < thr)[0].tolist()
            assert low[i] == expected

    def test_summary_and_dataframe(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X, attr_names=["A", "B", "C", "D"])
        assert isinstance(diag.summary(), str)
        assert list(diag.attr_names) == ["A", "B", "C", "D"]

        df = diag.to_dataframe()
        # 每个被试一行，列含 掌握/概率/信度/MAP后验概率/模式索引
        assert df.shape[0] == X.shape[0]
        for col in ("A_掌握", "A_概率", "A_信度", "MAP后验概率", "模式索引"):
            assert col in df.columns

    def test_pattern_distribution(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)
        dist = diag.pattern_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) == X.shape[0]

    def test_mean_confidence_in_range(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        diag = diagnose_patterns(res.to_model(q), X)
        mc = diag.mean_confidence()
        assert 0.0 < mc <= 1.0

    def test_non_uniform_prior_shifts_posterior(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        model = res.to_model(q)

        S = q.attribute_patterns().shape[0]
        prior = np.zeros(S)
        prior[-1] = 1.0  # 只信最后一个模式
        diag = diagnose_patterns(model, X, prior=prior)
        assert np.all(diag.pattern_indices == S - 1)

    def test_dimension_mismatch(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        res, _ = fit_with(q, X, "DINA")
        with pytest.raises(ValueError):
            diagnose_patterns(res.to_model(q), X[:, :-1])


# ====================================================================== #
# 6. 模型比较与 LRT
# ====================================================================== #
class TestModelComparison:
    def test_best_selection_on_dina_data(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        fits = {m: fit_with(q, X, m)[1] for m in ("DINA", "DINO", "G-DINA")}

        cmp = compare_models(fits)
        assert isinstance(cmp, ModelComparison)
        assert cmp.best_aic == "DINA"
        assert cmp.best_bic == "DINA"
        assert cmp.best_aicc == "DINA"

    def test_comparison_table(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        fits = {m: fit_with(q, X, m)[1] for m in ("DINA", "G-DINA")}

        cmp = compare_models(fits)
        table = cmp.to_dataframe()
        assert table.shape[0] == 2
        for col in ("model", "aic", "bic", "n_params"):
            assert col in table.columns
        assert isinstance(cmp.summary(), str)

    def test_lrt_nested(self, dina_data):
        """DINA 嵌套于 G-DINA：LRT 统计量应 ≥ 0，df = 参数差。"""
        q, (_, X, _, _, _) = dina_data
        r_simple = EMCalibrator(q, model_type="DINA", max_iter=200).fit(X)
        r_full = EMCalibrator(q, model_type="G-DINA", max_iter=200).fit(X)

        out = likelihood_ratio_test(
            r_simple.log_likelihood, r_simple.n_params,
            r_full.log_likelihood, r_full.n_params,
        )
        assert out["lr_statistic"] >= -1e-6
        assert out["df"] == max(r_full.n_params - r_simple.n_params, 0)
        assert 0.0 <= out["p_value"] <= 1.0
        assert out["significant"] in (0.0, 1.0)

    def test_lrt_same_model_gives_trivial(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        r = EMCalibrator(q, model_type="DINA", max_iter=150).fit(X)
        out = likelihood_ratio_test(
            r.log_likelihood, r.n_params, r.log_likelihood, r.n_params
        )
        assert out["lr_statistic"] == pytest.approx(0.0, abs=1e-9)
        assert out["df"] == 0
        assert out["p_value"] == pytest.approx(1.0)

    def test_lrt_chi2_sf_known_values(self):
        """卡方上尾概率的数值实现应与已知分位数吻合。"""
        # LRT 统计量 = 2·(ll_full - ll_restricted)，取 ll_restricted=0
        # chi2(1) 的 0.95 分位数 ≈ 3.8415  →  P(X>3.8415) ≈ 0.05
        out = likelihood_ratio_test(0.0, 0, 1.92075, 1)
        assert out["lr_statistic"] == pytest.approx(3.8415, abs=1e-4)
        assert out["p_value"] == pytest.approx(0.05, abs=5e-3)
        # chi2(2) 的 0.95 分位数 ≈ 5.9915
        out2 = likelihood_ratio_test(0.0, 0, 2.99575, 2)
        assert out2["lr_statistic"] == pytest.approx(5.9915, abs=1e-4)
        assert out2["p_value"] == pytest.approx(0.05, abs=5e-3)

    def test_lrt_degenerate_df(self):
        """full 参数比 restricted 少时，df<=0，应安全返回平凡结果。"""
        out = likelihood_ratio_test(-100.0, 20, -100.0, 10)
        assert out["df"] <= 0
        assert out["p_value"] == pytest.approx(1.0)

    def test_parameter_recovery_helper(self, dina_data):
        q, (_, X, _, slip, guess) = dina_data
        res = EMCalibrator(q, model_type="DINA", max_iter=250).fit(X)
        rec_s = parameter_recovery(res.slip, slip)
        rec_g = parameter_recovery(res.guess, guess)

        for rec in (rec_s, rec_g):
            assert set(rec) >= {"correlation", "mae", "rmse", "bias"}
            assert rec["mae"] < 0.06
            assert rec["correlation"] > 0.8

    def test_parameter_recovery_length_mismatch(self):
        with pytest.raises(ValueError):
            parameter_recovery(np.zeros(5), np.zeros(6))


# ====================================================================== #
# 7. 端到端流水线
# ====================================================================== #
class TestPipeline:
    def test_run_cdm_analysis_basic(self, dina_data):
        q, (_, X, alphas, slip, guess) = dina_data
        analysis = run_cdm_analysis(
            q, X,
            models=("DINA", "DINO"),
            true_slip=slip, true_guess=guess, true_patterns=alphas,
            max_iter=200,
        )
        assert set(analysis.models) == {"DINA", "DINO"}
        assert analysis.best_model in ("DINA", "DINO")
        assert analysis.comparison is not None
        assert analysis.lrt is not None
        assert analysis.recovery is not None

    def test_pipeline_selects_dina(self, dina_analysis):
        assert dina_analysis.best_model == "DINA"

    def test_fit_table_has_pmr_amr(self, dina_analysis):
        tbl = dina_analysis.fit_table()
        assert tbl.shape[0] == len(dina_analysis.models)
        assert "pmr" in tbl.columns and "amr" in tbl.columns
        assert np.isfinite(tbl["pmr"]).all()

    def test_pipeline_diagnosis_table(self, dina_analysis, dina_data):
        q, _ = dina_data
        tbl = dina_analysis.diagnosis_table()
        # 每个被试一行
        assert tbl.shape[0] == N_EXAMINEES

        named = dina_analysis.diagnosis_table(model_name="G-DINA")
        assert named.shape[0] == N_EXAMINEES

    def test_pipeline_param_table(self, dina_analysis, dina_data):
        q, _ = dina_data
        tbl = dina_analysis.param_table()
        # G-DINA 的 slip/guess 为 None，会被跳过；DINA/DINO 各 30 行
        expected = sum(
            1 for res in dina_analysis.models.values() if res.slip is not None
        ) * q.n_items
        assert tbl.shape[0] == expected
        for col in ("模型", "题号", "Q向量", "slip", "guess", "区分度"):
            assert col in tbl.columns

    def test_pipeline_summary_str(self, dina_analysis):
        s = dina_analysis.summary()
        assert isinstance(s, str)
        assert "DINA" in s
        assert len(s) > 100

    def test_get_model_by_name(self, dina_analysis):
        # model() 返回已标定的响应模型实例
        m = dina_analysis.model("DINA")
        assert m.name == "DINA"
        with pytest.raises(KeyError):
            dina_analysis.model("RRUM")

    def test_to_excel(self, dina_data, tmp_path):
        q, (_, X, _, _, _) = dina_data
        analysis = run_cdm_analysis(q, X, models=("DINA",), max_iter=150)
        out = tmp_path / "cdm_report.xlsx"
        analysis.to_excel(str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_all_four_models(self, dina_data):
        """四个模型应能同时跑通并各自产出拟合指标。"""
        q, (_, X, _, _, _) = dina_data
        analysis = run_cdm_analysis(
            q, X, models=("DINA", "DINO", "G-DINA", "RRUM"), max_iter=200
        )
        assert set(analysis.models) == {"DINA", "DINO", "G-DINA", "RRUM"}
        for name, fit in analysis.fits.items():
            assert np.isfinite(fit.aic), name
            assert np.isfinite(fit.bic), name
            assert fit.n_params > 0

    def test_attr_names_propagate(self, dina_data):
        q, (_, X, _, _, _) = dina_data
        names = [f"属性{k}" for k in range(q.n_attrs)]
        analysis = run_cdm_analysis(
            q, X, models=("DINA",), attr_names=names, max_iter=150
        )
        assert list(analysis.diagnosis[analysis.best_model].attr_names) == names


# ====================================================================== #
# 8. 可视化冒烟测试
# ====================================================================== #
class TestVisualization:
    def test_static_report_figure(self, dina_analysis, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cdcat.calibration_plots import calibration_report_figure

        fig = calibration_report_figure(dina_analysis)
        assert fig is not None
        assert len(fig.axes) == 6

        out = tmp_path / "report.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 1000

    def test_static_individual_plots(self, dina_analysis):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cdcat import calibration_plots as cp

        best = dina_analysis.best_model
        res = dina_analysis.models[best]
        diag = dina_analysis.diagnosis[best]
        fp = dina_analysis.fits

        figs = [
            cp.plot_ll_convergence(dina_analysis.models),
            cp.plot_parameter_recovery(res.slip, res.slip),
            cp.plot_item_discrimination(1.0 - res.slip - res.guess),
            cp.plot_fit_comparison(fp),
            cp.plot_mastery_proportion(diag.mastery_proportion()),
            cp.plot_attr_reliability(diag.attr_reliability),
            cp.plot_pattern_heatmap(diag),
        ]
        assert all(f is not None for f in figs)
        for f in figs:
            plt.close(f)

    def test_plotly_charts(self, dina_analysis):
        pytest.importorskip("plotly")

        from demo import cdm_charts as cc

        best = dina_analysis.best_model
        res = dina_analysis.models[best]
        diag = dina_analysis.diagnosis[best]

        figs = [
            cc.fig_ll_convergence(dina_analysis.models),
            cc.fig_param_recovery(res.slip, res.slip),
            cc.fig_item_discrimination(1.0 - res.slip - res.guess),
            cc.fig_fit_comparison(dina_analysis.fits),
            cc.fig_mastery_proportion(diag.mastery_proportion()),
            cc.fig_attr_reliability(diag.attr_reliability),
            cc.fig_pattern_distribution(diag),
        ]
        for f in figs:
            assert f is not None
            assert len(f.data) > 0
