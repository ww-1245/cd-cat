"""Round 4 新增能力测试：经验贝叶斯先验、并行仿真、绝对拟合与 Q3。

覆盖：

* :class:`~cdcat.calibration.EMCalibrator` 的 π(α) 经验贝叶斯估计
* :meth:`~cdcat.simulation.CDCATSimulator.run` 的 ``n_jobs`` 并行（结果须与串行一致）
* :mod:`cdcat.diagnostics.absolute_fit` 的题目级 χ²/G² 与 Yen Q3

运行::

    pytest tests/test_cdm_diagnostics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import (
    QMatrix,
    DINA,
    DINO,
    EMCalibrator,
    calibrate,
    build_test_items,
    run_cdm_analysis,
    item_fit,
    local_dependence,
    analyse_absolute_fit,
    ItemFitResult,
    LocalDependenceResult,
    AbsoluteFitResult,
    CDCATSimulator,
)


# ====================================================================== #
# 夹具
# ====================================================================== #
N = 1200


def _data(q, model="DINA", n=N, seed=1, pattern_dist=None):
    items = build_test_items(q.n_items, q.n_attrs, seed=seed)
    rng = np.random.default_rng(seed)
    if model == "DINA":
        m = DINA(q, slip=items["slip"], guess=items["guess"])
    elif model == "DINO":
        m = DINO(q, slip=items["slip"], guess=items["guess"])
    else:
        raise ValueError(model)

    s = q.n_patterns
    if pattern_dist is None:
        idx = rng.integers(0, s, n)
    else:
        w = np.asarray(pattern_dist, dtype=float)
        idx = rng.choice(s, size=n, p=w / w.sum())
    alphas = q.attribute_patterns()[idx]
    X = np.array([m.simulate(a, rng=rng) for a in alphas], dtype=float)
    return m, X, alphas, items


@pytest.fixture(scope="module")
def dina_pack():
    """均匀属性分布下的标准数据。"""
    q = QMatrix.from_random(30, 4, seed=101)
    model, X, alphas, items = _data(q, "DINA", seed=101)
    return q, model, X, alphas, items


@pytest.fixture(scope="module")
def skewed_pack():
    """非均匀属性分布 —— 用于检验先验估计的价值。"""
    q = QMatrix.from_random(30, 4, seed=101)
    w = np.linspace(0.2, 3.0, q.n_patterns)
    model, X, alphas, items = _data(q, "DINA", seed=101, pattern_dist=w)
    return q, model, X, alphas, items, w / w.sum()


# ====================================================================== #
# 1. 经验贝叶斯属性先验
# ====================================================================== #
class TestEmpiricalBayesPrior:
    def test_prior_estimated_by_default(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=300).fit(X)
        assert r.prior_estimated is True
        assert r.attr_prior is not None
        assert r.attr_prior.shape == (q.n_patterns,)
        assert r.attr_prior.sum() == pytest.approx(1.0)

    def test_prior_counts_as_free_params(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=300).fit(X)
        assert r.n_prior_params == q.n_patterns - 1
        assert r.n_item_params == 2 * q.n_items
        assert r.n_params == r.n_item_params + r.n_prior_params

    def test_disable_prior_estimation(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=300, estimate_prior=False).fit(X)
        assert r.prior_estimated is False
        assert r.n_prior_params == 0
        assert r.n_params == r.n_item_params
        # 未估计时先验仍为均匀
        assert np.allclose(r.attr_prior, 1.0 / q.n_patterns)

    def test_user_prior_is_respected(self, dina_pack):
        """显式给出 prior 时默认不再估计（除非强制开启）。"""
        q, _, X, _, _ = dina_pack
        p = np.zeros(q.n_patterns)
        p[0] = 1.0
        r = EMCalibrator(q, "DINA", max_iter=200, prior=p).fit(X)
        assert r.prior_estimated is False
        assert np.allclose(r.attr_prior, p)

        r2 = EMCalibrator(q, "DINA", max_iter=200, prior=p,
                          estimate_prior=True).fit(X)
        assert r2.prior_estimated is True
        assert not np.allclose(r2.attr_prior, p)   # 已被数据更新

    def test_recovers_skewed_distribution(self, skewed_pack):
        """估计出的 π(α) 应贴近真实的非均匀模式分布。"""
        q, _, X, _, _, w = skewed_pack
        r = EMCalibrator(q, "DINA", max_iter=400).fit(X)
        corr = float(np.corrcoef(w, r.attr_prior)[0, 1])
        assert corr > 0.85, f"π(α) 与真实分布相关仅 {corr:.4f}"

    def test_prior_improves_or_keeps_accuracy(self, skewed_pack):
        """先验估计不应让参数恢复变差（非均匀分布下通常明显变好）。"""
        q, _, X, _, items, _ = skewed_pack
        r_off = EMCalibrator(q, "DINA", max_iter=400, estimate_prior=False).fit(X)
        r_on = EMCalibrator(q, "DINA", max_iter=400, estimate_prior=True).fit(X)
        mae_off = np.abs(r_off.guess - items["guess"]).mean()
        mae_on = np.abs(r_on.guess - items["guess"]).mean()
        assert mae_on <= mae_off + 0.005, (mae_on, mae_off)

    def test_likelihood_monotone_with_prior(self, skewed_pack):
        """加上先验更新后，EM 的观测似然仍须单调不减。"""
        q, _, X, _, _, _ = skewed_pack
        r = EMCalibrator(q, "DINA", max_iter=300).fit(X)
        h = np.asarray(r.log_likelihood_history, dtype=float)
        assert np.diff(h).min() > -1e-3 * abs(h[0])

    def test_smoothing_prevents_zero_lock(self, dina_pack):
        """任何模式的先验都不应被压到 0（否则永不可恢复）。"""
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=300).fit(X)
        assert (r.attr_prior > 0).all()

    def test_all_models_estimate_prior(self, dina_pack):
        q, _, X, _, _ = dina_pack
        for m in ("DINA", "DINO", "RRUM"):
            r = EMCalibrator(q, m, max_iter=200).fit(X)
            assert r.prior_estimated, m
            assert r.attr_prior.sum() == pytest.approx(1.0), m

    def test_calibrate_convenience_passes_flag(self, dina_pack):
        q, _, X, _, _ = dina_pack
        _, r = calibrate(q, X, model_type="DINA", max_iter=200,
                         estimate_prior=False)
        assert r.prior_estimated is False

    def test_summary_mentions_prior(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=200).fit(X)
        assert "π(α)" in r.summary()


# ====================================================================== #
# 2. 拟合指标与先验参数口径一致
# ====================================================================== #
class TestFitIndicesPriorConsistency:
    def test_res_aic_matches_fit_aic(self, dina_pack):
        """CalibrationResult 与 compute_fit_indices 的 AIC 必须一致。"""
        from cdcat import compute_fit_indices

        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        fit = compute_fit_indices(
            X, r.log_likelihood, r.to_model(q), model_name="DINA",
            n_extra_params=r.n_prior_params,
        )
        assert fit.n_params == r.n_params
        assert fit.aic == pytest.approx(r.aic)
        assert fit.bic == pytest.approx(r.bic)

    def test_pipeline_fit_table_has_param_breakdown(self, dina_pack):
        q, _, X, _, _ = dina_pack
        a = run_cdm_analysis(q, X, models=("DINA",), max_iter=250)
        tbl = a.fit_table()
        for col in ("n_item_params", "n_prior_params", "prior_nonzero"):
            assert col in tbl.columns
        row = tbl.iloc[0]
        assert row["n_params"] == row["n_item_params"] + row["n_prior_params"]


# ====================================================================== #
# 3. 并行仿真
# ====================================================================== #
class TestParallelSimulation:
    @staticmethod
    def _sim(q, **kw):
        items = build_test_items(q.n_items, q.n_attrs, seed=42)
        return CDCATSimulator(
            q, model="DINA",
            model_params={"slip": items["slip"], "guess": items["guess"]},
            selector="PWKL", **kw
        )

    def test_parallel_matches_serial(self):
        """并行结果必须与串行逐条一致（可复现性）。"""
        q = QMatrix.from_random(40, 4, seed=42)
        r1 = self._sim(q).run(n_examinees=200, n_replications=6,
                              seed=7, n_jobs=1)
        r2 = self._sim(q).run(n_examinees=200, n_replications=6,
                              seed=7, n_jobs=2)
        assert len(r1.records) == len(r2.records)
        for a, b in zip(r1.records, r2.records):
            assert np.array_equal(a.true_pattern, b.true_pattern)
            assert np.array_equal(a.est_pattern, b.est_pattern)
            assert a.administered == b.administered
            assert a.stop_reason == b.stop_reason
        assert r1.pmr() == pytest.approx(r2.pmr())
        assert np.array_equal(r1.item_exposure_rate(), r2.item_exposure_rate())

    def test_n_jobs_negative_all_cores(self):
        q = QMatrix.from_random(30, 3, seed=3)
        r = self._sim(q).run(n_examinees=150, n_replications=4,
                             seed=1, n_jobs=-1)
        assert len(r.records) == 600

    def test_n_jobs_one_is_serial(self):
        q = QMatrix.from_random(30, 3, seed=3)
        r = self._sim(q).run(n_examinees=100, n_replications=3,
                             seed=2, n_jobs=1)
        assert len(r.records) == 300

    def test_single_replication_ignores_parallel(self):
        """只有 1 次重复时不必启动进程池。"""
        q = QMatrix.from_random(30, 3, seed=3)
        r = self._sim(q).run(n_examinees=100, n_replications=1,
                             seed=2, n_jobs=4)
        assert len(r.records) == 100

    def test_exposure_control_isolated_per_replication(self):
        """并行下曝光计数仍按每次重复独立累积，不跨重复污染。"""
        from cdcat import MaxExposureLimit

        q = QMatrix.from_random(40, 4, seed=42)
        sim = self._sim(q, exposure_control=MaxExposureLimit(q.n_items, 0.2))
        r1 = sim.run(n_examinees=200, n_replications=4, seed=5, n_jobs=1)

        sim2 = self._sim(q, exposure_control=MaxExposureLimit(q.n_items, 0.2))
        r2 = sim2.run(n_examinees=200, n_replications=4, seed=5, n_jobs=2)
        assert np.array_equal(r1.item_exposure_rate(), r2.item_exposure_rate())

    def test_n_jobs_recorded_in_config(self):
        q = QMatrix.from_random(30, 3, seed=3)
        r = self._sim(q).run(n_examinees=50, n_replications=2, seed=1, n_jobs=2)
        assert r.config["n_jobs"] == 2


# ====================================================================== #
# 4. 题目级绝对拟合
# ====================================================================== #
class TestItemFit:
    def test_shapes_and_finiteness(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = item_fit(r.to_model(q), X, prior=r.attr_prior)

        j = q.n_items
        assert isinstance(res, ItemFitResult)
        for arr in (res.chi_square, res.g_square, res.df, res.p_value,
                    res.p_value_g, res.n_cells, res.max_abs_residual,
                    res.rmse_residual):
            assert arr.shape == (j,)
            assert np.isfinite(arr).all(), arr
        assert (res.chi_square >= 0).all()
        assert (res.g_square >= 0).all()
        assert (res.df >= 0).all()
        assert ((res.p_value >= 0) & (res.p_value <= 1)).all()
        assert (res.max_abs_residual >= 0).all()
        assert (res.max_abs_residual <= 1).all()

    def test_good_model_passes(self, dina_pack):
        """正确模型下，不合格题目比例应接近名义水平（远低于 30%）。"""
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = item_fit(r.to_model(q), X, prior=r.attr_prior)
        rate = (res.p_value < 0.05).mean()
        assert rate < 0.30, f"正确模型下不合格率 {rate:.1%} 过高"

    def test_misspecified_model_fails(self):
        """真实 DINO 却用 DINA 拟合，应大量题目不合格。"""
        q = QMatrix.from_random(30, 4, seed=21)
        _, X, _, _ = _data(q, "DINO", seed=21)

        r_dina = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        fit_bad = item_fit(r_dina.to_model(q), X, prior=r_dina.attr_prior)

        r_dino = EMCalibrator(q, "DINO", max_iter=250).fit(X)
        fit_ok = item_fit(r_dino.to_model(q), X, prior=r_dino.attr_prior)

        bad = (fit_bad.p_value < 0.05).mean()
        good = (fit_ok.p_value < 0.05).mean()
        assert bad > 0.5, f"误设模型不合格率仅 {bad:.1%}"
        assert bad > good + 0.3
        assert fit_bad.max_abs_residual.mean() > fit_ok.max_abs_residual.mean()

    def test_alpha_hat_argument(self, dina_pack):
        """显式传入 alpha_hat 应与内部 MAP 估计给出同量级结果。"""
        q, _, X, alphas, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        model = r.to_model(q)
        a = item_fit(model, X, alpha_hat=alphas, prior=r.attr_prior)
        b = item_fit(model, X, prior=r.attr_prior)
        # 真值分组下拟合更好，但同一量级
        assert a.chi_square.mean() < b.chi_square.mean() * 1.5

    def test_n_bins_one_uses_alpha_only(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = item_fit(r.to_model(q), X, prior=r.attr_prior, n_bins=1)
        assert np.isfinite(res.chi_square).all()

    def test_dataframe_and_helpers(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = item_fit(r.to_model(q), X, prior=r.attr_prior)

        df = res.to_dataframe()
        assert df.shape[0] == q.n_items
        for col in ("题号", "χ²", "G²", "df", "p值", "最大残差"):
            assert col in df.columns
        assert res.poorly_fitting().shape[0] == int((res.p_value < 0.05).sum())
        assert res.worst_items(3).shape[0] == 3
        assert isinstance(res.summary(), str)

    def test_dimension_mismatch(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=200).fit(X)
        with pytest.raises(ValueError):
            item_fit(r.to_model(q), X[:, :-1])


# ====================================================================== #
# 5. 局部依赖 Q3
# ====================================================================== #
class TestLocalDependence:
    def test_matrix_properties(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = local_dependence(r.to_model(q), X, prior=r.attr_prior)

        j = q.n_items
        assert isinstance(res, LocalDependenceResult)
        assert res.q3.shape == (j, j)
        assert np.allclose(np.diag(res.q3), 0.0)
        assert np.allclose(res.q3, res.q3.T)
        assert np.isfinite(res.q3).all()
        assert len(res.pairs) == j * (j - 1) // 2

    def test_pairs_sorted_descending(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = local_dependence(r.to_model(q), X, prior=r.attr_prior)
        vals = [v for _, _, v in res.pairs]
        assert vals == sorted(vals, reverse=True)
        assert res.max_q3 == pytest.approx(vals[0])
        assert res.mean_q3 == pytest.approx(np.mean(vals))

    def test_correct_model_shows_no_dependence(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = local_dependence(r.to_model(q), X, prior=r.attr_prior)
        assert res.n_flagged == 0, f"正确模型下竟有 {res.n_flagged} 对超标"
        assert res.mean_q3 == pytest.approx(0.0, abs=0.05)

    def test_detects_injected_local_dependence(self):
        """人为让两题共享一个隐藏属性，Q3 应把这一对揪出来。"""
        q = QMatrix.from_random(30, 4, seed=5)
        model, X, _, _ = _data(q, "DINA", n=2000, seed=11)

        hidden = np.random.default_rng(99).random(2000) < 0.5
        for it in (5, 18):
            wrong = (X[:, it] == 0) & hidden
            X[wrong, it] = 1.0

        r = EMCalibrator(q, "DINA", max_iter=300).fit(X)
        res = local_dependence(r.to_model(q), X, prior=r.attr_prior)

        assert res.n_flagged >= 1
        top = {(a, b) for a, b, _ in res.pairs[:3]}
        assert (5, 18) in top, f"未能检出 (5,18)，top3 = {top}"
        assert res.q3[5, 18] > 0.2

    def test_flagged_pairs_and_helpers(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = local_dependence(r.to_model(q), X, prior=r.attr_prior)

        assert res.top_pairs(5).shape[0] == 5
        assert res.to_dataframe().shape == (q.n_items, q.n_items)
        assert res.flagged_pairs().shape[0] == res.n_flagged
        assert isinstance(res.summary(), str)

    def test_threshold_affects_flagging(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        lo = local_dependence(r.to_model(q), X, prior=r.attr_prior, threshold=-1.0)
        hi = local_dependence(r.to_model(q), X, prior=r.attr_prior, threshold=1.0)
        assert lo.n_flagged == len(lo.pairs)
        assert hi.n_flagged == 0


# ====================================================================== #
# 6. 汇总入口与流水线集成
# ====================================================================== #
class TestAbsoluteFitIntegration:
    def test_analyse_absolute_fit(self, dina_pack):
        q, _, X, _, _ = dina_pack
        r = EMCalibrator(q, "DINA", max_iter=250).fit(X)
        res = analyse_absolute_fit(r.to_model(q), X, prior=r.attr_prior)
        assert isinstance(res, AbsoluteFitResult)
        assert isinstance(res.item_fit, ItemFitResult)
        assert isinstance(res.local_dependence, LocalDependenceResult)
        assert res.items == q.n_items
        assert res.observations == X.shape[0]
        assert isinstance(res.summary(), str)

    def test_pipeline_includes_absolute_fit(self, dina_pack):
        q, _, X, _, _ = dina_pack
        a = run_cdm_analysis(q, X, models=("DINA",), max_iter=250)
        assert a.absolute_fit is not None
        assert a.absolute_fit.item_fit.chi_square.shape == (q.n_items,)
        assert "绝对拟合" in a.summary()

    def test_pipeline_can_disable(self, dina_pack):
        q, _, X, _, _ = dina_pack
        a = run_cdm_analysis(q, X, models=("DINA",), max_iter=250,
                             absolute_fit=False)
        assert a.absolute_fit is None

    def test_pipeline_excel_has_new_sheets(self, dina_pack, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        q, _, X, _, _ = dina_pack
        a = run_cdm_analysis(q, X, models=("DINA",), max_iter=250)
        out = tmp_path / "report.xlsx"
        a.to_excel(str(out))
        assert out.exists()
        sheets = set(pd.ExcelFile(out).sheet_names)
        for s in ("模型拟合指标", "题目参数", "被试诊断",
                  "题目绝对拟合", "局部依赖Q3", "属性先验"):
            assert s in sheets, f"缺少 sheet {s}；实际 {sheets}"

    def test_pipeline_uses_estimated_prior_for_diagnosis(self, skewed_pack):
        """诊断应沿用标定出的先验，而非退回均匀。"""
        q, _, X, _, _, _ = skewed_pack
        a = run_cdm_analysis(q, X, models=("DINA",), max_iter=300)
        prior = a.models["DINA"].attr_prior
        assert prior is not None
        # 均匀先验应与估计先验显著不同
        assert np.abs(prior - 1.0 / q.n_patterns).max() > 1e-3


# ====================================================================== #
# 7. 可视化冒烟
# ====================================================================== #
class TestNewPlots:
    def test_matplotlib_plots(self, dina_pack, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cdcat import calibration_plots as cp

        q, _, X, _, _ = dina_pack
        a = run_cdm_analysis(q, X, models=("DINA",), max_iter=200)

        figs = [
            cp.plot_item_fit(a.absolute_fit.item_fit),
            cp.plot_q3_heatmap(a.absolute_fit.local_dependence),
            cp.plot_attr_prior(a.models["DINA"]),
            cp.absolute_fit_report_figure(a),
        ]
        out = tmp_path / "absfit.png"
        fig = cp.absolute_fit_report_figure(a)
        fig.savefig(out, dpi=90)
        assert out.exists() and out.stat().st_size > 1000
        for f in figs:
            assert f is not None
            plt.close(f)

    def test_plotly_charts(self, dina_pack):
        pytest.importorskip("plotly")
        from demo import cdm_charts as cc

        q, _, X, _, _ = dina_pack
        a = run_cdm_analysis(q, X, models=("DINA",), max_iter=200)
        figs = [
            cc.fig_item_fit(a.absolute_fit.item_fit),
            cc.fig_q3_heatmap(a.absolute_fit.local_dependence),
            cc.fig_attr_prior(a.models["DINA"]),
        ]
        for f in figs:
            assert f is not None and len(f.data) > 0
