"""缺失作答处理的回归测试。

旧实现把 ``np.nan`` 直接 ``nan_to_num(nan=0)``，即当作答错。后果是被试在
"已掌握"的题上被记一次失误，slip 被系统性抬高——实测 20% 随机缺失就能把
slip 从 0.10 抬到 0.29、40% 顶到 0.499 上界。现在改为按题边际化：
缺失题既不进入该被试的似然项，也不进入 M 步的分子分母。

本文件锁定四条性质：

1. **无偏**：20% / 40% 随机缺失下，slip 均值仍接近真值（不是只要求"跑得通"）；
2. **等价**：全缺失的考生不改变任何参数（权重为 0）；
3. **数值正确**：E 步与边际似然和"只对观测题连乘"的暴力实现一致；
4. **安全**：整列缺失只警告、不产生 NaN，六个模型都能出结果。
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import QMatrix
from cdcat.calibration import EMCalibrator

# ====================================================================== #
# 数据生成
# ====================================================================== #
def _make_data(seed: int = 7, n: int = 1000, j: int = 30, k: int = 4):
    """生成 DINA 真值作答矩阵，返回 (q, X, rng)。"""
    rng = np.random.default_rng(seed)
    m = (rng.random((j, k)) < 0.45).astype(int)
    m[np.arange(k), np.arange(k)] = 1          # 每个属性至少有一道单属性题
    m[m.sum(axis=1) == 0, 0] = 1
    q = QMatrix(m)
    alpha = (rng.random((n, k)) < 0.6).astype(int)
    eta = np.all(alpha[:, None, :] >= m[None, :, :], axis=2)   # (n, j)
    p = np.where(eta, 0.9, 0.1)                                # slip = guess = 0.1
    x = (rng.random((n, j)) < p).astype(float)
    return q, x, rng


def _drop(x: np.ndarray, rng, rate: float) -> np.ndarray:
    out = x.copy()
    out[rng.random(x.shape) < rate] = np.nan
    return out


# ====================================================================== #
# 1. 参数恢复：缺失不能抬高 slip
# ====================================================================== #
class TestRecovery:
    @pytest.mark.parametrize("rate", [0.20, 0.40])
    def test_slip_not_inflated_by_missing(self, rate):
        """20% / 40% 随机缺失下 slip 均值仍应接近 0.10（旧实现是 0.29 / 0.46）。"""
        q, x, rng = _make_data()
        res = EMCalibrator(q, model_type="DINA", max_iter=300).fit(_drop(x, rng, rate))

        assert np.isfinite(res.slip).all()
        assert res.slip.mean() == pytest.approx(0.10, abs=0.05)
        assert np.abs(res.slip - 0.10).mean() < 0.05
        # guess 只被轻微压低，不能出现系统性崩坏
        assert res.guess.mean() == pytest.approx(0.10, abs=0.05)

    def test_missing_and_complete_estimates_close(self):
        """同一份数据的"完整版"与"20% 缺失版"应给出接近的题目参数。"""
        q, x, rng = _make_data()
        full = EMCalibrator(q, model_type="DINA", max_iter=300).fit(x)
        miss = EMCalibrator(q, model_type="DINA", max_iter=300).fit(_drop(x, rng, 0.2))
        assert np.abs(full.slip - miss.slip).mean() < 0.03
        assert np.abs(full.guess - miss.guess).mean() < 0.03

    def test_complete_data_unchanged(self):
        """无缺失时的结果不能因为改造而漂移。"""
        q, x, _ = _make_data()
        res = EMCalibrator(q, model_type="DINA", max_iter=300).fit(x)
        assert res.n_missing == 0
        assert res.slip.mean() == pytest.approx(0.105, abs=0.02)
        assert res.converged


# ====================================================================== #
# 2. 缺失行 / 缺失列的行为
# ====================================================================== #
class TestDegeneratePatterns:
    def test_all_missing_examinee_is_ignored(self):
        """全缺失考生的责任向量等于先验，且不影响固定先验下的参数。"""
        q, x, _ = _make_data(n=400)
        blank = np.full((5, q.n_items), np.nan)
        x2 = np.vstack([x, blank])

        base = EMCalibrator(q, model_type="DINA", max_iter=200, estimate_prior=False).fit(x)
        with_blank = EMCalibrator(
            q, model_type="DINA", max_iter=200, estimate_prior=False
        ).fit(x2)

        assert with_blank.n_missing == blank.size
        np.testing.assert_allclose(with_blank.slip, base.slip, atol=1e-10)
        np.testing.assert_allclose(with_blank.guess, base.guess, atol=1e-10)

    def test_all_missing_examinee_posterior_equals_prior(self):
        q, x, _ = _make_data(n=200)
        cal = EMCalibrator(q, model_type="DINA")
        p = cal._prob_matrix(cal._initial_params(None, None))
        rows = np.vstack([x[:1], np.full((1, q.n_items), np.nan)])
        resp = cal._e_step(rows, p, ~np.isnan(rows))
        np.testing.assert_allclose(resp[1], cal.prior, atol=1e-12)
        np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-12)

        # 防呆：即使调用方忘了传 observed，也应按 NaN 自动识别，
        # 而不是退回"缺失=答错"
        auto = cal._e_step(rows, p)
        np.testing.assert_allclose(auto, resp, atol=1e-12)

    def test_fully_missing_item_warns_but_is_finite(self):
        q, x, _ = _make_data(n=300, j=20)
        x[:, 3] = np.nan
        with pytest.warns(RuntimeWarning, match="全部作答缺失"):
            res = EMCalibrator(q, model_type="DINA", max_iter=150).fit(x)
        assert np.isfinite(res.slip).all() and np.isfinite(res.guess).all()

    def test_input_matrix_not_modified(self):
        """fit 不应把调用方的 NaN 悄悄改成 0。"""
        q, x, rng = _make_data(n=100, j=10)
        xm = _drop(x, rng, 0.2)
        before = xm.copy()
        EMCalibrator(q, model_type="DINA", max_iter=50).fit(xm)
        np.testing.assert_array_equal(np.isnan(xm), np.isnan(before))


# ====================================================================== #
# 3. 数值正确性：与暴力实现对照
# ====================================================================== #
def _brute_posterior(q, p_mat, row, prior):
    """只对观测题连乘的逐模式后验，朴实现。"""
    j = q.n_items
    ll = np.log(prior + 1e-300).copy()
    for item in range(j):
        v = row[item]
        if np.isnan(v):
            continue
        # p_mat 的行是模式，列是题目
        probs = p_mat[:, item] if v == 1 else (1.0 - p_mat[:, item])
        ll += np.log(np.clip(probs, 1e-10, 1 - 1e-10))
    ll -= ll.max()
    z = np.exp(ll)
    return z / z.sum()


class TestNumerics:
    def test_e_step_matches_bruteforce(self):
        q, x, rng = _make_data(n=60, j=12)
        xm = _drop(x, rng, 0.35)
        cal = EMCalibrator(q, model_type="DINA")
        p = cal._prob_matrix(cal._initial_params(None, None))
        got = cal._e_step(xm, p, ~np.isnan(xm))
        for i in range(xm.shape[0]):
            want = _brute_posterior(q, p, xm[i], cal.prior)
            np.testing.assert_allclose(got[i], want, atol=1e-10)

    def test_log_likelihood_matches_bruteforce(self):
        q, x, rng = _make_data(n=40, j=10)
        xm = _drop(x, rng, 0.3)
        cal = EMCalibrator(q, model_type="DINA")
        p = cal._prob_matrix(cal._initial_params(None, None))
        got = cal._log_likelihood(xm, p, ~np.isnan(xm))

        want = 0.0
        for i in range(xm.shape[0]):
            pats_ll = np.log(cal.prior + 1e-300).copy()
            for item in range(q.n_items):
                v = xm[i, item]
                if np.isnan(v):
                    continue
                col = p[:, item] if v == 1 else 1.0 - p[:, item]
                pats_ll += np.log(np.clip(col, 1e-10, 1 - 1e-10))
            m = pats_ll.max()
            want += m + np.log(np.exp(pats_ll - m).sum())
        assert got == pytest.approx(want, abs=1e-8)

    def test_no_nan_when_person_has_one_response(self):
        """只剩一道题作答的极端情形也要给出合法后验。"""
        q, x, _ = _make_data(n=200)
        cal = EMCalibrator(q, model_type="DINA")
        p = cal._prob_matrix(cal._initial_params(None, None))
        row = np.full((1, q.n_items), np.nan)
        row[0, 0] = 1.0
        resp = cal._e_step(row, p, ~np.isnan(row))
        assert np.isfinite(resp).all()
        assert resp.sum() == pytest.approx(1.0)


# ====================================================================== #
# 4. 六个模型都要能吃 NaN
# ====================================================================== #
@pytest.mark.parametrize(
    "model", ["DINA", "DINO", "G-DINA", "RRUM", "ACDM", "LLM"]
)
def test_all_models_tolerate_missing(model):
    q, x, rng = _make_data(seed=11, n=600, j=20)
    xm = _drop(x, rng, 0.2)
    res = EMCalibrator(q, model_type=model, max_iter=80).fit(xm)

    assert res.n_missing == int(np.isnan(xm).sum())
    assert np.isfinite(res.slip).all(), model
    assert np.isfinite(res.guess).all(), model
    assert np.isfinite(res.log_likelihood)
    assert "缺失作答" in res.summary()


def test_missing_count_in_summary():
    q, x, rng = _make_data(n=100, j=10)
    xm = _drop(x, rng, 0.1)
    res = EMCalibrator(q, model_type="DINA", max_iter=60).fit(xm)
    assert res.n_missing == int(np.isnan(xm).sum())
    assert f"{res.n_missing} 个单元格" in res.summary()
