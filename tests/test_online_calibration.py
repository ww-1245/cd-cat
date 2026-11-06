"""新题在线标定测试。

核心是**参数能否从考生诊断结果里恢复**（相关要够高），以及**样本不足时
必须报出来**，而不是悄悄给一个看起来正常的数字。
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import (
    DINA,
    NewItemResult,
    OnlineCalibrationReport,
    QMatrix,
    calibrate,
    calibrate_new_items,
)


# ====================================================================== #
# 工具
# ====================================================================== #
def _setup(K=4, J_old=30, J_new=10, n=1200, seed=2024):
    """生成"老题已标定 + 新题待标定"的数据。"""
    rng = np.random.default_rng(seed)
    J = J_old + J_new
    m = np.zeros((J, K), dtype=int)
    for j in range(J):
        m[j, rng.choice(K, size=1 + (j % 2), replace=False)] = 1
    q = QMatrix(m, attr_names=[f"A{i + 1}" for i in range(K)])

    slip = rng.uniform(0.05, 0.20, J)
    guess = rng.uniform(0.05, 0.25, J)
    alpha = rng.integers(0, 2, size=(n, K))
    w = 1 << np.arange(K - 1, -1, -1)
    true = DINA(q, slip=slip, guess=guess)
    X = (rng.random((n, J)) < true._get_prob()[(alpha * w).sum(1)]).astype(float)

    m_old, _ = calibrate(QMatrix(m[:J_old]), X[:, :J_old], model_type="DINA")
    prod = DINA(
        q,
        slip=np.concatenate([m_old.get_params()["slip"], np.full(J_new, 0.1)]),
        guess=np.concatenate([m_old.get_params()["guess"], np.full(J_new, 0.2)]),
    )
    return q, prod, X, slip[J_old:], guess[J_old:], J_old


@pytest.fixture(scope="module")
def pack():
    return _setup()


# ====================================================================== #
# 1. 参数恢复
# ====================================================================== #
class TestRecovery:
    def test_slip_guess_recovered(self, pack):
        q, model, X, true_s, true_g, j_old = pack
        rep = calibrate_new_items(model, X, new_items=range(j_old, q.n_items))
        est_s = np.array([it.slip for it in rep.items])
        est_g = np.array([it.guess for it in rep.items])
        assert np.corrcoef(true_s, est_s)[0, 1] > 0.70
        assert np.corrcoef(true_g, est_g)[0, 1] > 0.85
        assert np.abs(true_s - est_s).mean() < 0.05
        assert np.abs(true_g - est_g).mean() < 0.05

    def test_parameters_in_range(self, pack):
        q, model, X, _, _, j_old = pack
        rep = calibrate_new_items(model, X, new_items=range(j_old, q.n_items))
        for it in rep.items:
            assert 0.0 <= it.slip <= 1.0
            assert 0.0 <= it.guess <= 1.0
            assert it.discrimination == pytest.approx(1 - it.slip - it.guess)

    def test_master_counts_consistent(self, pack):
        q, model, X, _, _, j_old = pack
        rep = calibrate_new_items(model, X, new_items=range(j_old, q.n_items))
        for it in rep.items:
            assert it.n_masters + it.n_nonmasters == pytest.approx(rep.n_examinees)
            assert it.n_masters > 0 and it.n_nonmasters > 0

    def test_beta_smoothing_shrinks_extremes(self, pack):
        """样本极少时估计应被先验收缩，不应出现 0 或 1。"""
        q, model, X, _, _, j_old = pack
        rep = calibrate_new_items(model, X[:30], new_items=range(j_old, q.n_items),
                                  min_effective_n=10)
        for it in rep.items:
            if not it.usable:
                continue
            assert 0.0 < it.slip < 1.0
            assert 0.0 < it.guess < 1.0


# ====================================================================== #
# 2. 样本不足与不可标定
# ====================================================================== #
class TestWarnings:
    def test_small_sample_flagged(self, pack):
        q, model, X, _, _, j_old = pack
        small = calibrate_new_items(model, X[:20], new_items=range(j_old, q.n_items))
        assert len(small.flagged) > 0
        assert any(it.status == "unstable" for it in small.flagged)
        assert "样本不足" in small.verdict() or "无法标定" in small.verdict()

    def test_large_sample_clean(self, pack):
        q, model, X, _, _, j_old = pack
        rep = calibrate_new_items(model, X, new_items=range(j_old, q.n_items))
        assert len(rep.flagged) == 0
        assert "全部完成标定" in rep.verdict()

    def test_no_master_unusable(self, pack):
        """没有任何考生掌握该题所需属性 → slip 无法估计。"""
        q, model, X, _, _, j_old = pack
        n = X.shape[0]
        # 全部考生都是"什么都没掌握"
        post = np.zeros((n, q.n_patterns))
        post[:, 0] = 1.0
        rep = calibrate_new_items(model, X, new_items=[j_old], alpha_hat=post)
        it = rep.items[0]
        if np.asarray(q.matrix[j_old]).sum() > 0:
            assert it.n_masters == pytest.approx(0.0)
            assert it.status == "unusable"
            assert not it.usable

    def test_all_master_unusable(self, pack):
        """所有考生都掌握全部属性 → guess 无法估计。"""
        q, model, X, _, _, j_old = pack
        n = X.shape[0]
        post = np.zeros((n, q.n_patterns))
        post[:, -1] = 1.0
        rep = calibrate_new_items(model, X, new_items=[j_old], alpha_hat=post)
        it = rep.items[0]
        assert it.n_nonmasters == pytest.approx(0.0)
        assert it.status == "unusable"


# ====================================================================== #
# 3. 输入路径与校验
# ====================================================================== #
class TestInterface:
    def test_alpha_hat_path(self, pack):
        q, model, X, _, _, j_old = pack
        n = X.shape[0]
        post = np.full((n, q.n_patterns), 1.0 / q.n_patterns)
        rep = calibrate_new_items(model, X, new_items=range(j_old, q.n_items),
                                  alpha_hat=post)
        assert rep.n_new_items == q.n_items - j_old
        for it in rep.items:
            assert 0.0 <= it.slip <= 1.0

    def test_explicit_old_items(self, pack):
        q, model, X, _, _, j_old = pack
        rep = calibrate_new_items(
            model, X, new_items=range(j_old, q.n_items),
            old_items=range(0, j_old),
        )
        assert rep.n_old_items == j_old

    def test_missing_values(self, pack):
        q, model, X, _, _, j_old = pack
        x = X.copy()
        rng = np.random.default_rng(7)
        x[rng.random(x.shape) < 0.03] = np.nan
        rep = calibrate_new_items(model, x, new_items=range(j_old, q.n_items))
        for it in rep.items:
            assert it.n_observed <= rep.n_examinees
            assert np.isfinite(it.slip) or it.status == "unusable"

    def test_outputs(self, pack):
        q, model, X, _, _, j_old = pack
        rep = calibrate_new_items(model, X, new_items=range(j_old, q.n_items))
        assert isinstance(rep, OnlineCalibrationReport)
        assert isinstance(rep.items[0], NewItemResult)

        df = rep.to_dataframe()
        assert len(df) == rep.n_new_items
        assert {"item_idx", "slip", "guess", "status"} <= set(df.columns)

        d = rep.to_dict()
        assert d["n_new_items"] == rep.n_new_items
        assert len(d["items"]) == rep.n_new_items

        text = rep.summary()
        assert "在线标定" in text
        assert rep.get(j_old).item_idx == j_old

    def test_validation(self, pack):
        q, model, X, _, _, j_old = pack
        with pytest.raises(ValueError):
            calibrate_new_items(model, X[:, :-1], new_items=[j_old])
        with pytest.raises(ValueError):
            calibrate_new_items(model, X, new_items=[])
        with pytest.raises(ValueError):
            calibrate_new_items(model, X, new_items=[999])
        with pytest.raises(ValueError):
            calibrate_new_items(model, X, new_items=[j_old], old_items=[j_old, 1])
        with pytest.raises(ValueError):
            calibrate_new_items(model, X, new_items=[j_old],
                                alpha_hat=np.ones((3, q.n_patterns)))
