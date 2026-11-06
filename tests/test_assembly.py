"""题库装配优化测试。

重点是**约束必须被遵守**（覆盖下限、必含题、题量）与**目标函数真的有用**
（明显优于随机抽题），而不是某个具体的最优解。
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import DINA, QMatrix, assemble_test_bank
from cdcat.selection import AssemblyResult
from cdcat.selection.assembly import _expected_accuracy, _item_kl_matrix


# ====================================================================== #
# 工具
# ====================================================================== #
def _bank(K=4, J=80, seed=3):
    rng = np.random.default_rng(seed)
    m = np.zeros((J, K), dtype=int)
    for j in range(J):
        m[j, rng.choice(K, size=min(1 + (j % 3), K), replace=False)] = 1
    q = QMatrix(m, attr_names=[f"A{i + 1}" for i in range(K)])
    model = DINA(q, slip=rng.uniform(0.05, 0.25, J), guess=rng.uniform(0.05, 0.30, J))
    return q, model


def _min_kl(model, items, n_patterns):
    prob = model._get_prob()
    acc = np.zeros((n_patterns, n_patterns))
    for j in items:
        acc += _item_kl_matrix(prob[:, j])
    iu = np.triu_indices(n_patterns, k=1)
    return float(acc[iu[0], iu[1]].min())


# ====================================================================== #
# 1. 约束
# ====================================================================== #
class TestConstraints:
    def test_length_and_uniqueness(self):
        q, model = _bank()
        res = assemble_test_bank(q, model, n_items=16, min_per_attr=2)
        assert isinstance(res, AssemblyResult)
        assert res.n_items == 16
        assert len(set(res.selected.tolist())) == 16
        assert res.selected.min() >= 0 and res.selected.max() < q.n_items

    def test_must_include_preserved(self):
        q, model = _bank()
        must = [3, 17, 42]
        res = assemble_test_bank(q, model, n_items=16, must_include=must, min_per_attr=2)
        assert set(must) <= set(res.selected.tolist())
        assert res.must_include == must

    def test_min_per_attr_satisfied(self):
        q, model = _bank()
        res = assemble_test_bank(q, model, n_items=18, min_per_attr=3)
        assert res.coverage_ok()
        assert np.all(res.attr_coverage >= 3)
        # 覆盖计数应与 Q 矩阵里入选题的属性数一致
        expect = np.asarray(q.matrix)[res.selected].sum(axis=0)
        assert np.array_equal(res.attr_coverage, expect)

    def test_max_per_attr_respected(self):
        q, model = _bank(K=4, J=60)
        res = assemble_test_bank(q, model, n_items=12, min_per_attr=1, max_per_attr=8)
        assert np.all(res.attr_coverage <= 8)

    def test_infeasible_cap_raises(self):
        q, model = _bank(K=4, J=60)
        # 12 题 × 每属性至多 2 题 = 8 个属性位 < 12，必要条件就不成立
        with pytest.raises(ValueError, match="不可行"):
            assemble_test_bank(q, model, n_items=12, min_per_attr=1, max_per_attr=2)
        # 20 题 × 至多 4 题 = 16 < 20
        with pytest.raises(ValueError, match="不可行"):
            assemble_test_bank(q, model, n_items=20, min_per_attr=1, max_per_attr=4)

    def test_attr_share_sums_to_one(self):
        q, model = _bank()
        res = assemble_test_bank(q, model, n_items=16, min_per_attr=2)
        assert res.attr_share.sum() == pytest.approx(1.0)

    def test_default_length(self):
        q, model = _bank(K=4, J=80)
        res = assemble_test_bank(q, model)
        # 默认 max(2K, K*min_per_attr) 且不超过题库
        assert res.n_items == max(2 * q.n_attrs, q.n_attrs)


# ====================================================================== #
# 2. 优化效果
# ====================================================================== #
class TestOptimisation:
    def test_beats_random_forms(self):
        q, model = _bank()
        res = assemble_test_bank(q, model, n_items=16, min_per_attr=2)
        rng = np.random.default_rng(0)
        rand = np.array([
            _min_kl(model, rng.choice(q.n_items, 16, replace=False), q.n_patterns)
            for _ in range(150)
        ])
        assert res.score > rand.mean()
        # 应明显超过随机抽题的"最好"一份，否则这个优化就没意义
        assert res.score > rand.max()

    def test_local_search_never_hurts(self):
        q, model = _bank(K=5, J=120, seed=11)
        a = assemble_test_bank(q, model, n_items=20, min_per_attr=2, local_search=False)
        b = assemble_test_bank(q, model, n_items=20, min_per_attr=2, local_search=True)
        assert b.score >= a.score - 1e-9

    def test_expected_pmr_improves(self):
        q, model = _bank()
        res = assemble_test_bank(q, model, n_items=16, min_per_attr=2)
        prob = model._get_prob()
        prior = np.full(q.n_patterns, 1.0 / q.n_patterns)
        assert 0.0 < res.expected_pmr <= 1.0
        rng = np.random.default_rng(1)
        baseline = np.mean([
            _expected_accuracy(prob, np.sort(rng.choice(q.n_items, 16, replace=False)),
                               prior)[0]
            for _ in range(60)
        ])
        assert res.expected_pmr > baseline

    def test_more_items_do_not_reduce_objective(self):
        q, model = _bank()
        small = assemble_test_bank(q, model, n_items=10, min_per_attr=1)
        large = assemble_test_bank(q, model, n_items=20, min_per_attr=1)
        assert large.score >= small.score

    def test_history_monotone(self):
        q, model = _bank()
        res = assemble_test_bank(q, model, n_items=14, min_per_attr=2)
        h = res.history
        assert len(h) >= 2
        # 只有加入/交换确实带来的改进才会被记录，因此不应下降
        assert all(h[i + 1] >= h[i] - 1e-9 for i in range(len(h) - 1))


# ====================================================================== #
# 3. 输出与校验
# ====================================================================== #
class TestOutputs:
    def test_outputs(self):
        q, model = _bank()
        res = assemble_test_bank(q, model, n_items=12, min_per_attr=2)
        df = res.to_dataframe()
        assert len(df) == 12
        assert {"item_idx", "item"} <= set(df.columns)

        d = res.to_dict()
        assert d["n_items"] == 12
        assert len(d["selected"]) == 12
        assert len(d["expected_amr"]) == q.n_attrs
        assert all(0.0 <= v <= 1.0 for v in d["expected_amr"].values())

        text = res.summary()
        assert "题库装配" in text
        assert "理论判准率上限 PMR" in text

    def test_validation(self):
        q, model = _bank(K=4, J=40)
        with pytest.raises(ValueError):
            assemble_test_bank(q, model, n_items=0)
        with pytest.raises(ValueError):
            assemble_test_bank(q, model, n_items=999)
        with pytest.raises(ValueError):
            assemble_test_bank(q, model, n_items=5, must_include=[1, 2, 3, 4, 5, 6])
        with pytest.raises(ValueError):
            assemble_test_bank(q, model, n_items=5, must_include=[999])
        with pytest.raises(ValueError):
            assemble_test_bank(q, model, n_items=5, prior=np.ones(3))
        with pytest.raises(ValueError):
            assemble_test_bank(q, model, n_items=5, min_per_attr=2, max_per_attr=1)
        # 模型与 Q 矩阵题量不符
        other = DINA(QMatrix(np.ones((10, 2), dtype=int)))
        with pytest.raises(ValueError):
            assemble_test_bank(q, other, n_items=5)

    def test_attribute_without_enough_items(self):
        # 属性 3 只有 1 道题，却要求每个属性至少 2 题
        m = np.zeros((10, 4), dtype=int)
        m[0:4] = [1, 0, 0, 0]
        m[4:7] = [0, 1, 0, 0]
        m[7:9] = [0, 0, 1, 0]
        m[9] = [0, 0, 0, 1]
        q = QMatrix(m)
        with pytest.raises(ValueError, match="min_per_attr"):
            assemble_test_bank(q, DINA(q), n_items=6, min_per_attr=2)
