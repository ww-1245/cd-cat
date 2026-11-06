"""ACDM 与 LLM 两个新增响应模型的性质测试。

重点不是"跑得通"，而是三条容易写错、又不容易被发现的性质：

1. **单调性**：多掌握一个属性，答对概率不能下降（约束有没有真的生效）；
2. **结构**：ACDM 在概率尺度线性、LLM 在 logit 尺度线性（公式有没有写错）；
3. **工厂投影**： ``create_model(..., slip=, guess=)`` 对这两个模型要能用，
   否则仿真器 / 界面这些只持有 slip/guess 的调用点会直接崩。
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import QMatrix, build_test_items, create_model
from cdcat.models import ACDM, DINA, LLM

# ====================================================================== #
# 公共夹具
# ====================================================================== #
@pytest.fixture(scope="module")
def bank():
    q = QMatrix.from_random(18, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=31)
    items = build_test_items(18, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=31)
    return q, items


def _neighbour_pairs(q):
    """列出"只在属性 k 上不同、且前者未掌握 k"的模式对 (s, t, k)。"""
    pats = q.attribute_patterns()
    for k in range(q.n_attrs):
        for s in range(q.n_patterns):
            if pats[s, k] == 0:
                yield s, s | (1 << (q.n_attrs - 1 - k)), k


# ====================================================================== #
# 概率矩阵的结构性质
# ====================================================================== #
class TestProbMatrix:
    @pytest.mark.parametrize("cls", [ACDM, LLM])
    def test_shape_and_range(self, bank, cls):
        q, items = bank
        p = cls.from_dina(q, items["slip"], items["guess"]).prob_matrix()
        assert p.shape == (q.n_patterns, q.n_items)
        assert np.isfinite(p).all()
        assert (p > 0).all() and (p < 1).all()

    @pytest.mark.parametrize("cls", [ACDM, LLM])
    def test_monotone_in_mastery(self, bank, cls):
        """多掌握一个属性，答对概率不得下降。"""
        q, items = bank
        p = cls.from_dina(q, items["slip"], items["guess"]).prob_matrix()
        for s, t, k in _neighbour_pairs(q):
            affected = np.where(q.matrix[:, k] == 1)[0]
            if len(affected) == 0:
                continue
            assert np.all(p[t, affected] >= p[s, affected] - 1e-12)

    def test_acdm_is_linear_in_probability(self, bank):
        """ACDM: P(α) = δ0 + Σ δ_k α_k，逐个模式验证。"""
        q, items = bank
        m = ACDM.from_dina(q, items["slip"], items["guess"])
        pats = q.attribute_patterns().astype(float)
        p = m.prob_matrix()
        for i in range(q.n_items):
            req = np.where(q.matrix[i] == 1)[0]
            expect = m.delta0[i] + pats[:, req] @ m.delta[i]
            expect = np.clip(expect, 1e-6, 1 - 1e-6)
            assert np.allclose(p[:, i], expect, atol=1e-10)

    def test_llm_is_linear_in_logit(self, bank):
        """LLM: logit P(α) = β0 + Σ β_k α_k。"""
        q, items = bank
        m = LLM.from_dina(q, items["slip"], items["guess"])
        pats = q.attribute_patterns().astype(float)
        p = m.prob_matrix()
        for i in range(q.n_items):
            req = np.where(q.matrix[i] == 1)[0]
            logit = m.beta0[i] + pats[:, req] @ m.beta[i]
            assert np.allclose(p[:, i], 1.0 / (1.0 + np.exp(-logit)), atol=1e-10)

    @pytest.mark.parametrize("cls", [ACDM, LLM])
    def test_single_attribute_item_is_exact(self, bank, cls):
        """只考 1 个属性的题，两模型都能精确复现 DINA 的两点分布。"""
        q = QMatrix.from_random(6, 3, min_attrs_per_item=1, max_attrs_per_item=1, seed=12)
        slip = np.full(6, 0.2)
        guess = np.full(6, 0.15)
        p = cls.from_dina(q, slip, guess).prob_matrix()
        d = DINA(q, slip=slip, guess=guess).prob_matrix()
        assert np.allclose(p, d, atol=1e-10)


# ====================================================================== #
# 参数校验与工厂
# ====================================================================== #
class TestValidation:
    def test_wrong_delta_length(self, bank):
        q, _ = bank
        with pytest.raises(ValueError, match="delta 长度"):
            ACDM(q, delta=[np.zeros(1) for _ in range(q.n_items)])

    def test_wrong_beta_length(self, bank):
        q, _ = bank
        with pytest.raises(ValueError, match="beta 长度"):
            LLM(q, beta=[np.zeros(1) for _ in range(q.n_items)])

    @pytest.mark.parametrize("name", ["ACDM", "acdm", "additive", "LLM", "llm", "linear-logistic"])
    def test_registry_aliases(self, bank, name):
        q, _ = bank
        assert create_model(name, q) in (ACDM, LLM) or isinstance(
            create_model(name, q), (ACDM, LLM)
        )

    @pytest.mark.parametrize("name", ["ACDM", "LLM"])
    def test_factory_accepts_slip_guess(self, bank, name):
        """仿真器与界面只持有 slip/guess，工厂必须能把它们投影到这两个模型。"""
        q, items = bank
        m = create_model(name, q, slip=items["slip"], guess=items["guess"])
        assert isinstance(m, (ACDM, LLM))
        assert m.prob_matrix().shape == (q.n_patterns, q.n_items)

    @pytest.mark.parametrize("name", ["ACDM", "LLM"])
    def test_factory_slip_only(self, bank, name):
        q, items = bank
        m = create_model(name, q, slip=items["slip"])
        assert np.isfinite(m.prob_matrix()).all()

    def test_unknown_model_message_lists_new_ones(self, bank):
        q, _ = bank
        with pytest.raises(ValueError, match="ACDM"):
            create_model("nope", q)
