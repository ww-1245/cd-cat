"""三个非参数选题准则（NPSHE / ERHD / AMI）的测试。

这些准则最容易出的错不是崩溃，而是**算错但看不出来**：熵、汉明距离、
边际概率都很容易少归一化一次或漏掉一支后验，结果照样是有限的正数。
所以这里用三类断言把数学钉死：

1. **对照暴力实现**：逐题逐分支重算，必须与向量化实现逐位一致；
2. **语义**：能把这个后验劈开的题，得分必须高于"对所有模式理想反应相同"的题；
3. **不依赖题目参数**：换 slip/guess 结果必须完全不变，否则就不是非参数准则。
"""

from __future__ import annotations

import numpy as np
import pytest

from cdcat import DINA, QMatrix
from cdcat.selection import (
    SELECTION_REGISTRY,
    AMI,
    ERHD,
    MIV,
    NPSHE,
    create_selector,
)

# ====================================================================== #
# 夹具：一个 6 题 / 3 属性的小题库，覆盖 1~2 个属性的题目
# ====================================================================== #
@pytest.fixture(scope="module")
def setup():
    q = QMatrix.from_random(6, 3, min_attrs_per_item=1, max_attrs_per_item=2, seed=8)
    model = DINA(q, slip=np.full(6, 0.15), guess=np.full(6, 0.20))
    rng = np.random.default_rng(2)
    post = rng.random(q.n_patterns)
    post /= post.sum()
    return q, model, post


# ====================================================================== #
# 暴力参照实现
# ====================================================================== #
def _entropy(d: np.ndarray) -> float:
    d = d[d > 0]
    return float(-(d * np.log(d)).sum())


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def _update(post: np.ndarray, eta_col: np.ndarray, value: int):
    """给定理想反应列与观察值，返回 (分支概率, 归一化后验)。"""
    mask = (eta_col > 0.5) if value == 1 else (eta_col <= 0.5)
    w = float(post[mask].sum())
    if w <= 0:
        return 0.0, np.zeros_like(post)
    return w, post * mask / w


def _brute(q, post, kind: str) -> np.ndarray:
    eta = q.ideal_response("conjunctive")
    pats = q.attribute_patterns().astype(float)
    out = []
    for j in range(q.n_items):
        branch = []
        for value in (1, 0):
            w, p = _update(post, eta[:, j], value)
            if kind == "NPSHE":
                branch.append((w, _entropy(p)))
            elif kind == "AMI":
                marg = [float(p[pats[:, k] == 1].sum()) for k in range(q.n_attrs)]
                branch.append((w, sum(_binary_entropy(m) for m in marg)))
            else:  # ERHD
                mode = int(np.argmax(p))
                dist = float(sum(p[a] * np.abs(pats[a] - pats[mode]).sum() for a in range(len(p))))
                branch.append((w, dist))
        if kind == "NPSHE":
            cur = _entropy(post)
        elif kind == "AMI":
            marg = [float(post[pats[:, k] == 1].sum()) for k in range(q.n_attrs)]
            cur = sum(_binary_entropy(m) for m in marg)
        else:
            mode = int(np.argmax(post))
            cur = float(sum(post[a] * np.abs(pats[a] - pats[mode]).sum() for a in range(len(post))))
        out.append(cur - sum(w * v for w, v in branch))
    return np.array(out)


# ====================================================================== #
# 1. 与暴力实现一致
# ====================================================================== #
class TestAgainstBruteForce:
    @pytest.mark.parametrize("cls,kind", [(NPSHE, "NPSHE"), (AMI, "AMI"), (ERHD, "ERHD")])
    def test_matches_bruteforce(self, setup, cls, kind):
        q, model, post = setup
        got = cls(model).compute(post, np.arange(q.n_items))
        ref = _brute(q, post, kind)
        assert np.allclose(got, ref, atol=1e-10), f"{kind} 与暴力实现不一致"

    @pytest.mark.parametrize("cls", [NPSHE, AMI, ERHD])
    def test_nonnegative(self, setup, cls):
        """信息量不该为负：观察数据不会让后验"变差"。"""
        q, model, post = setup
        vals = cls(model).compute(post, np.arange(q.n_items))
        assert (vals >= -1e-9).all()


# ====================================================================== #
# 2. 语义：有区分度 vs 无区分度
# ====================================================================== #
class TestSemantics:
    def _semantic_setup(self):
        """只用 1 个属性的 2 道题：题 0 考属性 0，题 1 也考属性 0？

        改为手工构造 Q：题 0 考属性 0（能把后验劈开），
        题 1 考属性 2（对只在属性 0 上不同的两个模式完全无区分度）。
        """
        q = QMatrix(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.int8),
                    attr_names=["A", "B", "C"])
        model = DINA(q, slip=np.full(2, 0.1), guess=np.full(2, 0.1))
        pats = q.attribute_patterns()
        # 后验只压在 000 与 100 上（差在属性 A）
        post = np.zeros(q.n_patterns)
        for idx in range(q.n_patterns):
            if tuple(pats[idx][1:]) == (0, 0):
                post[idx] = 0.5
        return q, model, post

    @pytest.mark.parametrize("cls", [NPSHE, ERHD, AMI])
    def test_discriminating_item_wins(self, cls):
        q, model, post = self._semantic_setup()
        vals = cls(model).compute(post, np.arange(2))
        assert vals[0] > 0.05, f"{cls.__name__}: 能区分的题得分应明显为正"
        assert abs(vals[1]) < 1e-9, f"{cls.__name__}: 无区分度的题得分应为 0"

    @pytest.mark.parametrize("cls", [NPSHE, ERHD, AMI])
    def test_select_returns_discriminating(self, cls):
        q, model, post = self._semantic_setup()
        assert cls(model).select(post) == 0


# ====================================================================== #
# 3. 非参数性：与题目参数无关
# ====================================================================== #
class TestParameterFree:
    @pytest.mark.parametrize("cls", [NPSHE, AMI, ERHD, MIV])
    def test_independent_of_slip_guess(self, setup, cls):
        q, _, post = setup
        a = cls(DINA(q, slip=np.full(6, 0.05), guess=np.full(6, 0.05)))
        b = cls(DINA(q, slip=np.full(6, 0.45), guess=np.full(6, 0.40)))
        va = a.compute(post, np.arange(6))
        vb = b.compute(post, np.arange(6))
        assert np.array_equal(va, vb), f"{cls.__name__} 竟然受题目参数影响"

    @pytest.mark.parametrize("cls", [NPSHE, AMI, ERHD])
    def test_ideal_mode_follows_model(self, setup, cls):
        """DINO 模型的理想反应是析取，准则应跟着模型走。"""
        q, _, post = setup
        from cdcat import DINO

        dino = DINO(q, slip=np.full(6, 0.1), guess=np.full(6, 0.1))
        explicit = cls(dino, ideal_mode="disjunctive")
        implicit = cls(dino)
        vals = implicit.compute(post, np.arange(6))
        assert np.allclose(vals, explicit.compute(post, np.arange(6)))


# ====================================================================== #
# 4. 边界情形
# ====================================================================== #
class TestEdgeCases:
    def test_branch_with_zero_mass(self):
        """后验全压在一类理想反应上时，另一分支概率为 0，不得产生 NaN。"""
        q = QMatrix(np.array([[1, 0]], dtype=np.int8))
        model = DINA(q, slip=np.array([0.1]), guess=np.array([0.1]))
        pats = q.attribute_patterns()
        idx = int(np.where(pats[:, 0] == 1)[0][0])
        post = np.zeros(q.n_patterns)
        post[idx] = 1.0
        for cls in (NPSHE, AMI, ERHD):
            vals = cls(model).compute(post, np.arange(1))
            assert np.isfinite(vals).all()
            # 该题已无信息可挖：两个分支只剩一个，判据必须为 0
            assert abs(float(vals[0])) < 1e-9

    def test_single_candidate(self, setup):
        q, model, post = setup
        assert ERHD(model).compute(post, np.array([3])).shape == (1,)

    def test_unnormalised_posterior(self, setup):
        """传进来的后验没归一化也应得到同样结果（内部要归一化）。"""
        q, model, post = setup
        for cls in (NPSHE, AMI, ERHD):
            a = cls(model).compute(post, np.arange(6))
            b = cls(model).compute(post * 37.0, np.arange(6))
            assert np.allclose(a, b, atol=1e-10)


# ====================================================================== #
# 5. 注册表与仿真器集成
# ====================================================================== #
class TestRegistryAndSimulation:
    @pytest.mark.parametrize("name", ["NPSHE", "ERHD", "AMI"])
    def test_registry(self, setup, name):
        q, model, _ = setup
        assert name in SELECTION_REGISTRY
        assert isinstance(create_selector(name, model), SELECTION_REGISTRY[name])

    @pytest.mark.parametrize("name", ["NPSHE", "ERHD", "AMI"])
    def test_runs_inside_simulator(self, name):
        """在真实 CD-CAT 循环里跑一遍：必须能完成、且优于随机猜测。"""
        from cdcat import CDCATSimulator, build_test_items

        q = QMatrix.from_random(20, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=4)
        items = build_test_items(20, 4, min_attrs_per_item=1, max_attrs_per_item=2, seed=4)
        sim = CDCATSimulator(
            q, model="DINA",
            model_params={"slip": items["slip"], "guess": items["guess"]},
            selector=name, max_items=20, min_items=5,
        )
        res = sim.run(n_examinees=40, n_replications=1, seed=0)
        assert res.records, "未产出任何施测记录"
        assert res.pmr() > 0.25, f"{name} 的 PMR 只有 {res.pmr():.3f}，接近随机"
