"""认知诊断响应模型 (CDM)。

所有模型统一接口：

- ``prob_matrix()``          -> ndarray (2^K, J)，P(X_j=1 | α)
- ``likelihood(patterns, responses)`` -> ndarray (n_patterns,)，似然
- ``log_likelihood(...)``    -> ndarray (n_patterns,)
- ``simulate(alpha, rng)``   -> ndarray (J,)，按模型生成作答

包含模型
--------
- ``DINA``   确定性输入噪声与门 (连接型, 每题 2 参数: slip, guess)
- ``DINO``   确定性输入噪声或门 (析取型, 每题 2 参数: slip, guess)
- ``GDINA``  一般化 DINA (每题 2^(K_j*) 参数, 表达力最强)
- ``RRUM``   降秩统一模型 (连接型, 含 η 参数)
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np

from ..qmatrix import QMatrix

__all__ = ["CDModel", "DINA", "DINO", "GDINA", "RRUM", "ACDM", "LLM", "create_model"]


class CDModel:
    """认知诊断模型基类。"""

    #: 模型名（小写）
    name: str = "base"
    #: 该模型的理想反应模式
    ideal_mode: str = "conjunctive"
    #: 模型参数名列表
    param_names: tuple = ()

    def __init__(self, q_matrix: QMatrix) -> None:
        if not isinstance(q_matrix, QMatrix):
            q_matrix = QMatrix(q_matrix)
        self.q = q_matrix
        self._prob_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # 子类必须实现
    # ------------------------------------------------------------------ #
    def prob_matrix(self) -> np.ndarray:
        """返回 P(X_j=1 | α) 矩阵，形状 ``(2^K, J)``。"""
        raise NotImplementedError

    def get_params(self) -> Dict[str, np.ndarray]:
        """导出模型参数。"""
        raise NotImplementedError

    def set_params(self, **params) -> "CDModel":
        """设置模型参数。"""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 由概率矩阵派生的通用方法
    # ------------------------------------------------------------------ #
    def clear_cache(self) -> None:
        self._prob_cache = None

    def _get_prob(self) -> np.ndarray:
        if self._prob_cache is None:
            p = np.asarray(self.prob_matrix(), dtype=np.float64)
            # 数值稳定：裁剪到 (eps, 1-eps)
            eps = 1e-9
            self._prob_cache = np.clip(p, eps, 1.0 - eps)
        return self._prob_cache

    @property
    def n_patterns(self) -> int:
        return self.q.n_patterns

    @property
    def n_items(self) -> int:
        return self.q.n_items

    def probability(self, pattern_idx: int, item_idx: int) -> float:
        """单个 (模式, 题目) 的答对概率。"""
        return float(self._get_prob()[pattern_idx, item_idx])

    # ------------------------------------------------------------------ #
    # 似然与模拟
    # ------------------------------------------------------------------ #
    def log_likelihood(self, responses: np.ndarray, item_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """给定作答，返回每个属性模式的对数似然。

        Parameters
        ----------
        responses : array-like, shape (n_answered,)
            0/1 作答向量。
        item_indices : array-like, optional
            每题对应的题目索引；默认按 0..n_answered-1 顺序。

        Returns
        -------
        ndarray, shape (n_patterns,)
        """
        responses = np.asarray(responses, dtype=np.float64).ravel()
        if item_indices is None:
            item_indices = np.arange(len(responses))
        item_indices = np.asarray(item_indices, dtype=np.int64).ravel()
        if len(responses) != len(item_indices):
            raise ValueError("responses 与 item_indices 长度必须一致")

        p = self._get_prob()[:, item_indices]                 # (n_patterns, n_ans)
        r = responses[None, :]                                # (1, n_ans)
        ll = r * np.log(p) + (1.0 - r) * np.log(1.0 - p)
        return ll.sum(axis=1)

    def likelihood(self, responses: np.ndarray, item_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """给定作答，返回每个属性模式的似然 (未归一化)。"""
        ll = self.log_likelihood(responses, item_indices)
        ll = ll - ll.max()          # 防止指数溢出
        return np.exp(ll)

    def simulate(
        self,
        alpha: np.ndarray,
        item_indices: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """按模型为一个属性模式生成作答向量。

        Parameters
        ----------
        alpha : array-like, shape (K,)
            真实属性掌握模式。
        item_indices : array-like, optional
            要作答的题目索引；默认全部题目。
        rng : numpy Generator, optional

        Returns
        -------
        ndarray, shape (n_items_simulated,), dtype=int8
        """
        if rng is None:
            rng = np.random.default_rng()
        alpha = np.asarray(alpha).ravel()
        pattern_idx = int(sum(int(a) << (self.q.n_attrs - 1 - i) for i, a in enumerate(alpha)))
        if item_indices is None:
            item_indices = np.arange(self.n_items)
        item_indices = np.asarray(item_indices, dtype=np.int64).ravel()

        probs = self._get_prob()[pattern_idx, item_indices]
        draws = rng.random(len(item_indices))
        return (draws < probs).astype(np.int8)

    def probability_for_alpha(self, alpha: np.ndarray, item_idx: int) -> float:
        """指定属性模式在某题上的答对概率。"""
        alpha = np.asarray(alpha).ravel()
        pattern_idx = int(sum(int(a) << (self.q.n_attrs - 1 - i) for i, a in enumerate(alpha)))
        return float(self._get_prob()[pattern_idx, item_idx])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_items={self.n_items}, n_attrs={self.q.n_attrs})"


# ====================================================================== #
# DINA
# ====================================================================== #
class DINA(CDModel):
    """确定性输入噪声与门模型。

    P(X_j=1 | α) = (1-s_j)^{η_ij} · g_j^{1-η_ij}

    其中 η_ij 为连接型理想反应（掌握全部所需属性则为 1）。
    ``s_j`` = slip（失误概率），``g_j`` = guess（猜测概率），
    且要求每个被试有 ``P(correct|mastered) > P(correct|not mastered)``，
    即 ``1 - s_j > g_j``。
    """

    name = "DINA"
    ideal_mode = "conjunctive"
    param_names = ("slip", "guess")

    def __init__(self, q_matrix: QMatrix, slip=None, guess=None) -> None:
        super().__init__(q_matrix)
        j = self.n_items
        self.slip = np.full(j, 0.1) if slip is None else np.asarray(slip, dtype=np.float64).ravel()
        self.guess = np.full(j, 0.1) if guess is None else np.asarray(guess, dtype=np.float64).ravel()
        self._check_params()

    def _check_params(self) -> None:
        if len(self.slip) != self.n_items or len(self.guess) != self.n_items:
            raise ValueError(f"参数长度需为 {self.n_items}")
        if np.any((self.slip < 0) | (self.slip > 1)) or np.any((self.guess < 0) | (self.guess > 1)):
            raise ValueError("slip / guess 必须在 [0, 1] 区间")

    def prob_matrix(self) -> np.ndarray:
        eta = self.q.ideal_response("conjunctive")           # (2^K, J)
        p = (1.0 - self.slip)[None, :] ** eta * self.guess[None, :] ** (1 - eta)
        return p

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"slip": self.slip.copy(), "guess": self.guess.copy()}

    def set_params(self, slip=None, guess=None, **kwargs) -> "DINA":
        if slip is not None:
            self.slip = np.asarray(slip, dtype=np.float64).ravel()
        if guess is not None:
            self.guess = np.asarray(guess, dtype=np.float64).ravel()
        self._check_params()
        self.clear_cache()
        return self

    @property
    def discriminating(self) -> np.ndarray:
        """题目区分度指标 1 - slip - guess，越大越好。"""
        return 1.0 - self.slip - self.guess


# ====================================================================== #
# DINO
# ====================================================================== #
class DINO(CDModel):
    """确定性输入噪声或门模型。

    P(X_j=1 | α) = (1-s_j)^{ω_ij} · g_j^{1-ω_ij}

    其中 ω_ij 为析取型理想反应（掌握任一所需属性即为 1）。
    """

    name = "DINO"
    ideal_mode = "disjunctive"
    param_names = ("slip", "guess")

    def __init__(self, q_matrix: QMatrix, slip=None, guess=None) -> None:
        super().__init__(q_matrix)
        j = self.n_items
        self.slip = np.full(j, 0.1) if slip is None else np.asarray(slip, dtype=np.float64).ravel()
        self.guess = np.full(j, 0.1) if guess is None else np.asarray(guess, dtype=np.float64).ravel()
        self._check_params()

    def _check_params(self) -> None:
        if len(self.slip) != self.n_items or len(self.guess) != self.n_items:
            raise ValueError(f"参数长度需为 {self.n_items}")
        if np.any((self.slip < 0) | (self.slip > 1)) or np.any((self.guess < 0) | (self.guess > 1)):
            raise ValueError("slip / guess 必须在 [0, 1] 区间")

    def prob_matrix(self) -> np.ndarray:
        omega = self.q.ideal_response("disjunctive")
        return (1.0 - self.slip)[None, :] ** omega * self.guess[None, :] ** (1 - omega)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"slip": self.slip.copy(), "guess": self.guess.copy()}

    def set_params(self, slip=None, guess=None, **kwargs) -> "DINO":
        if slip is not None:
            self.slip = np.asarray(slip, dtype=np.float64).ravel()
        if guess is not None:
            self.guess = np.asarray(guess, dtype=np.float64).ravel()
        self._check_params()
        self.clear_cache()
        return self

    @property
    def discriminating(self) -> np.ndarray:
        return 1.0 - self.slip - self.guess


# ====================================================================== #
# G-DINA
# ====================================================================== #
class GDINA(CDModel):
    """一般化 DINA 模型（logit 链接的饱和参数化）。

    对第 j 题，令其考查属性集为 K_j*（大小 r_j），共有 2^{r_j} 个潜在组。
    本实现用 **指示函数基**（indicator basis）表达每个潜在组：

        logit[P(X_j=1 | α)] = λ_{j,0} + Σ_{∅≠S⊆K_j*} λ_{j,S} · 1[α 在 S 上全部掌握]

    其中 ``λ_{j,0}`` 为 baseline（``beta``），``λ_{j,S}`` 为各交互项系数
    （``gamma``，按 mask 顺序排列）。当某个模式掌握 S 中全部属性时，
    对应的指示项为 1。

    这样每个潜在组的答对概率完全由参数决定，DINA 与 DINO 都是其特例：
    - DINA：只有全掌握组概率升高
    - DINO：只要掌握任一属性组概率就升高

    ``from_dina`` 通过求解线性方程组反解出与 DINA 完全等价的 λ 参数。
    """

    name = "G-DINA"
    ideal_mode = "conjunctive"
    param_names = ("beta", "gamma")

    def __init__(self, q_matrix: QMatrix, beta=None, gamma=None) -> None:
        super().__init__(q_matrix)
        j = self.n_items
        self.beta = np.full(j, -1.5) if beta is None else np.asarray(beta, dtype=np.float64).ravel()
        if gamma is None:
            self.gamma = [np.zeros((1 << int(self.q.matrix[i].sum())) - 1) for i in range(j)]
        else:
            self.gamma = [np.asarray(g, dtype=np.float64).ravel() for g in gamma]
        self._check_params()

    def _check_params(self) -> None:
        if len(self.beta) != self.n_items:
            raise ValueError(f"beta 长度需为 {self.n_items}")
        if len(self.gamma) != self.n_items:
            raise ValueError(f"gamma 长度需为 {self.n_items}")
        for i, g in enumerate(self.gamma):
            expected = (1 << int(self.q.matrix[i].sum())) - 1
            if len(g) != expected:
                raise ValueError(
                    f"第 {i} 题 gamma 长度应为 {expected}（考查属性数 "
                    f"{int(self.q.matrix[i].sum())}），实际为 {len(g)}"
                )

    def _logit_matrix(self) -> np.ndarray:
        """构造 logit[P(X_j=1|α)] 矩阵，形状 (2^K, J)。"""
        patterns = self.q.attribute_patterns().astype(np.float64)
        j = self.n_items
        logits = np.tile(self.beta[None, :], (self.n_patterns, 1))

        for i in range(j):
            req = np.where(self.q.matrix[i] == 1)[0]
            if len(req) == 0:
                continue
            sub = patterns[:, req]                                # (2^K, r)
            r = len(req)
            for m in range(1, 1 << r):
                cols = [b for b in range(r) if (m >> b) & 1]
                # 指示函数：该子集内全部属性都被掌握
                indicator = sub[:, cols].astype(bool).all(axis=1).astype(np.float64)
                logits[:, i] += self.gamma[i][m - 1] * indicator
        return logits

    def _group_design(self, item_idx: int) -> np.ndarray:
        """返回第 j 题的潜在组设计矩阵 X，形状 (2^r, r+1)。

        行对应各潜在组（r 位二进制，按 mask 顺序），
        首列为 1（baseline），后续列为各交互项的指示值。
        解 ``X @ theta = logit(P_group)`` 即可得到 lambda 参数。
        """
        r = int(self.q.matrix[item_idx].sum())
        if r == 0:
            return np.ones((1, 1))
        groups = ((np.arange(1 << r)[:, None] >> np.arange(r - 1, -1, -1)) & 1).astype(np.float64)
        x = np.ones((1 << r, 1 << r))
        for m in range(1, 1 << r):
            cols = [b for b in range(r) if (m >> b) & 1]
            x[:, m] = groups[:, cols].astype(bool).all(axis=1).astype(np.float64)
        return x

    def prob_matrix(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self._logit_matrix()))

    def get_params(self) -> Dict[str, object]:
        return {"beta": self.beta.copy(), "gamma": [g.copy() for g in self.gamma]}

    def set_params(self, beta=None, gamma=None, **kwargs) -> "GDINA":
        if beta is not None:
            self.beta = np.asarray(beta, dtype=np.float64).ravel()
        if gamma is not None:
            self.gamma = [np.asarray(g, dtype=np.float64).ravel() for g in gamma]
        self._check_params()
        self.clear_cache()
        return self

    @classmethod
    def from_dina(cls, q_matrix: QMatrix, slip, guess) -> "GDINA":
        """由 DINA 参数构造**完全等价**的 G-DINA 模型。

        DINA 的潜在组只有两类概率：未掌握全部属性 -> guess，
        掌握全部属性 -> 1 - slip。而 G-DINA 的指示函数基恰好可以精确
        表达这一阶梯函数：只需令"全掌握"那一组对应的组合项升高即可。

        做法：对第 j 题（考查 r 个属性），先写出 2^r 个潜在组的目标
        logit 值，再用设计矩阵解出 λ 参数：
        ``λ = X^{-1} · logit(P_group)``。由于 X 可逆（三角结构），
        所得 G-DINA 与 DINA 在所有模式上概率完全一致。
        """
        model = cls(q_matrix)
        slip = np.asarray(slip, dtype=np.float64).ravel()
        guess = np.asarray(guess, dtype=np.float64).ravel()

        def logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        for i in range(q_matrix.n_items):
            r = int(q_matrix.matrix[i].sum())
            if r == 0:
                model.beta[i] = logit(0.5)
                continue
            # 目标：只有"全掌握"组为 1-slip，其余组均为 guess
            p_all_mastered = 1.0 - slip[i]
            target = np.full(1 << r, logit(guess[i]))
            target[-1] = logit(p_all_mastered)     # mask = 2^r - 1 对应全掌握

            x = model._group_design(i)
            theta = np.linalg.solve(x, target)
            model.beta[i] = theta[0]
            model.gamma[i] = theta[1:]
        model.clear_cache()
        return model
        model.clear_cache()
        return model


# ====================================================================== #
# RRUM
# ====================================================================== #
class RRUM(CDModel):
    """降秩统一模型 (Reduced Reparameterized Unified Model)。

    P(X_j=1 | α) = π_j* · Π_{k ∈ K_j*} r_{jk*}^{(1-α_k)}

    其中 ``π_j*`` 为掌握全部所需属性的答对概率，``r_{jk*}`` 为缺少属性 k
    时的惩罚乘子（0 < r < 1）。这里用 logit 形式参数化以保证数值稳定。
    """

    name = "RRUM"
    ideal_mode = "conjunctive"
    param_names = ("pi_star", "r_star")

    def __init__(self, q_matrix: QMatrix, pi_star=None, r_star=None) -> None:
        super().__init__(q_matrix)
        j = self.n_items
        self.pi_star = np.full(j, 0.9) if pi_star is None else np.asarray(pi_star, dtype=np.float64).ravel()
        if r_star is None:
            r_star = []
            for i in range(j):
                r = int(self.q.matrix[i].sum())
                r_star.append(np.full(max(r, 1), 0.8))
        self.r_star = [np.asarray(x, dtype=np.float64).ravel() for x in r_star]
        self._check_params()

    def _check_params(self) -> None:
        if len(self.pi_star) != self.n_items:
            raise ValueError(f"pi_star 长度需为 {self.n_items}")
        for i, r in enumerate(self.r_star):
            expected = max(int(self.q.matrix[i].sum()), 1)
            if len(r) != expected:
                raise ValueError(f"第 {i} 题 r_star 长度应为 {expected}，实际 {len(r)}")

    def prob_matrix(self) -> np.ndarray:
        patterns = self.q.attribute_patterns().astype(np.float64)
        p = np.tile(self.pi_star[None, :].clip(1e-9, 1 - 1e-9), (self.n_patterns, 1))
        for i in range(self.n_items):
            req = np.where(self.q.matrix[i] == 1)[0]
            for pos, attr in enumerate(req):
                # (1 - alpha_k) 为 1 时应用惩罚
                missing = 1.0 - patterns[:, attr]
                p[:, i] *= self.r_star[i][pos].clip(1e-9, 1.0) ** missing
        return p

    def get_params(self) -> Dict[str, object]:
        return {"pi_star": self.pi_star.copy(), "r_star": [r.copy() for r in self.r_star]}

    def set_params(self, pi_star=None, r_star=None, **kwargs) -> "RRUM":
        if pi_star is not None:
            self.pi_star = np.asarray(pi_star, dtype=np.float64).ravel()
        if r_star is not None:
            self.r_star = [np.asarray(x, dtype=np.float64).ravel() for x in r_star]
        self._check_params()
        self.clear_cache()
        return self


# ====================================================================== #
# ACDM
# ====================================================================== #
class ACDM(CDModel):
    """加性认知诊断模型 (Additive CDM; de la Torre, 2011)。

        P(X_j=1 | α) = δ_{j0} + Σ_{k ∈ K_j*} δ_{jk} · α_k

    与 DINA 的"全或无"不同，每个掌握的属性各自贡献一份增量，因此是
    **完全补偿**的：掌握 A 或掌握 B 都能把答对概率推高，不需要两者兼备。

    约束（构造与标定时都会强制）：

    - ``δ_{jk} ≥ 0``：掌握某个属性不该降低答对概率（单调性）；
    - ``δ_{j0} + Σ_k δ_{jk} ≤ 1``：概率不越界。

    参数个数 ``1 + r_j``（``r_j`` 为该题考查属性数），介于 DINA 的 2 个
    与 G-DINA 的 ``2^{r_j}`` 个之间——这是它能同时容纳补偿性与简约性的原因。
    """

    name = "ACDM"
    ideal_mode = "conjunctive"
    param_names = ("delta0", "delta")

    def __init__(self, q_matrix: QMatrix, delta0=None, delta=None) -> None:
        super().__init__(q_matrix)
        j = self.n_items
        self.delta0 = np.full(j, 0.2) if delta0 is None else np.asarray(delta0, dtype=np.float64).ravel()
        if delta is None:
            delta = [np.full(max(int(self.q.matrix[i].sum()), 1), 0.5 / max(int(self.q.matrix[i].sum()), 1))
                     for i in range(j)]
        self.delta = [np.asarray(x, dtype=np.float64).ravel() for x in delta]
        self._check_params()

    def _check_params(self) -> None:
        if len(self.delta0) != self.n_items:
            raise ValueError(f"delta0 长度需为 {self.n_items}")
        if len(self.delta) != self.n_items:
            raise ValueError(f"delta 长度需为 {self.n_items}")
        for i, d in enumerate(self.delta):
            expected = max(int(self.q.matrix[i].sum()), 1)
            if len(d) != expected:
                raise ValueError(f"第 {i} 题 delta 长度应为 {expected}，实际 {len(d)}")

    def prob_matrix(self) -> np.ndarray:
        patterns = self.q.attribute_patterns().astype(np.float64)
        p = np.tile(self.delta0[None, :], (self.n_patterns, 1))
        for i in range(self.n_items):
            req = np.where(self.q.matrix[i] == 1)[0]
            for pos, attr in enumerate(req):
                p[:, i] += self.delta[i][pos] * patterns[:, attr]
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def get_params(self) -> Dict[str, object]:
        return {"delta0": self.delta0.copy(), "delta": [d.copy() for d in self.delta]}

    def set_params(self, delta0=None, delta=None, **kwargs) -> "ACDM":
        if delta0 is not None:
            self.delta0 = np.asarray(delta0, dtype=np.float64).ravel()
        if delta is not None:
            self.delta = [np.asarray(x, dtype=np.float64).ravel() for x in delta]
        self._check_params()
        self.clear_cache()
        return self

    @classmethod
    def from_dina(cls, q_matrix: QMatrix, slip, guess) -> "ACDM":
        """用最小二乘把 DINA 的分组概率投影到加性结构上（仅作初值用）。

        注意这不是等价变换：DINA 是"全掌握才跳变"的阶梯函数，加性结构
        表达不了这个台阶，只能给出最接近的线性近似。标定时它比随便取的
        初值收敛快得多。
        """
        model = cls(q_matrix)
        slip = np.asarray(slip, dtype=np.float64).ravel()
        guess = np.asarray(guess, dtype=np.float64).ravel()
        for i in range(q_matrix.n_items):
            r = int(q_matrix.matrix[i].sum())
            if r == 0:
                continue
            groups = ((np.arange(1 << r)[:, None] >> np.arange(r - 1, -1, -1)) & 1).astype(np.float64)
            target = np.where(groups.all(axis=1), 1.0 - slip[i], guess[i])
            design = np.column_stack([np.ones(1 << r), groups])
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            d0 = float(np.clip(coef[0], 1e-3, 1.0 - 1e-3))
            d = np.clip(coef[1:], 0.0, None)
            if d0 + d.sum() > 0.999 and d.sum() > 1e-12:
                d = d * (0.999 - d0) / d.sum()
            model.delta0[i] = d0
            model.delta[i] = d
        return model


# ====================================================================== #
# LLM
# ====================================================================== #
class LLM(CDModel):
    """线性 logistic 模型 (Linear Logistic Model; Maris, 1999)。

        logit P(X_j=1 | α) = β_{j0} + Σ_{k ∈ K_j*} β_{jk} · α_k

    这是 **LCDM（log-linear CDM）只保留主效应**时的特化形式
    (Henson, Templin & Willse, 2009)：logit 尺度上每个属性贡献一个加性项，
    因此模型是补偿的；同时 logit 链接自带值域保证，绝不会像 ACDM 那样
    出现概率越界，代价是参数的解释不如概率尺度直观。

    参数个数 ``1 + r_j``，与 ACDM 相同。
    """

    name = "LLM"
    ideal_mode = "conjunctive"
    param_names = ("beta0", "beta")

    def __init__(self, q_matrix: QMatrix, beta0=None, beta=None) -> None:
        super().__init__(q_matrix)
        j = self.n_items
        self.beta0 = np.full(j, -1.386) if beta0 is None else np.asarray(beta0, dtype=np.float64).ravel()
        if beta is None:
            beta = []
            for i in range(j):
                r = max(int(self.q.matrix[i].sum()), 1)
                # 让"全掌握"组的 logit 落在 1.4 附近（约 0.8）
                beta.append(np.full(r, (1.4 - (-1.386)) / r))
        self.beta = [np.asarray(x, dtype=np.float64).ravel() for x in beta]
        self._check_params()

    def _check_params(self) -> None:
        if len(self.beta0) != self.n_items:
            raise ValueError(f"beta0 长度需为 {self.n_items}")
        if len(self.beta) != self.n_items:
            raise ValueError(f"beta 长度需为 {self.n_items}")
        for i, b in enumerate(self.beta):
            expected = max(int(self.q.matrix[i].sum()), 1)
            if len(b) != expected:
                raise ValueError(f"第 {i} 题 beta 长度应为 {expected}，实际 {len(b)}")

    def prob_matrix(self) -> np.ndarray:
        patterns = self.q.attribute_patterns().astype(np.float64)
        logits = np.tile(self.beta0[None, :], (self.n_patterns, 1))
        for i in range(self.n_items):
            req = np.where(self.q.matrix[i] == 1)[0]
            for pos, attr in enumerate(req):
                logits[:, i] += self.beta[i][pos] * patterns[:, attr]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

    def get_params(self) -> Dict[str, object]:
        return {"beta0": self.beta0.copy(), "beta": [b.copy() for b in self.beta]}

    def set_params(self, beta0=None, beta=None, **kwargs) -> "LLM":
        if beta0 is not None:
            self.beta0 = np.asarray(beta0, dtype=np.float64).ravel()
        if beta is not None:
            self.beta = [np.asarray(x, dtype=np.float64).ravel() for x in beta]
        self._check_params()
        self.clear_cache()
        return self

    @classmethod
    def from_dina(cls, q_matrix: QMatrix, slip, guess) -> "LLM":
        """在 logit 尺度上对 DINA 的分组概率做最小二乘（仅作初值用）。"""
        def logit(p):
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            return np.log(p / (1.0 - p))

        model = cls(q_matrix)
        slip = np.asarray(slip, dtype=np.float64).ravel()
        guess = np.asarray(guess, dtype=np.float64).ravel()
        for i in range(q_matrix.n_items):
            r = int(q_matrix.matrix[i].sum())
            if r == 0:
                continue
            groups = ((np.arange(1 << r)[:, None] >> np.arange(r - 1, -1, -1)) & 1).astype(np.float64)
            target = np.where(groups.all(axis=1), logit(1.0 - slip[i]), logit(guess[i]))
            design = np.column_stack([np.ones(1 << r), groups])
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            model.beta0[i] = float(np.clip(coef[0], -20.0, 20.0))
            model.beta[i] = np.clip(coef[1:], 0.0, None)
        return model


# ====================================================================== #
# 工厂函数
# ====================================================================== #
_MODEL_REGISTRY = {
    "dina": DINA,
    "dino": DINO,
    "gdina": GDINA,
    "g-dina": GDINA,
    "rrum": RRUM,
    "acdm": ACDM,
    "additive": ACDM,
    "llm": LLM,
    "linearlogistic": LLM,
}


def _normalise_model_key(name: object) -> str:
    """模型名归一化：去掉空格 / 连字符 / 下划线并转小写。

    这样 ``"G-DINA"``、``"g dina"``、``"g_dina"``、``"gdina"`` 都指向同一模型，
    用户不必记住别名表里到底写的是哪一种写法。
    """
    return re.sub(r"[\s\-_]+", "", str(name)).strip().lower()


# 归一化后的查表（多个别名可能塌陷成同一个键，取先出现者即可）
_MODEL_LOOKUP = {_normalise_model_key(k): v for k, v in _MODEL_REGISTRY.items()}


def create_model(name: str, q_matrix: QMatrix, **params) -> CDModel:
    """按名称创建响应模型。

    Parameters
    ----------
    name : str
        模型名，可选 ``'DINA'`` / ``'DINO'`` / ``'G-DINA'`` / ``'RRUM'`` /
        ``'ACDM'`` / ``'LLM'``（大小写、空格、连字符、下划线均不敏感）。
    q_matrix : QMatrix
    **params
        传给模型构造函数的参数，如 ``slip=``, ``guess=``。

    Examples
    --------
    >>> create_model("DINA", q, slip=[0.1]*10, guess=[0.2]*10)  # doctest: +SKIP
    """
    key = _normalise_model_key(name)
    if key not in _MODEL_LOOKUP:
        raise ValueError(
            f"未知模型 '{name}'，可选: DINA / DINO / G-DINA / RRUM / ACDM / LLM"
        )
    cls = _MODEL_LOOKUP[key]

    # ACDM / LLM 的参数是 δ / β，没有 slip、guess。但仿真器、界面、流水线
    # 这些调用点普遍只持有 slip/guess 这一对，所以统一在工厂里做投影
    # （最小二乘拟合到目标模型的结构），免得每个调用点各写一遍判断。
    if cls in (ACDM, LLM) and ("slip" in params or "guess" in params):
        q = q_matrix if isinstance(q_matrix, QMatrix) else QMatrix(q_matrix)
        slip = params.pop("slip", None)
        guess = params.pop("guess", None)
        slip = np.full(q.n_items, 0.1) if slip is None else np.asarray(slip, dtype=np.float64).ravel()
        guess = np.full(q.n_items, 0.1) if guess is None else np.asarray(guess, dtype=np.float64).ravel()
        return cls.from_dina(q, slip, guess)

    return cls(q_matrix, **params)
