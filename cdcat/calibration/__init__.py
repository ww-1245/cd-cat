"""题目参数标定：MMLE/EM 算法。

在真实 CD-CAT 应用中，Q 矩阵给定后需要用**已有的大样本作答数据**估计
题目参数。本模块用**边际极大似然估计 (MMLE) 的 EM 算法**实现六类模型：

E 步（所有模型共用）：由当前参数与作答数据，计算每个被试属于每个属性
模式的后验概率（"后验期望计数"）。

    h_i(α) = P(α | X_i) ∝ π(α) · Π_j P(X_ij | α)

M 步（按模型解析或半解析更新）：

- **DINA / DINO**（每题 2 参数 slip, guess）：闭式解析更新
  ``s_j = Σ_i [h_i(α:η_j=1)(1-X_ij)] / Σ_i h_i(α:η_j=1)``
- **G-DINA**（每题 2^{r_j} 参数）：对每个潜在组，用该组责任之和与该组
  答对次数直接得到组概率 ``P_g = Σ h·X / Σ h``，再经 ``logit`` 转换即得
  ``λ`` 参数（指示函数基下组到参数是一一对应的）。
- **RRUM**（每题 1 + r_j 参数）：``π*`` 与各属性惩罚 ``r*`` 用期望计数
  的比值更新。
- **ACDM**（每题 1 + r_j 参数）：加性结构在概率尺度上线性，M 步是
  分组加权最小二乘，有闭式解。
- **LLM**（每题 1 + r_j 参数）：logit 尺度上的主效应模型，M 步是分组
  二项 logistic 回归（IRLS），斜率强制非负。

**属性模式先验 π(α) 的经验贝叶斯估计**（默认开启）：标准 MMLE 不只估题目
参数，还把结构参数 π(α) 一并估出来：

    π(α) ← (1/n) Σ_i h_i(α)

固定成均匀先验会让属性掌握率被先验扭曲。开启后 AIC/BIC 会把 π(α) 的
``2^K - 1`` 个自由参数计入惩罚，保证与"固定先验"的可比性。可用
``estimate_prior=False`` 关闭，或自行传入 ``prior`` 固定。

**缺失作答按题边际化**：``np.nan`` 表示未作答，该题既不进入这一被试的
似然项，也不进入各 M 步的分子分母（MAR 假设）。把缺失当成答错会让
已掌握属性者被记一次失误：实测 20% 随机缺失时 slip 会从 0.10 抬到 0.29，
40% 时顶到上界。

所有 M 步都接受 ``clip`` 边界以保证数值稳定与 ``1-s > g`` 的单调性约束。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import ACDM, CDModel, DINA, DINO, GDINA, LLM, RRUM, create_model
from ..qmatrix import QMatrix

__all__ = ["EMCalibrator", "CalibrationResult", "calibrate"]


# ====================================================================== #
# 结果容器
# ====================================================================== #
@dataclass
class CalibrationResult:
    """标定结果。

    Attributes
    ----------
    model_name : str
        模型名。
    slip, guess : ndarray
        DINA / DINO 的题目参数（其他模型下由等价转换得到，便于横向比较）。
    beta, gamma : optional
        G-DINA 的 lambda 参数（baseline 与交互项）。
    pi_star, r_star : optional
        RRUM 的参数。
    log_likelihood_history : List[float]
        每次迭代的边际对数似然。
    n_iterations : int
    converged : bool
    log_likelihood : float
        最终边际对数似然。
    aic, bic, aicc : float
    n_params : int
        自由参数总数 = 题目参数 + 属性先验参数（若估计）。
    attr_prior : ndarray, shape (2^K,)
        属性模式先验 π(α)。经验贝叶斯估计所得，或用户指定值。
    n_missing : int
        作答矩阵中缺失单元格（``np.nan``）的个数。E 步按题边际化处理，
        不是当作答错——后者会把 slip 系统性抬高（见模块文档）。
    """

    model_name: str = "DINA"
    slip: Optional[np.ndarray] = None
    guess: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    gamma: Optional[List[np.ndarray]] = None
    pi_star: Optional[np.ndarray] = None
    r_star: Optional[List[np.ndarray]] = None
    delta0: Optional[np.ndarray] = None
    delta: Optional[List[np.ndarray]] = None
    llm_beta0: Optional[np.ndarray] = None
    llm_beta: Optional[List[np.ndarray]] = None
    log_likelihood_history: List[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    aicc: float = 0.0
    n_params: int = 0
    n_item_params: int = 0
    n_prior_params: int = 0
    attr_prior: Optional[np.ndarray] = None
    prior_estimated: bool = False
    n_missing: int = 0

    # ------------------------------------------------------------------ #
    @property
    def deviance(self) -> float:
        """偏差 ``-2·logL``。"""
        return -2.0 * self.log_likelihood

    def to_model(self, q_matrix: QMatrix) -> CDModel:
        """把标定结果还原成模型实例。"""
        if self.model_name in ("DINA", "DINO"):
            return create_model(self.model_name, q_matrix,
                                slip=self.slip, guess=self.guess)
        if self.model_name in ("G-DINA", "GDINA"):
            return GDINA(q_matrix, beta=self.beta, gamma=self.gamma)
        if self.model_name == "RRUM":
            return RRUM(q_matrix, pi_star=self.pi_star, r_star=self.r_star)
        if self.model_name == "ACDM":
            return ACDM(q_matrix, delta0=self.delta0, delta=self.delta)
        if self.model_name == "LLM":
            return LLM(q_matrix, beta0=self.llm_beta0, beta=self.llm_beta)
        raise ValueError(f"未知模型 {self.model_name}")

    def summary(self) -> str:
        last_ll = self.log_likelihood_history[-1] if self.log_likelihood_history else 0.0
        head = (
            f"标定结果 · {self.model_name}\n"
            f"  {self.n_iterations} 次迭代, "
            f"{'已收敛' if self.converged else '未收敛'}, "
            f"边际对数似然={last_ll:.4f}\n"
            f"  自由参数={self.n_params}  "
            f"AIC={self.aic:.2f}  BIC={self.bic:.2f}  AICc={self.aicc:.2f}"
        )
        if self.n_prior_params:
            head += (
                f"\n  参数构成: 题目 {self.n_item_params} + 先验 {self.n_prior_params}"
                f"（经验贝叶斯估计 π(α)）"
            )
        if self.n_missing:
            head += (
                f"\n  缺失作答: {self.n_missing} 个单元格（已按题边际化，未当作答错）"
            )
        if self.slip is not None and self.guess is not None:
            head += (
                f"\n  slip  范围 [{self.slip.min():.3f}, {self.slip.max():.3f}], "
                f"均值 {self.slip.mean():.3f}"
                f"\n  guess 范围 [{self.guess.min():.3f}, {self.guess.max():.3f}], "
                f"均值 {self.guess.mean():.3f}"
            )
        if self.attr_prior is not None:
            head += (
                f"\n  属性先验 π(α): 最大 {self.attr_prior.max():.4f}, "
                f"最小 {self.attr_prior.min():.6f}, "
                f"非零模式数 {(self.attr_prior > 1e-5).sum()}"
            )
        return head


# ====================================================================== #
# 标定器
# ====================================================================== #
class EMCalibrator:
    """DINA / DINO / G-DINA / RRUM / ACDM / LLM 的 EM 参数标定器。

    Parameters
    ----------
    q_matrix : QMatrix
    model_type : {'DINA', 'DINO', 'G-DINA', 'RRUM', 'ACDM', 'LLM'}
    max_iter : int
        最大迭代次数。
    tol : float
        对数似然增量收敛阈值。
    prior : array-like, optional
        属性模式先验（长度 2^K）。
    estimate_prior : bool, optional
        是否在 M 步用经验贝叶斯估计 π(α)（``π(α) = Σ_i h_i(α) / n``）。
        默认 ``None`` 表示"用户未显式给出 prior 时自动开启"。
        开启后题目参数估计效率更高，属性掌握率不再被均匀先验扭曲。
    prior_smoothing : float
        估计 π(α) 时加入的 Dirichlet 伪计数，防止某模式先验锁死在 0。
    bounds : tuple
        参数下界与上界，用于数值保护。

    Examples
    --------
    >>> cal = EMCalibrator(q, model_type='DINA')
    >>> result = cal.fit(responses, verbose=False)
    >>> result.slip.shape == (q.n_items,)
    True
    """

    _ALIASES = {
        "DINA": "DINA",
        "DINO": "DINO",
        "G-DINA": "G-DINA",
        "GDINA": "G-DINA",
        "RRUM": "RRUM",
        "ACDM": "ACDM",
        "ADDITIVE": "ACDM",
        "LLM": "LLM",
        "LINEAR-LOGISTIC": "LLM",
    }

    def __init__(
        self,
        q_matrix: QMatrix,
        model_type: str = "DINA",
        max_iter: int = 500,
        tol: float = 1e-6,
        prior: Optional[np.ndarray] = None,
        estimate_prior: Optional[bool] = None,
        prior_smoothing: float = 1e-6,
        lock_prior_zeros: bool = False,
        bounds: Tuple[float, float] = (0.001, 0.499),
    ) -> None:
        self.q = q_matrix if isinstance(q_matrix, QMatrix) else QMatrix(q_matrix)
        key = str(model_type).strip().upper().replace("_", "-")
        if key not in self._ALIASES:
            raise ValueError(
                f"不支持的模型类型 '{model_type}'；"
                f"仅支持 {sorted(set(self._ALIASES))}（别名不区分大小写）"
            )
        self.model_type = self._ALIASES[key]
        self.max_iter = max_iter
        self.tol = tol
        self.bounds = bounds
        self.prior_smoothing = float(prior_smoothing)

        # 未显式指定时，只在用户没有自带先验的情况下才开启经验贝叶斯
        if estimate_prior is None:
            estimate_prior = prior is None
        self.estimate_prior = bool(estimate_prior)
        self.user_prior = prior is not None

        if prior is None:
            self.prior = np.full(self.q.n_patterns, 1.0 / self.q.n_patterns)
        else:
            p = np.asarray(prior, dtype=np.float64).ravel()
            if len(p) != self.q.n_patterns:
                raise ValueError(f"先验长度应为 {self.q.n_patterns}")
            if (p < 0).any():
                raise ValueError("先验不能含负值")
            if p.sum() <= 0:
                raise ValueError("先验之和必须为正")
            self.prior = p / p.sum()

        # 先验被固定为 0 的模式（如属性层级约束下的非法模式）不可估计，
        # 不该计入自由参数。此处记录起始的非零模式数，供 n_prior_params 使用；
        # 必须在此刻取，因为 EM 中的先验平滑会把 0 抬成 1e-6。
        self._n_active_prior = int(np.count_nonzero(self.prior > 0.0))
        # 用户显式给出的 0 先验是否作为**硬约束**：为 True 时 EM 中不允许被
        # 平滑项复活（属性层级等结构性约束需要它）；默认 False，即把先验当作
        # 初值，允许数据自由更新——这也是 estimate_prior=True 的原有语义。
        self.lock_prior_zeros = bool(lock_prior_zeros)
        self._prior_mask: Optional[np.ndarray] = None
        if prior is not None and self._n_active_prior < self.q.n_patterns:
            self._prior_mask = (self.prior > 0.0).astype(np.float64)

        ideal_mode = "disjunctive" if self.model_type == "DINO" else "conjunctive"
        self.eta = self.q.ideal_response(ideal_mode).astype(np.float64)   # (S, J)
        self.patterns = self.q.attribute_patterns().astype(np.float64)    # (S, K)

    # ------------------------------------------------------------------ #
    # 概率矩阵（按模型）
    # ------------------------------------------------------------------ #
    @property
    def n_params_per_model(self) -> int:
        """题目自由参数个数（不含属性先验）。"""
        j = self.q.n_items
        if self.model_type == "G-DINA":
            return sum(1 << int(self.q.matrix[i].sum()) for i in range(j))
        if self.model_type == "RRUM":
            return sum(1 + max(int(self.q.matrix[i].sum()), 1) for i in range(j))
        if self.model_type in ("ACDM", "LLM"):
            # 截距 + 每个考查属性一个斜率
            return sum(1 + max(int(self.q.matrix[i].sum()), 1) for i in range(j))
        return 2 * j

    @property
    def n_prior_params(self) -> int:
        """估计 π(α) 引入的自由参数个数（非零先验模式数 - 1；不估计时为 0）。

        属性层级约束下非法模式的先验恒为 0，它们不参与估计，也不占自由参数，
        否则带约束的模型会被信息准则无端重罚。
        """
        if not self.estimate_prior:
            return 0
        if getattr(self, "lock_prior_zeros", False) and getattr(self, "_prior_mask", None) is not None:
            return max(int(self._prior_mask.sum()) - 1, 0)
        return max(int(self.q.n_patterns) - 1, 0)

    @property
    def n_free_params(self) -> int:
        """题目参数 + 先验参数。"""
        return int(self.n_params_per_model + self.n_prior_params)

    def _update_prior(self, resp: np.ndarray) -> None:
        """经验贝叶斯更新：``π(α) = 各模式后验责任的平均``。

        等价于对 π(α) 做极大似然更新（Bock & Aitkin, 1981 的 MMLE 结构参数）。
        加 ``prior_smoothing`` 伪计数构成 Dirichlet 正则，避免某个模式的先验
        被压到 0 之后再也无法恢复。

        若用户显式把部分模式的先验固定为 0（属性层级约束），这些模式在更新后
        仍然保持 0——它们不参与估计，是硬约束而非初值。
        """
        counts = resp.sum(axis=0) + self.prior_smoothing
        if getattr(self, "lock_prior_zeros", False) and getattr(self, "_prior_mask", None) is not None:
            counts = counts * self._prior_mask
        total = counts.sum()
        if total <= 0:
            return
        self.prior = counts / total

    def _prob_matrix(self, params: Dict[str, object]) -> np.ndarray:
        """由参数构造 P(X=1|α) 矩阵，形状 (S, J)。"""
        if self.model_type in ("DINA", "DINO"):
            slip = params["slip"]
            guess = params["guess"]
            return (1.0 - slip)[None, :] ** self.eta * guess[None, :] ** (1 - self.eta)

        if self.model_type == "G-DINA":
            beta = params["beta"]
            gamma = params["gamma"]
            logits = np.tile(beta[None, :], (self.q.n_patterns, 1))
            for i in range(self.q.n_items):
                req = np.where(self.q.matrix[i] == 1)[0]
                if len(req) == 0:
                    continue
                r = len(req)
                sub = self.patterns[:, req]
                for m in range(1, 1 << r):
                    cols = [b for b in range(r) if (m >> b) & 1]
                    indicator = sub[:, cols].astype(bool).all(axis=1).astype(np.float64)
                    logits[:, i] += gamma[i][m - 1] * indicator
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

        if self.model_type == "RRUM":
            pi_star = params["pi_star"]
            r_star = params["r_star"]
            p = np.tile(np.clip(pi_star, 1e-9, 1 - 1e-9)[None, :],
                        (self.q.n_patterns, 1))
            for i in range(self.q.n_items):
                req = np.where(self.q.matrix[i] == 1)[0]
                for pos, attr in enumerate(req):
                    missing = 1.0 - self.patterns[:, attr]
                    p[:, i] *= np.clip(r_star[i][pos], 1e-9, 1.0) ** missing
            return p

        if self.model_type == "ACDM":
            delta0 = params["delta0"]
            delta = params["delta"]
            p = np.tile(delta0[None, :], (self.q.n_patterns, 1))
            for i in range(self.q.n_items):
                req = np.where(self.q.matrix[i] == 1)[0]
                for pos, attr in enumerate(req):
                    p[:, i] += delta[i][pos] * self.patterns[:, attr]
            # 加性结构没有值域保证，这里必须裁：越界概率会让 E 步的对数爆掉
            return np.clip(p, 1e-6, 1.0 - 1e-6)

        if self.model_type == "LLM":
            beta0 = params["beta0"]
            beta = params["beta"]
            logits = np.tile(beta0[None, :], (self.q.n_patterns, 1))
            for i in range(self.q.n_items):
                req = np.where(self.q.matrix[i] == 1)[0]
                for pos, attr in enumerate(req):
                    logits[:, i] += beta[i][pos] * self.patterns[:, attr]
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

        raise ValueError(self.model_type)

    # ------------------------------------------------------------------ #
    # E 步
    # ------------------------------------------------------------------ #
    def _e_step(
        self,
        responses: np.ndarray,
        p: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """E 步：返回后验责任矩阵 (n_examinees, n_patterns)。

        ``p`` 为 P(X=1|α) 矩阵，形状 (S, J)。

        缺失作答（``observed=False``）**不参与**该被试的似然，等价于对缺失题
        做边际化：``Π_j P(x_ij|α)^{obs_ij}``。若把缺失当成答错，
        已掌握属性者会被记一次失误，slip 会被系统性高估。

        实现上按行分块做两次矩阵乘法（``X @ logP1.T + (1-X) @ logP0.T``），
        比逐人循环快一个数量级，同时用分块把中间矩阵控制在几十 MB 内。
        """
        n = responses.shape[0]
        s = self.q.n_patterns
        observed = _resolve_observed(responses, observed)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        log_p1 = np.log(p)
        log_p0 = np.log(1.0 - p)
        log_prior = np.log(self.prior + 1e-300)

        x = np.nan_to_num(responses, nan=0.0)
        w = None if observed is None else observed.astype(np.float64)

        out = np.empty((n, s), dtype=np.float64)
        step = max(1, 2_000_000 // max(s, 1))     # 每块约 16 MB
        for start in range(0, n, step):
            stop = min(start + step, n)
            xc = x[start:stop]
            if w is None:
                ll = xc @ log_p1.T + (1.0 - xc) @ log_p0.T
            else:
                wc = w[start:stop]
                # 缺失项两项都为 0，等价于从连乘中剔除
                ll = (xc * wc) @ log_p1.T + ((1.0 - xc) * wc) @ log_p0.T
            ll += log_prior
            ll -= ll.max(axis=1, keepdims=True)
            z = np.exp(ll)
            out[start:stop] = z / z.sum(axis=1, keepdims=True)
        return out

    # ------------------------------------------------------------------ #
    # M 步
    # ------------------------------------------------------------------ #
    def _m_step(
        self,
        responses: np.ndarray,
        resp: np.ndarray,
        params: Dict[str, object],
        observed: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        observed = _resolve_observed(responses, observed)
        if self.model_type in ("DINA", "DINO"):
            return self._m_step_dina(responses, resp, observed)
        if self.model_type == "G-DINA":
            return self._m_step_gdina(responses, resp, observed)
        if self.model_type == "RRUM":
            return self._m_step_rrum(responses, resp, observed)
        if self.model_type == "ACDM":
            return self._m_step_acdm(responses, resp, observed)
        if self.model_type == "LLM":
            return self._m_step_llm(responses, resp, observed)
        raise ValueError(self.model_type)

    def _m_step_dina(
        self,
        responses: np.ndarray,
        resp: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """DINA / DINO 的闭式 M 步。

        分母只累计**观测到**该题的被试责任：缺失者既不算对也不算错，
        否则 slip 会被分母里那些"被当成 0 分"的缺失抬高。
        """
        lo, hi = self.bounds
        j = self.q.n_items
        slip = np.zeros(j)
        guess = np.zeros(j)
        for item in range(j):
            eta_j = self.eta[:, item]
            x_j = responses[:, item]
            if observed is None:
                w = resp
            else:
                obs_j = observed[:, item]
                x_j = np.where(obs_j, x_j, 0.0)
                w = resp * obs_j[:, None]
            w1 = w * eta_j[None, :]
            w0 = w * (1.0 - eta_j)[None, :]
            n1 = w1.sum()
            n0 = w0.sum()
            slip[item] = float((w1 * (1.0 - x_j)[:, None]).sum() / n1) if n1 > 1e-10 else lo
            guess[item] = float((w0 * x_j[:, None]).sum() / n0) if n0 > 1e-10 else lo
        slip = np.clip(slip, lo, hi)
        guess = np.clip(guess, lo, hi)
        # 保证 1-s > g（区分度为正）
        bad = (1.0 - slip) <= guess
        if bad.any():
            mid = (1.0 - slip + guess) / 2.0
            slip[bad] = np.clip(1.0 - mid[bad] - 1e-4, lo, hi)
            guess[bad] = np.clip(mid[bad] - 1e-4, lo, hi)
        return {"slip": slip, "guess": guess}

    def _m_step_gdina(
        self,
        responses: np.ndarray,
        resp: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """G-DINA 的 M 步：各潜在组概率 = 组内期望答对率，再取 logit。

        对第 j 题的某个潜在组 g（由"哪些属性被掌握"的模式表示），
        该组的期望人数为 ``N_g = Σ_i Σ_{α∈g} h_i(α)``，
        其中答对人数为 ``C_g = Σ_i Σ_{α∈g} h_i(α)·X_ij``。
        组概率 ``P_g = C_g / N_g``，在指示函数基下与 λ 参数一一对应：
        先算组 logit，再用设计矩阵解出 λ。

        缺失该题的被试同时从 ``N_g`` 与 ``C_g`` 中剔除。
        """
        beta = np.zeros(self.q.n_items)
        gamma: List[np.ndarray] = []

        for i in range(self.q.n_items):
            req = np.where(self.q.matrix[i] == 1)[0]
            r = len(req)
            obs_i = None if observed is None else observed[:, i]
            x_i = responses[:, i]
            if obs_i is not None:
                x_i = np.where(obs_i, x_i, 0.0)
            if r == 0:
                # 不考查任何属性：单一组，直接取样本答对率
                denom = resp.sum() if obs_i is None else resp.sum(axis=1) @ obs_i
                p1 = (
                    float((resp.sum(axis=1) * x_i).sum() / denom)
                    if denom > 1e-10
                    else 0.5
                )
                beta[i] = _logit_scalar(p1)
                gamma.append(np.zeros(0))
                continue

            sub = self.patterns[:, req]                 # (S, r)
            # 每个潜在组 g（0..2^r-1，按 mask 顺序）统计
            n_groups = 1 << r
            group_denom = np.zeros(n_groups)
            group_correct = np.zeros(n_groups)
            for g in range(n_groups):
                bits = np.array([(g >> (r - 1 - pos)) & 1 for pos in range(r)])
                in_group = (sub == bits[None, :]).all(axis=1)      # (S,)
                w = resp[:, in_group].sum(axis=1)                  # (n,)
                if obs_i is not None:
                    w = w * obs_i
                group_denom[g] = w.sum()
                group_correct[g] = (w * x_i).sum()

            # 组概率
            p_group = np.where(
                group_denom > 1e-10,
                group_correct / np.maximum(group_denom, 1e-10),
                0.5,
            )
            p_group = np.clip(p_group, 0.005, 0.995)
            target = _logit_scalar_vec(p_group)
            design = self._group_design(i, r)
            try:
                theta = np.linalg.solve(design, target)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(design, target, rcond=None)[0]
            beta[i] = theta[0]
            gamma.append(theta[1:])

        return {"beta": beta, "gamma": gamma}

    def _group_design(self, item_idx: int, r: int) -> np.ndarray:
        """潜在组设计矩阵 X，形状 (2^r, 2^r)。

        首列为 1（baseline），第 m 列为"该组是否掌握第 m 个子集全部属性"。
        """
        if r == 0:
            return np.ones((1, 1))
        groups = ((np.arange(1 << r)[:, None] >> np.arange(r - 1, -1, -1)) & 1).astype(np.float64)
        x = np.ones((1 << r, 1 << r))
        for m in range(1, 1 << r):
            cols = [b for b in range(r) if (m >> b) & 1]
            x[:, m] = groups[:, cols].astype(bool).all(axis=1).astype(np.float64)
        return x

    def _m_step_rrum(
        self,
        responses: np.ndarray,
        resp: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """RRUM 的 M 步。

        ``π*_j`` 为掌握**全部**所需属性者的答对率；
        其余模式按缺少的属性数应用惩罚 ``r*_{jk}``。
        这里用组期望答对率与惩罚累乘的解析关系更新：
        对每组求观测答对率，再逐属性拟合惩罚因子。

        缺失该题的被试同时从各组分子与分母中剔除。
        """
        lo, hi = self.bounds
        pi_star = np.zeros(self.q.n_items)
        r_star: List[np.ndarray] = []

        for i in range(self.q.n_items):
            req = np.where(self.q.matrix[i] == 1)[0]
            r = len(req)
            obs_i = None if observed is None else observed[:, i]
            x_i = responses[:, i]
            if obs_i is not None:
                x_i = np.where(obs_i, x_i, 0.0)
            if r == 0:
                denom = resp.sum() if obs_i is None else resp.sum(axis=1) @ obs_i
                pi_star[i] = (
                    float((resp.sum(axis=1) * x_i).sum() / denom)
                    if denom > 1e-10
                    else 0.5
                )
                r_star.append(np.zeros(1))
                continue

            sub = self.patterns[:, req]
            n_groups = 1 << r
            denom = np.zeros(n_groups)
            correct = np.zeros(n_groups)
            for g in range(n_groups):
                bits = np.array([(g >> (r - 1 - pos)) & 1 for pos in range(r)])
                in_group = (sub == bits[None, :]).all(axis=1)
                w = resp[:, in_group].sum(axis=1)
                if obs_i is not None:
                    w = w * obs_i
                denom[g] = w.sum()
                correct[g] = (w * x_i).sum()

            p_group = np.where(denom > 1e-10, correct / np.maximum(denom, 1e-10), np.nan)
            # π* 为全掌握组（mask = 2^r - 1，即 bits 全 1）
            full_idx = n_groups - 1
            pi = p_group[full_idx] if np.isfinite(p_group[full_idx]) else 0.9
            pi_star[i] = float(np.clip(pi, 0.30, 0.995))

            # 惩罚 r_{jk}: 对只缺属性 k 的组，r = p_group / pi_star
            penalties = np.ones(r)
            for pos in range(r):
                bits = np.ones(r, dtype=int)
                bits[pos] = 0
                idx = int(sum(int(bits[p]) << (r - 1 - p) for p in range(r)))
                if np.isfinite(p_group[idx]) and pi_star[i] > 1e-6:
                    penalties[pos] = float(np.clip(p_group[idx] / pi_star[i], 0.05, 0.99))
            r_star.append(penalties)

        return {"pi_star": pi_star, "r_star": r_star}

    # ------------------------------------------------------------------ #
    # ACDM / LLM 的 M 步
    # ------------------------------------------------------------------ #
    def _group_stats(
        self,
        item_idx: int,
        responses: np.ndarray,
        resp: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """第 i 题各潜在组的（属性组合, 后验权重, 加权答对数）。

        三者的行数都是 ``2^{r_j}``，第 g 行对应 r_j 位二进制 g；
        ``权重 = Σ_人 该组责任``、``答对 = Σ_人 责任 · x_ij``，
        两者相除就是该组的加权答对率——这是 ACDM / LLM / RRUM 三个
        M 步共同的输入，抽出来避免写三遍。

        缺失该题的被试权重记 0，即从分子与分母里同时消失。
        """
        req = np.where(self.q.matrix[item_idx] == 1)[0]
        r = len(req)
        obs_i = None if observed is None else observed[:, item_idx]
        x_i = responses[:, item_idx]
        if obs_i is not None:
            x_i = np.where(obs_i, x_i, 0.0)
        if r == 0:
            weight0 = resp.sum() if obs_i is None else float(resp.sum(axis=1) @ obs_i)
            return (
                np.zeros((1, 0)),
                np.array([weight0]),
                np.array([float((resp.sum(axis=1) * x_i).sum())]),
            )

        sub = self.patterns[:, req]
        n_groups = 1 << r
        bits = ((np.arange(n_groups)[:, None] >> np.arange(r - 1, -1, -1)) & 1).astype(np.float64)
        weight = np.zeros(n_groups)
        correct = np.zeros(n_groups)
        for g in range(n_groups):
            in_group = (sub == bits[g][None, :]).all(axis=1)
            w_g = resp[:, in_group].sum(axis=1)
            if obs_i is not None:
                w_g = w_g * obs_i
            weight[g] = w_g.sum()
            correct[g] = (w_g * x_i).sum()
        return bits, weight, correct

    def _m_step_acdm(
        self,
        responses: np.ndarray,
        resp: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """ACDM 的 M 步：对分组答对率做加权最小二乘。

        ACDM 在**概率尺度**上是线性的（``P = Xδ``），所以给定后验责任后
        M 步有闭式解，不需要迭代。权重用组责任，等价于按信息量加权。
        """
        delta0 = np.zeros(self.q.n_items)
        delta: List[np.ndarray] = []

        for i in range(self.q.n_items):
            bits, weight, correct = self._group_stats(i, responses, resp, observed)
            r = bits.shape[1]
            if r == 0:
                p = correct[0] / max(weight[0], 1e-10)
                delta0[i] = float(np.clip(p, 1e-3, 1.0 - 1e-3))
                delta.append(np.zeros(1))
                continue

            design = np.column_stack([np.ones(len(weight)), bits])
            rate = correct / np.maximum(weight, 1e-10)
            sw = np.sqrt(np.maximum(weight, 0.0))          # 空组权重为 0，自然被剔除
            coef, *_ = np.linalg.lstsq(design * sw[:, None], rate * sw, rcond=None)

            d0 = float(np.clip(coef[0], 1e-3, 1.0 - 1e-3))
            d = np.clip(coef[1:], 0.0, None)
            # 上界约束 δ0 + Σδ ≤ 1：超出就把斜率按比例压回去，保持形状
            if d0 + d.sum() > 0.999 and d.sum() > 1e-12:
                d = d * max(0.999 - d0, 1e-6) / d.sum()
            delta0[i] = d0
            delta.append(d)

        return {"delta0": delta0, "delta": delta}

    def _m_step_llm(
        self,
        responses: np.ndarray,
        resp: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """LLM 的 M 步：对分组答对率做二项 logistic 回归（IRLS）。

        每组只有"权重 / 加权答对数"，所以是 ``2^{r_j}`` 行的加权回归，
        而不是逐被试回归——这是它能放进 EM 循环里的关键。
        IRLS 一般 5~8 步收敛；斜率强制非负，保证掌握属性不会降低答对概率。
        """
        beta0 = np.zeros(self.q.n_items)
        beta: List[np.ndarray] = []

        for i in range(self.q.n_items):
            bits, weight, correct = self._group_stats(i, responses, resp, observed)
            r = bits.shape[1]
            if r == 0:
                p = np.clip(correct[0] / max(weight[0], 1e-10), 1e-3, 1.0 - 1e-3)
                beta0[i] = float(np.log(p / (1.0 - p)))
                beta.append(np.zeros(1))
                continue

            design = np.column_stack([np.ones(len(weight)), bits])
            keep = weight > 1e-8
            if not keep.any():
                beta0[i] = -1.386
                beta.append(np.zeros(r))
                continue

            x = design[keep]
            w = weight[keep]
            rate = np.clip(correct[keep] / w, 1e-6, 1.0 - 1e-6)

            coef = np.zeros(r + 1)
            m0 = float(rate.mean())
            coef[0] = float(np.log(m0 / (1.0 - m0)))
            for _ in range(30):
                eta = np.clip(x @ coef, -30.0, 30.0)
                mu = 1.0 / (1.0 + np.exp(-eta))
                v = np.maximum(mu * (1.0 - mu), 1e-8)
                z = eta + (rate - mu) / v
                xtw = x.T * (w * v)
                # 加一点岭惩罚：某组全对/全错时 Hessian 会接近奇异
                step = np.linalg.solve(xtw @ x + 1e-6 * np.eye(r + 1), xtw @ z)
                step = np.clip(step, -30.0, 30.0)
                delta_step = float(np.max(np.abs(step - coef)))
                coef = step
                if delta_step < 1e-9:
                    break

            coef = np.clip(coef, -30.0, 30.0)
            beta0[i] = float(coef[0])
            beta.append(np.clip(coef[1:], 0.0, None))

        return {"beta0": beta0, "beta": beta}

    # ------------------------------------------------------------------ #
    # 似然
    # ------------------------------------------------------------------ #
    def _log_likelihood(
        self,
        responses: np.ndarray,
        p: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> float:
        """边际对数似然。缺失作答按题边际化（不进入该被试的似然项）。"""
        observed = _resolve_observed(responses, observed)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        log_p1 = np.log(p)
        log_p0 = np.log(1.0 - p)
        log_prior = np.log(self.prior + 1e-300)

        n, s = responses.shape[0], self.q.n_patterns
        x = np.nan_to_num(responses, nan=0.0)
        w = None if observed is None else observed.astype(np.float64)

        total = 0.0
        step = max(1, 2_000_000 // max(s, 1))
        for start in range(0, n, step):
            stop = min(start + step, n)
            xc = x[start:stop]
            if w is None:
                ll = xc @ log_p1.T + (1.0 - xc) @ log_p0.T
            else:
                wc = w[start:stop]
                ll = (xc * wc) @ log_p1.T + ((1.0 - xc) * wc) @ log_p0.T
            ll += log_prior
            m = ll.max(axis=1, keepdims=True)
            total += float((m[:, 0] + np.log(np.exp(ll - m).sum(axis=1))).sum())
        return total

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def _initial_params(self, slip_init, guess_init) -> Dict[str, object]:
        j = self.q.n_items
        if self.model_type in ("DINA", "DINO"):
            slip = np.full(j, 0.1) if slip_init is None else np.asarray(slip_init, float).ravel()
            guess = np.full(j, 0.1) if guess_init is None else np.asarray(guess_init, float).ravel()
            return {"slip": slip.copy(), "guess": guess.copy()}

        if self.model_type == "G-DINA":
            # 用 DINA 参数经 from_dina 转换得到等价初值（比全零更接近真解）
            slip = np.full(j, 0.1) if slip_init is None else np.asarray(slip_init, float).ravel()
            guess = np.full(j, 0.1) if guess_init is None else np.asarray(guess_init, float).ravel()
            m = GDINA.from_dina(self.q, slip, guess)
            return {"beta": m.beta.copy(), "gamma": [g.copy() for g in m.gamma]}

        if self.model_type == "RRUM":
            pi = np.full(j, 0.9) if slip_init is None else (1.0 - np.asarray(slip_init, float).ravel())
            rstar = []
            for i in range(j):
                r = max(int(self.q.matrix[i].sum()), 1)
                base = 0.8 if guess_init is None else float(np.mean(guess_init))
                rstar.append(np.full(r, base))
            return {"pi_star": pi.copy(), "r_star": rstar}

        if self.model_type in ("ACDM", "LLM"):
            # 用 DINA 参数投影到对应结构（LS），比全零初值收敛快得多
            slip = np.full(j, 0.1) if slip_init is None else np.asarray(slip_init, float).ravel()
            guess = np.full(j, 0.1) if guess_init is None else np.asarray(guess_init, float).ravel()
            if self.model_type == "ACDM":
                m = ACDM.from_dina(self.q, slip, guess)
                return {"delta0": m.delta0.copy(), "delta": [d.copy() for d in m.delta]}
            m_llm = LLM.from_dina(self.q, slip, guess)
            return {"beta0": m_llm.beta0.copy(), "beta": [b.copy() for b in m_llm.beta]}

        raise ValueError(self.model_type)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def fit(
        self,
        responses: np.ndarray,
        slip_init=None,
        guess_init=None,
        verbose: bool = False,
        epsilon: float = 1e-10,
    ) -> CalibrationResult:
        """执行 EM 标定。

        Parameters
        ----------
        responses : ndarray, shape (n_examinees, n_items)
            0/1 作答矩阵。缺失值用 ``np.nan`` 标记：E 步与各 M 步按题**边际化**
            （该题不进入这一被试的似然，也不进参数的分母），
            等价于 MAR 假设下的处理。若整列缺失，该题参数不可识别，会保持初值。
        slip_init, guess_init : array-like, optional
            初始值；DINA/DINO 默认 0.1，也可用于其他模型的初值来源。
        verbose : bool
        epsilon : float
            似然增量收敛阈值（覆盖实例的 tol 时使用更小值）。

        Returns
        -------
        CalibrationResult
        """
        responses = np.asarray(responses, dtype=np.float64)
        if responses.ndim != 2:
            raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
        if responses.shape[1] != self.q.n_items:
            raise ValueError(
                f"作答矩阵列数 {responses.shape[1]} 与题目数 {self.q.n_items} 不符"
            )

        observed: Optional[np.ndarray] = None
        n_missing = 0
        if np.isnan(responses).any():
            observed = ~np.isnan(responses)
            n_missing = int((~observed).sum())
            empty = np.where(~observed.any(axis=0))[0]
            if empty.size:
                warnings.warn(
                    f"第 {empty.tolist()} 题全部作答缺失，其题目参数不可识别，"
                    "将保持初值（DINA 类落到下界）。",
                    RuntimeWarning,
                    stacklevel=2,
                )

        params = self._initial_params(slip_init, guess_init)
        ll_hist: List[float] = []
        prev_ll = -np.inf
        converged = False
        it = 0

        for it in range(1, self.max_iter + 1):
            p = self._prob_matrix(params)
            resp = self._e_step(responses, p, observed)
            params = self._m_step(responses, resp, params, observed)
            # M 步之二：经验贝叶斯更新属性模式先验 π(α)
            if self.estimate_prior:
                self._update_prior(resp)

            ll = self._log_likelihood(responses, self._prob_matrix(params), observed)
            ll_hist.append(ll)
            if verbose and (it % 5 == 0 or it == 1):
                print(f"  iter {it:4d}  loglik={ll:.4f}  Δ={ll - prev_ll:.6f}")
            if it > 1 and abs(ll - prev_ll) < max(self.tol, epsilon):
                converged = True
                break
            prev_ll = ll

        n_item_params = self.n_params_per_model
        n_prior_params = self.n_prior_params
        n_params = self.n_free_params
        n = responses.shape[0]
        ll_final = ll_hist[-1] if ll_hist else 0.0
        deviance = -2.0 * ll_final
        aic = deviance + 2 * n_params
        bic = deviance + n_params * np.log(max(n, 2))
        denom = max(n - n_params - 1, 1)
        aicc = aic + (2 * n_params * (n_params + 1)) / denom

        result = CalibrationResult(
            model_name=self.model_type,
            log_likelihood_history=ll_hist,
            n_iterations=it,
            converged=converged,
            log_likelihood=float(ll_final),
            aic=float(aic),
            bic=float(bic),
            aicc=float(aicc),
            n_params=int(n_params),
            n_item_params=int(n_item_params),
            n_prior_params=int(n_prior_params),
            attr_prior=self.prior.copy(),
            prior_estimated=bool(self.estimate_prior),
            n_missing=n_missing,
        )
        # 填充各模型参数
        if self.model_type in ("DINA", "DINO"):
            result.slip = params["slip"].copy()
            result.guess = params["guess"].copy()
        elif self.model_type == "G-DINA":
            result.beta = params["beta"].copy()
            result.gamma = [g.copy() for g in params["gamma"]]
            # 由 G-DINA 反推等价 DINA 参数用于横向比较
            slip, guess = self._gdina_to_dina(params)
            result.slip, result.guess = slip, guess
        elif self.model_type == "RRUM":
            result.pi_star = params["pi_star"].copy()
            result.r_star = [r.copy() for r in params["r_star"]]
            slip, guess = self._prob_to_dina(params)
            result.slip, result.guess = slip, guess
        elif self.model_type == "ACDM":
            result.delta0 = params["delta0"].copy()
            result.delta = [d.copy() for d in params["delta"]]
            slip, guess = self._prob_to_dina(params)
            result.slip, result.guess = slip, guess
        elif self.model_type == "LLM":
            result.llm_beta0 = params["beta0"].copy()
            result.llm_beta = [b.copy() for b in params["beta"]]
            slip, guess = self._prob_to_dina(params)
            result.slip, result.guess = slip, guess
        return result

    # ------------------------------------------------------------------ #
    # 模型间参数转换（便于横向比较与初始化）
    # ------------------------------------------------------------------ #
    def _prob_to_dina(self, params: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
        """由任意模型的概率矩阵提取"全掌握 / 非全掌握"两组的等价 DINA 参数。

        为什么统一到 DINA 口径：slip/guess 是唯一跨模型可比的题目参数
        （2 个标量），所以模型比较表、题目区分度、仿真器都读这一对；
        各模型的原始参数（λ、π*、δ、β）留在 ``CalibrationResult`` 里供
        需要精确复现时使用。取组均值而不是取某个单点，是为了让"等价"
        在最小二乘意义上成立。
        """
        p = self._prob_matrix(params)
        j = self.q.n_items
        slip = np.full(j, 0.1)
        guess = np.full(j, 0.1)
        for i in range(j):
            eta_i = self.eta[:, i] > 0.5          # 掌握全部所需属性
            if eta_i.sum() > 0:
                slip[i] = float(np.clip(1.0 - p[eta_i, i].mean(), 0.001, 0.499))
            if (~eta_i).sum() > 0:
                guess[i] = float(np.clip(p[~eta_i, i].mean(), 0.001, 0.499))
        return slip, guess

    def _gdina_to_dina(self, params: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
        """兼容旧接口，等价于 :meth:`_prob_to_dina`。"""
        return self._prob_to_dina(params)

    def _rrum_to_dina(self, params: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
        """兼容旧接口，等价于 :meth:`_prob_to_dina`。"""
        return self._prob_to_dina(params)

    # ------------------------------------------------------------------ #
    # 便捷方法
    # ------------------------------------------------------------------ #
    def calibrate_to_model(self, responses: np.ndarray, **kwargs) -> Tuple[CDModel, CalibrationResult]:
        """标定并直接返回构建好的模型实例与结果。"""
        result = self.fit(responses, **kwargs)
        return result.to_model(self.q), result

    def examinee_patterns(self, responses: np.ndarray, slip=None, guess=None,
                          model: Optional[CDModel] = None) -> np.ndarray:
        """给定参数，用 E 步的**众数 (MAP)** 估计各被试属性模式。

        沿用标定时得到的属性先验 π(α)（若有），保证与标定口径一致。
        为向后兼容保留；建议改用 :func:`cdcat.diagnostics.diagnose_patterns`
        以获得更完整的信息（EAP、后验、信度）。
        """
        if model is None:
            if slip is None or guess is None:
                raise ValueError("需提供 slip/guess 或 model")
            model = self._build_model(slip, guess)
        diag = diagnose_patterns(model, responses, prior=self.prior)
        return diag.map_patterns

    def _build_model(self, slip, guess) -> CDModel:
        """由等价 DINA 参数构造当前类型的模型实例。"""
        if self.model_type in ("DINA", "DINO"):
            return create_model(self.model_type, self.q, slip=slip, guess=guess)
        if self.model_type == "G-DINA":
            return GDINA.from_dina(self.q, slip, guess)
        if self.model_type == "ACDM":
            return ACDM.from_dina(self.q, slip, guess)
        if self.model_type == "LLM":
            return LLM.from_dina(self.q, slip, guess)
        return create_model("DINA", self.q, slip=slip, guess=guess)

    def item_discrimination(self, slip, guess) -> np.ndarray:
        """题目区分度 ``1 - slip - guess``。"""
        return 1.0 - np.asarray(slip) - np.asarray(guess)


# ====================================================================== #
# 便捷函数
# ====================================================================== #
def calibrate(
    q_matrix: QMatrix,
    responses: np.ndarray,
    model_type: str = "DINA",
    verbose: bool = False,
    **kwargs,
) -> Tuple[CDModel, CalibrationResult]:
    """一行完成「标定 + 建模」。

    默认启用经验贝叶斯属性先验估计；传 ``estimate_prior=False`` 可关闭。

    Examples
    --------
    >>> model, res = calibrate(q, responses, model_type="G-DINA")
    >>> res.aic < 0  # 数值取决于数据  # doctest: +SKIP
    """
    allowed = ("max_iter", "tol", "prior", "bounds",
               "estimate_prior", "prior_smoothing", "lock_prior_zeros")
    cal = EMCalibrator(q_matrix, model_type=model_type, **{
        k: v for k, v in kwargs.items() if k in allowed
    })
    return cal.calibrate_to_model(responses, verbose=verbose)


# 延迟导入以避免循环依赖
from ..diagnostics import diagnose_patterns  # noqa: E402


def _logit_scalar(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _resolve_observed(
    responses: np.ndarray, observed: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """返回观测掩码；未显式传入时按 ``np.nan`` 自动推断。

    显式传参是为了让 ``fit`` 只扫一遍 NaN；自动推断是防呆——若调用方忘了传，
    缺失值会退化成"答错"，正是本轮修复要消除的偏差来源。
    """
    if observed is not None:
        return observed
    if np.isnan(responses).any():
        return ~np.isnan(responses)
    return None


def _logit_scalar_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


# 在线标定（新题参数）
from .online import NewItemResult, OnlineCalibrationReport, calibrate_new_items  # noqa: E402,F401
