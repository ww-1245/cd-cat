"""后验预测模型检验（PPMC, Posterior Predictive Model Checking）。

为什么需要它
------------
现有的模型比较（AIC/BIC）与似然比检验都建立在**渐近**理论上：
样本小时 χ² 近似不准，且 LRT 只能比较**嵌套**模型。

PPMC 绕开这两点：它不问"哪个模型更好"，而问

    如果这个世界真的是这个模型，我有多大可能看到**比现在更极端**的数据？

做法（Rubin 1984；Levy & Mislevy 2016 的 CDM 版本）：

1. 用观测数据估计模型，得到每名考生的属性模式后验 :math:`p(\\alpha \\mid X)`；
2. 从中抽 :math:`\\alpha_i^{*}`，再按 :math:`P(X_{ij}=1\\mid\\alpha_i^{*})` 生成**复制数据集** :math:`X^{*}`；
3. 选一个差异统计量 :math:`T`，比较 :math:`T(X)` 与 :math:`T(X^{*})` 的分布。

后验预测 p 值（PPP）::

    PPP = P(T(X*) ≥ T(X) | X)

PPP 接近 0 表示"观测数据比模型预期的还极端"，接近 1 表示"模型预期过度分散"，
**两端都是失配信号**——这与传统 p 值只看一端的直觉不同，需特别注意。

关于"双重使用数据"
------------------
差异统计量 :math:`T(X, P)` 里要放一份"模型认为的答对概率" :math:`P`。
如果观测侧用 MAP 硬模式的条件概率，而复制侧用抽样得到的概率，两侧口径就不对称：
MAP 模式本身是**挑出来最贴合该考生作答**的，天然过拟合，会让 :math:`T(X)`
被压小，PPP 系统性偏向 1，于是再正确的模型也会被判失配（假阳性）。

本实现的处理：

- 观测侧与复制侧都用**后验期望答对概率** :math:`\\bar p_{ij}=\\sum_s p(\\alpha_s\\mid X_i)P(X_{ij}=1\\mid\\alpha_s)`；
- 当属性模式由数据估计时，每份复制数据都**重新走一遍后验估计**再算统计量，
  使"估计带来的收缩"在两侧完全对等；
- 当调用方通过 ``alpha_hat`` 直接把属性模式当作**已知**给出时，不做重估
  （已知量不存在选择性偏差），复制数据直接按该模式的条件概率生成。

这两种口径分别对应"属性未知"与"属性已知"两类问题，结果中由
``PPMCResult.alpha_source`` 标明，避免误读。

可用的差异统计量
----------------
====================  =========================================================
``pair_srmsr``        题目对标准化残差均方根（局部依赖的总体指标，越小越好）
``q3_max``            最大 Yen Q3（局部依赖的极端情形）
``joint_max``         题目对联合答对率与模型隐含值最大偏差
``score_sd``          总分标准差（分布的离散程度是否被重现）
====================  =========================================================

为什么没有"题目边际答对率"
--------------------------
DINA/DINO 每题 2 个自由参数，EM 收敛后似然方程保证 :math:`\\sum_i \\bar p_{ij}`
恒等于 :math:`\\bar x_j`——边际答对率**被精确匹配**，观测侧与复制侧都恒为 0，
PPP 只剩数值噪声。这不是本实现的取舍，而是模型本身的饱和性质；认知诊断的拟合
评估因此要看二阶（题目对残差相关）结构，上面四个统计量都在这一层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..models import CDModel

__all__ = ["PPMCResult", "posterior_predictive_check", "PPMC_STATISTICS"]

PPMC_STATISTICS = ("pair_srmsr", "q3_max", "joint_max", "score_sd")

#: 基于题目对的统计量（计算与内存均为 O(J²)，题目过多时跳过）
_PAIR_STATS = ("pair_srmsr", "q3_max", "joint_max")


# ---------------------------------------------------------------------- #
# 结果
# ---------------------------------------------------------------------- #
@dataclass
class PPMCResult:
    """PPMC 检验结果。"""

    ppp: Dict[str, float]                  # 各统计量的后验预测 p 值
    observed: Dict[str, float]
    replicated: np.ndarray                 # (n_replicates, n_stats)
    stat_names: List[str]
    n_replicates: int
    n_examinees: int
    n_items: int
    seed: int
    alpha: float = 0.05
    alpha_source: str = "estimated"        # "estimated"（由数据估）或 "given"（外部给定）
    used_refit: bool = False               # 复制侧是否重估了题目参数

    # ---- 判读 ----
    @property
    def misfit_stats(self) -> List[str]:
        """PPP 落在两端尾部（默认 5%/95%）的统计量。"""
        out = []
        for name, p in self.ppp.items():
            if not np.isfinite(p):
                continue
            if p < self.alpha or p > 1.0 - self.alpha:
                out.append(name)
        return out

    @property
    def rmsea_ppp(self) -> float:
        """SRMSR 的 PPP（近似 RMSEA 的失配判读）。"""
        return float(self.ppp.get("pair_srmsr", float("nan")))

    def verdict(self) -> str:
        """一句话结论。"""
        bad = self.misfit_stats
        if not bad:
            return "模型与数据的后验预测一致，未检出系统性失配。"
        names = {
            "pair_srmsr": "局部依赖（SRMSR）",
            "q3_max": "局部依赖（Q3）",
            "score_sd": "总分离散程度",
            "joint_max": "题目对联合答对率",
        }
        return "检测到失配信号：" + "、".join(names.get(b, b) for b in bad) + "。"

    # ---- 输出 ----
    def to_dataframe(self):
        import pandas as pd

        rows = []
        for i, name in enumerate(self.stat_names):
            rep = self.replicated[:, i]
            rows.append({
                "统计量": name,
                "观测值": round(float(self.observed[name]), 4),
                "复制均值": round(float(rep.mean()), 4) if rep.size else float("nan"),
                "复制 SD": round(float(rep.std(ddof=1)), 4) if rep.size > 1 else float("nan"),
                "PPP": round(float(self.ppp[name]), 4),
                "判读": ("失配" if name in self.misfit_stats else "可接受"),
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {
            "n_replicates": self.n_replicates,
            "n_examinees": self.n_examinees,
            "n_items": self.n_items,
            "seed": self.seed,
            "alpha": self.alpha,
            "alpha_source": self.alpha_source,
            "ppp": {k: round(float(v), 6) for k, v in self.ppp.items()},
            "observed": {k: round(float(v), 6) for k, v in self.observed.items()},
            "misfit_stats": self.misfit_stats,
            "verdict": self.verdict(),
        }

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "后验预测模型检验（PPMC）",
            "=" * 68,
            f"  被试数 / 题目数   : {self.n_examinees} / {self.n_items}",
            f"  复制数据集        : {self.n_replicates} 份（seed={self.seed}）",
            f"  属性模式来源      : "
            f"{'外部给定（视为已知）' if self.alpha_source == 'given' else '由数据估计'}",
            f"  复制侧重估参数    : {'是' if self.used_refit else '否（近似）'}",
            "",
            f"  {'统计量':<16}{'观测':>10}{'复制均值':>12}{'PPP':>9}   判读",
            "  " + "-" * 62,
        ]
        for i, name in enumerate(self.stat_names):
            rep = self.replicated[:, i]
            m = float(rep.mean()) if rep.size else float("nan")
            flag = "失配" if name in self.misfit_stats else "可接受"
            lines.append(
                f"  {name:<16}{self.observed[name]:>10.4f}{m:>12.4f}"
                f"{self.ppp[name]:>9.4f}   {flag}"
            )
        lines += ["", "  " + self.verdict()]
        lines.append(
            "  提示：PPP 是**双侧**判读——接近 1 同样说明模型未能重现数据的离散程度。"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 差异统计量
#
# 注意：这里**没有**"题目边际答对率"类的统计量。DINA/DINO 每题 2 个自由参数，
# EM 收敛后似然方程保证 Σᵢ p̄ᵢⱼ 恒等于 x̄ⱼ，边际答对率被精确匹配；把它当差异
# 统计量，观测侧恒为 0、复制侧也恒为 0，PPP 只剩数值噪声。所以认知诊断的拟合
# 评估要看**二阶**结构（题目对的残差相关），下面的统计量都基于这一层。
# ---------------------------------------------------------------------- #
@dataclass
class _Moments:
    """一侧数据的模型隐含矩（用于计算差异统计量）。"""

    p_bar: np.ndarray      # (n, J) 每名被试的后验期望答对概率
    mean: np.ndarray       # (J,)  模型隐含的题目边际
    joint: np.ndarray      # (J, J) 模型隐含的两两联合答对率


def _moments(post: np.ndarray, prob: np.ndarray) -> _Moments:
    """由属性模式后验与概率矩阵求模型隐含矩。

    两两联合率只需**平均后验** ``w̄``：``Σ_s w̄_s P_sa P_sb``，
    因为对 i 的求和与对 s 的求和可以交换。
    """
    w_bar = post.mean(axis=0)                       # (S,)
    p_bar = post @ prob                            # (n, J)
    mean = w_bar @ prob                            # (J,)
    joint = (w_bar[:, None] * prob).T @ prob       # (J, J)
    return _Moments(p_bar=p_bar, mean=mean, joint=joint)


def _pair_index(j: int):
    return np.triu_indices(j, k=1)


def _observed_joint(x: np.ndarray, mask: np.ndarray) -> tuple:
    """成对完整信息（pairwise-complete）的联合答对率。

    只在该题目对**两题都被作答**的被试上统计，缺失位（填补值）不进入分子分母。
    直接用填补值相乘会让缺失比例转化为虚假的题目对关联，把 SRMSR 抬高到误判水平。
    """
    m = mask.astype(np.float64)
    xm = np.where(mask, x, 0.0)
    n_ab = m.T @ m                            # 每对可用人数
    s_ab = xm.T @ xm                          # 每对共同答对人数
    with np.errstate(invalid="ignore", divide="ignore"):
        joint = s_ab / np.maximum(n_ab, 1.0)
    return joint, n_ab


def _stat_pair_srmsr(x: np.ndarray, mom: _Moments, mask: np.ndarray) -> float:
    """题目对标准化残差均方根（SRMSR，局部依赖的总体指标）。"""
    j = x.shape[1]
    if j < 2:
        return 0.0
    obs, n_ab = _observed_joint(x, mask)
    sd = np.sqrt(np.clip(mom.mean * (1.0 - mom.mean), 1e-12, None))
    iu = _pair_index(j)
    ok = n_ab[iu] >= 2
    if not ok.any():
        return 0.0
    num = obs[iu] - mom.joint[iu]
    den = sd[iu[0]] * sd[iu[1]]
    return float(np.sqrt(np.mean((num[ok] / den[ok]) ** 2)))


def _stat_q3_max(x: np.ndarray, mom: _Moments, mask: np.ndarray) -> float:
    """最大 |Yen Q3|：条件于属性模式后的题间残差相关。"""
    j = x.shape[1]
    if j < 2:
        return 0.0
    r = np.where(mask, x, mom.p_bar) - mom.p_bar
    r = r - r.mean(axis=0, keepdims=True)
    sd = r.std(axis=0)
    ok = sd > 1e-9
    if ok.sum() < 2:
        return 0.0
    z = r[:, ok] / sd[ok]
    c = (z.T @ z) / max(x.shape[0], 1)
    np.fill_diagonal(c, 0.0)
    return float(np.abs(c).max())


def _stat_score_sd(x: np.ndarray, mom: _Moments, mask: np.ndarray) -> float:
    """总分标准差。"""
    return float(np.where(mask, x, mom.p_bar).sum(axis=1).std())


def _stat_joint_max(x: np.ndarray, mom: _Moments, mask: np.ndarray) -> float:
    """题目对共同答对率与模型隐含联合率的最大绝对偏差。"""
    j = x.shape[1]
    if j < 2:
        return 0.0
    obs, n_ab = _observed_joint(x, mask)
    iu = _pair_index(j)
    ok = n_ab[iu] >= 2
    if not ok.any():
        return 0.0
    return float(np.abs(obs[iu][ok] - mom.joint[iu][ok]).max())


_STAT_FUNCS = {
    "pair_srmsr": _stat_pair_srmsr,
    "q3_max": _stat_q3_max,
    "score_sd": _stat_score_sd,
    "joint_max": _stat_joint_max,
}


def _clone_params(params: dict) -> dict:
    """深拷贝模型参数字典（值可能是 ndarray，也可能是"每题一个数组"的列表）。"""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = v.astype(np.float64, copy=True)
        elif isinstance(v, (list, tuple)):
            out[k] = [np.asarray(x, dtype=np.float64).copy() for x in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------- #
# 主函数
# ---------------------------------------------------------------------- #
def posterior_predictive_check(
    model: CDModel,
    responses: np.ndarray,
    *,
    prior: Optional[np.ndarray] = None,
    alpha_hat: Optional[np.ndarray] = None,
    statistics: Sequence[str] = PPMC_STATISTICS,
    n_replicates: int = 500,
    seed: int = 2024,
    alpha: float = 0.05,
    max_items_pair: int = 150,
    refit: bool = True,
    refit_max_iter: int = 8,
    refit_tol: float = 1e-6,
) -> PPMCResult:
    """参数化后验预测检验。

    Parameters
    ----------
    model : CDModel
        已标定的响应模型。
    responses : ndarray, shape (n_examinees, n_items)
        0/1 作答矩阵；含 NaN 的行按"仅用已作答题"计算统计量。
    prior : ndarray, optional
        属性模式先验，用于样本的属性后验。
    alpha_hat : ndarray, optional
        直接给定属性模式（``(n, K)`` 或模式索引 ``(n,)``），跳过后验抽样。
    statistics : sequence of str
        要检验的统计量，可选 ``pair_srmsr`` / ``q3_max`` / ``joint_max`` / ``score_sd``。
    n_replicates : int
        复制数据集份数，默认 500（PPP 的蒙特卡洛误差约 0.02）。
    seed : int
    alpha : float
        双尾判读阈值。
    max_items_pair : int
        题目数超过此值时跳过基于题目对的统计量（计算与内存为 O(J²)）。
    refit : bool
        复制数据集是否**重新估计题目参数**。默认 ``True``。
        关闭后只重估属性、沿用原参数，速度快数倍，但边际类统计量
        会让题目对联合率相对"模型隐含值"出现轻微漂移，PPP 判读不如默认设置可靠。
    refit_max_iter : int
        重估时每次热启动 EM 的迭代上限（默认 8，实测足够收敛）。
    refit_tol : float
        概率矩阵最大变化量的收敛阈值。

    Returns
    -------
    PPMCResult

    Examples
    --------
    >>> res = posterior_predictive_check(model, X, n_replicates=200)  # doctest: +SKIP
    >>> print(res.summary())                                          # doctest: +SKIP
    >>> res.misfit_stats                                              # doctest: +SKIP
    """
    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维矩阵 (n_examinees, n_items)")
    n, j = resp.shape
    if j != model.n_items:
        raise ValueError(f"作答矩阵列数 ({j}) 与模型题目数 ({model.n_items}) 不一致")

    stat_names = [s for s in statistics if s in _STAT_FUNCS]
    if not stat_names:
        raise ValueError(f"没有可用的统计量，可选：{list(_STAT_FUNCS)}")
    if n_replicates < 20:
        raise ValueError("n_replicates 至少为 20，否则 PPP 的蒙特卡洛误差过大")

    # ---- 缺失值 ----
    # 逐题观测均值填补：按 0 填会把边际答对率系统性拉低，进而污染题目对统计量
    obs = np.isfinite(resp)
    x_obs = np.where(obs, resp, 0.0)
    col_mean = np.array([
        x_obs[obs[:, k], k].mean() if obs[:, k].any() else 0.0 for k in range(j)
    ])
    x_filled = np.where(obs, x_obs, col_mean[None, :])

    prob = np.asarray(model._get_prob(), dtype=np.float64)     # (S, J)
    s_total = prob.shape[0]

    # ---- 属性先验 ----
    if prior is None:
        prior_vec = np.full(s_total, 1.0 / s_total, dtype=np.float64)
    else:
        pr = np.asarray(prior, dtype=np.float64).ravel()
        if pr.size != s_total:
            raise ValueError(f"prior 长度 ({pr.size}) 应等于属性模式数 ({s_total})")
        pr = np.clip(pr, 1e-300, None)
        prior_vec = pr / pr.sum()

    def _posterior(x: np.ndarray, p_mat: np.ndarray, pr_vec: np.ndarray) -> np.ndarray:
        """给定作答矩阵与概率矩阵，求属性模式后验，形状 ``(n, 2^K)``。

        缺失位按"不计入似然"处理（只累加已作答题目的对数似然），避免填补值
        参与后验计算造成偏差。
        """
        xo = np.where(obs, x, 0.0)
        ll = (xo @ np.log(p_mat).T + (obs - xo) @ np.log1p(-p_mat).T
              + np.log(np.clip(pr_vec, 1e-300, None)))
        ll -= ll.max(axis=1, keepdims=True)
        w = np.exp(ll)
        return w / w.sum(axis=1, keepdims=True)

    # ---- 属性后验：外部给定 vs 由数据估计 ----
    if alpha_hat is not None:
        a = np.asarray(alpha_hat)
        if a.ndim == 1:
            idx = a.astype(np.int64)                          # 已是模式索引
            post_obs = np.zeros((n, s_total), dtype=np.float64)
            post_obs[np.arange(n), idx] = 1.0
        elif a.shape[1] == s_total:
            w_given = np.clip(a.astype(np.float64), 0.0, None)
            rs = w_given.sum(axis=1, keepdims=True)
            post_obs = np.divide(w_given, np.where(rs > 0, rs, 1.0))
        else:                                                 # (n, K) 硬模式
            w = 1 << np.arange(a.shape[1] - 1, -1, -1)
            idx = (a.astype(np.int64) * w).sum(axis=1)
            post_obs = np.zeros((n, s_total), dtype=np.float64)
            post_obs[np.arange(n), idx] = 1.0
        alpha_source = "given"
    else:
        # 经验贝叶斯：先验 π(α) 与后验互为不动点，迭代几次即可
        pr = prior_vec
        for _ in range(3):
            post_obs = _posterior(x_filled, prob, pr)
            pr = post_obs.mean(axis=0)
            pr = np.clip(pr, 1e-12, None)
            pr = pr / pr.sum()
        prior_vec = pr
        alpha_source = "estimated"

    rng = np.random.default_rng(seed)

    def _draw(state: np.ndarray) -> np.ndarray:
        """按行从后验分布抽一个属性模式索引。"""
        cum = np.cumsum(state, axis=1)
        u = rng.random(state.shape[0])
        return np.minimum((cum < u[:, None]).sum(axis=1), s_total - 1)

    # ---- 复制侧是否重估题目参数 ----
    # 只重估 α 是不够的：EM 极大化边际似然后 θ̂ 会让 Σ_i p̄_ij 精确等于 x̄_j，
    # 于是**任何**数据的观测侧边际统计量都≈0，而复制侧 θ 未重估、没有这个性质，
    # PPP 被系统性推向 1，正确模型也会被判失配。复制侧必须同样重估 θ。
    refit_fn = None
    if refit and alpha_source == "estimated":
        from ..calibration import EMCalibrator

        try:
            cal = EMCalibrator(model.q, model_type=model.name)
        except Exception:                                     # 模型类型不支持重估
            cal = None
        if cal is not None:
            params0 = _clone_params(model.get_params())

            def _refit(x_star: np.ndarray):
                cal.prior = prior_vec.copy()
                params = _clone_params(params0)
                p_mat = cal._prob_matrix(params)
                for _ in range(int(refit_max_iter)):
                    post = _posterior(x_star, p_mat, cal.prior)
                    params = cal._m_step(x_star, post, params)
                    cal._update_prior(post)
                    p_new = cal._prob_matrix(params)
                    gap = float(np.abs(p_new - p_mat).max())
                    p_mat = p_new
                    if gap < float(refit_tol):
                        break
                return p_mat, _posterior(x_star, p_mat, cal.prior)

            refit_fn = _refit

    # ---- 观测统计量：条件于"后验期望答对概率"，而非 MAP 硬模式 ----
    # 用 MAP 硬模式会因"模式本身就是挑出来最拟合该人的"而过拟合，
    # 观测统计量被压小、PPP 系统性趋近 1。改用后验期望矩，两侧口径一致。
    mom_obs = _moments(post_obs, prob)
    observed: Dict[str, float] = {}
    for name in stat_names:
        if name in _PAIR_STATS and j > max_items_pair:
            continue
        observed[name] = _STAT_FUNCS[name](x_filled, mom_obs, obs)
    stat_names = [s for s in stat_names if s in observed]

    # ---- 复制数据集 ----
    replicated = np.zeros((n_replicates, len(stat_names)), dtype=np.float64)
    for r in range(n_replicates):
        star = _draw(post_obs)                    # 抽属性模式
        p_star = prob[star]                       # (n, J)
        x_star = (rng.random((n, j)) < p_star).astype(np.float64)
        # 缺失位与观测侧用**同一组**列均值填补，保证两侧统计量口径完全一致
        x_star = np.where(obs, x_star, col_mean[None, :])
        if alpha_source == "given":
            # 属性与参数都视为已知：复制侧沿用同一套矩，无需重算
            mom_rep = mom_obs
        elif refit_fn is not None:
            # 属性与题目参数都由数据估计：复制数据走完全相同的估计流程
            p_mat, post_star = refit_fn(x_star)
            mom_rep = _moments(post_star, p_mat)
        else:
            # 快速近似：只重估属性、沿用题目参数（边际类统计量会有已知偏差）
            mom_rep = _moments(_posterior(x_star, prob, prior_vec), prob)
        for c, name in enumerate(stat_names):
            replicated[r, c] = _STAT_FUNCS[name](x_star, mom_rep, obs)

    # ---- PPP ----
    ppp: Dict[str, float] = {}
    for c, name in enumerate(stat_names):
        col = replicated[:, c]
        ppp[name] = float((col >= observed[name]).mean())

    return PPMCResult(
        ppp=ppp,
        observed=observed,
        replicated=replicated,
        stat_names=stat_names,
        n_replicates=n_replicates,
        n_examinees=n,
        n_items=j,
        seed=seed,
        alpha=alpha,
        alpha_source=alpha_source,
        used_refit=refit_fn is not None,
    )
