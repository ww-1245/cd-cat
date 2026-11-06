"""曝光控制 (Exposure Control)。

CD-CAT 中，信息量大的题目会被反复选中，导致：
1. **题库安全性风险** —— 高频题目一旦泄漏，测验效度受损。
2. **题库利用率低** —— 大量题目从不被使用，浪费建设成本。
3. **被试体验下降** —— 同类题目反复出现。

本模块提供三种经典曝光控制策略：

- ``SympsonHetter``   Sympson-Hetter 法：按预计算的选择概率随机屏蔽题目。
- ``RandomesqueExposure`` 从准则值前 N 名中随机选一题，用随机性摊平曝光。
- ``MaxExposureLimit``   硬上限：曝光率超过阈值的题目直接从候选集剔除。
- ``StratifiedExposure`` 分层：按曝光率把题目分档，优先选曝光率低的档。

用法：把曝光控制器传给 ``CDCATSimulator(exposure_control=...)``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = [
    "ExposureControl",
    "SympsonHetter",
    "RandomesqueExposure",
    "MaxExposureLimit",
    "StratifiedExposure",
    "NoExposureControl",
    "create_exposure_control",
]


class ExposureControl(ABC):
    """曝光控制基类。"""

    name: str = "base"

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.exposure_count = np.zeros(n_items, dtype=np.int64)
        self.admin_count = 0
        self.n_examinees_done = 0

    # ------------------------------------------------------------------ #
    def update(self, item_idx: int, was_administered: bool = True) -> None:
        """施测后更新计数。"""
        if was_administered:
            self.exposure_count[item_idx] += 1
            self.admin_count += 1

    def exposure_rate(self) -> np.ndarray:
        """当前各题曝光率。

        以「已施测的被试数」为分母（而非施测总题次），
        这样曝光率可直接解读为"有多大比例的被试见过这道题"，
        与 Sympson-Hetter 等经典方法中 $K_j$ 的定义一致。
        """
        if self.n_examinees_done == 0:
            return np.zeros(self.n_items)
        return self.exposure_count / self.n_examinees_done

    def end_examinee(self) -> None:
        """一名被试施测结束时调用，用于更新分母。"""
        self.n_examinees_done += 1

    def reset(self) -> None:
        self.exposure_count = np.zeros(self.n_items, dtype=np.int64)
        self.admin_count = 0
        self.n_examinees_done = 0

    @abstractmethod
    def filter_candidates(
        self,
        candidates: np.ndarray,
        criterion_values: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """返回允许参与的候选题目索引（用于过滤）。"""

    def pick(
        self,
        candidates: np.ndarray,
        criterion_values: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """在准则值和曝光控制共同作用下选出题目。

        默认实现：先过滤，再取准则值最大者；子类可覆盖以实现随机化。
        """
        allowed = self.filter_candidates(candidates, criterion_values, rng)
        if len(allowed) == 0:
            allowed = candidates              # 全部被屏蔽时退回全集
        # 在 allowed 中找到准则值最大的
        pos = np.array([np.where(candidates == a)[0][0] for a in allowed])
        best_local = np.argmax(criterion_values[pos])
        return int(allowed[best_local])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_items={self.n_items})"


class NoExposureControl(ExposureControl):
    """不施加任何曝光控制（基线对照用）。"""

    name = "无控制"

    def filter_candidates(self, candidates, criterion_values, rng):
        return candidates


class MaxExposureLimit(ExposureControl):
    """硬上限策略：曝光率超过 ``max_rate`` 的题目直接剔除。

    Parameters
    ----------
    max_rate : float
        允许的最高曝光率，如 0.2 表示任何题目最多负担 20% 的施测量。
    warmup : int
        前 ``warmup`` 次施测不做限制，避免早期无题可选。
    """

    name = "曝光硬上限"

    def __init__(self, n_items: int, max_rate: float = 0.2, warmup: int = 0) -> None:
        super().__init__(n_items)
        self.max_rate = max_rate
        self.warmup = warmup

    def filter_candidates(self, candidates, criterion_values, rng):
        if self.admin_count < self.warmup:
            return candidates
        rate = self.exposure_rate()
        allowed = candidates[rate[candidates] < self.max_rate]
        return allowed if len(allowed) > 0 else candidates

    def __repr__(self) -> str:
        return f"MaxExposureLimit(max_rate={self.max_rate}, warmup={self.warmup})"


class SympsonHetter(ExposureControl):
    """Sympson-Hetter 曝光控制 (Sympson & Hetter, 1985)。

    对每道题设一个"允许施测概率" $K_j$。选题时以概率 $K_j$ 允许该题入选，
    以 $1-K_j$ 将其临时屏蔽（若被屏蔽则退而选次优题）。

    经典做法是通过预先的蒙特卡洛运行标定 $K_j$，使实际曝光率逼近目标。
    本实现提供 ``calibrate()`` 方法在给定仿真器上自动标定：

        K_j^{(t+1)} = K_j^{(t)} · (target_rate / observed_rate_j)

    并进行裁剪与平滑，迭代至收敛。
    """

    name = "Sympson-Hetter"

    def __init__(
        self,
        n_items: int,
        target_rate: float = 0.2,
        k_init: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(n_items)
        self.target_rate = target_rate
        self.k = (
            np.full(n_items, 1.0) if k_init is None else np.asarray(k_init, dtype=float).copy()
        )
        self.k = np.clip(self.k, 0.01, 1.0)

    def filter_candidates(self, candidates, criterion_values, rng):
        # 以概率 k_j 允许；至少保留一道题避免空集
        allow_mask = rng.random(len(candidates)) < self.k[candidates]
        allowed = candidates[allow_mask]
        if len(allowed) == 0:
            # 至少保留准则值最高的一题
            best = candidates[int(np.argmax(criterion_values))]
            allowed = np.array([best])
        return allowed

    def calibrate(
        self,
        simulator,
        n_examinees: int = 200,
        n_replications: int = 3,
        n_iterations: int = 8,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """自动标定 K_j 参数。

        在给定仿真器上反复运行，逐步收紧 K_j 使曝光率逼近目标。

        Returns
        -------
        ndarray
            标定后的 K 向量。
        """
        rng = np.random.default_rng(seed)
        for it in range(n_iterations):
            self.reset()
            result = simulator.run(
                n_examinees=n_examinees, n_replications=n_replications,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            obs = result.item_exposure_rate()
            # 只对超出目标的题目收紧
            over = obs > self.target_rate
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(over, self.target_rate / np.maximum(obs, 1e-9), 1.0)
            self.k = np.clip(self.k * (0.5 + 0.5 * ratio), 0.01, 1.0)
            if verbose:
                used = obs > 0
                print(
                    f"  [SH 标定] iter {it+1}: 最大曝光率={obs.max():.4f}, "
                    f"利用率={used.mean():.4f}, K 均值={self.k.mean():.4f}"
                )
        return self.k

    def __repr__(self) -> str:
        return f"SympsonHetter(target_rate={self.target_rate})"


class RandomesqueExposure(ExposureControl):
    """随机化策略：从准则值最高的前 ``top_n`` 题中等概率随机选一题。

    Parameters
    ----------
    top_n : int
        候选池大小。
    weighting : {'uniform', 'proportional'}
        ``'uniform'`` 等概率抽取；``'proportional'`` 按准则值加权（值越大越可能被选中）。
    """

    name = "随机化选题"

    def __init__(self, n_items: int, top_n: int = 5, weighting: str = "uniform") -> None:
        super().__init__(n_items)
        self.top_n = top_n
        self.weighting = weighting

    def pick(self, candidates, criterion_values, rng):
        n = min(self.top_n, len(candidates))
        # 取准则值前 n 名
        order = np.argsort(-criterion_values)[:n]
        pool = candidates[order]
        if n == 1:
            return int(pool[0])
        if self.weighting == "proportional":
            w = criterion_values[order]
            w = np.maximum(w - w.min() + 1e-9, 1e-9)
            p = w / w.sum()
            return int(rng.choice(pool, p=p))
        return int(rng.choice(pool))

    def filter_candidates(self, candidates, criterion_values, rng):
        return candidates

    def __repr__(self) -> str:
        return f"RandomesqueExposure(top_n={self.top_n}, weighting='{self.weighting}')"


class StratifiedExposure(ExposureControl):
    """分层曝光控制 (Chang & Ying, 1999 思想在 CD-CAT 的迁移)。

    按题目当前曝光率分档，优先从曝光率最低的档中选题。
    实现上采用"按曝光率加权的准则值惩罚"：

        adjusted_value_j = criterion_j − λ · exposure_rate_j

    通过 ``penalty`` 控制曝光均衡强度，比硬阈值更平滑。
    """

    name = "分层曝光"

    def __init__(self, n_items: int, penalty: float = 0.5, warmup: int = 20) -> None:
        super().__init__(n_items)
        self.penalty = penalty
        self.warmup = warmup

    def filter_candidates(self, candidates, criterion_values, rng):
        return candidates

    def pick(self, candidates, criterion_values, rng):
        if self.admin_count < self.warmup:
            return int(candidates[int(np.argmax(criterion_values))])
        rate = self.exposure_rate()[candidates]
        # 归一化准则值到 [0,1]，再加惩罚
        cv = criterion_values - criterion_values.min()
        scale = cv.max() if cv.max() > 0 else 1.0
        adjusted = cv / scale - self.penalty * rate
        return int(candidates[int(np.argmax(adjusted))])

    def __repr__(self) -> str:
        return f"StratifiedExposure(penalty={self.penalty}, warmup={self.warmup})"


_REGISTRY = {
    "none": NoExposureControl,
    "no": NoExposureControl,
    "noexposurecontrol": NoExposureControl,
    "maxlimit": MaxExposureLimit,
    "max": MaxExposureLimit,
    "sympsonhetter": SympsonHetter,
    "sh": SympsonHetter,
    "randomesque": RandomesqueExposure,
    "random": RandomesqueExposure,
    "stratified": StratifiedExposure,
    "strata": StratifiedExposure,
}


def create_exposure_control(name: str, n_items: int, **kwargs) -> ExposureControl:
    """按名称创建曝光控制器。

    Examples
    --------
    >>> ctrl = create_exposure_control('maxlimit', 60, max_rate=0.15)  # doctest: +SKIP
    >>> ctrl = create_exposure_control('randomesque', 60, top_n=5)     # doctest: +SKIP
    """
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key not in _REGISTRY:
        raise ValueError(f"未知曝光控制 '{name}'，可选: {sorted(set(_REGISTRY))}")
    return _REGISTRY[key](n_items, **kwargs)
