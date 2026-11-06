"""终止规则 (Termination Rules)。

CD-CAT 需要在"诊断足够准"和"避免过多作答"之间平衡。常见三类规则：

1. **后验阈值**：所有属性的边际概率都 > 阈值 (如 0.8)，或最大后验概率 > 阈值。
2. **固定题量**：达到预设题数即停止。
3. **稳定性**：连续 N 题估计模式不再变化。

实际系统通常组合使用：满足任一即停止（或全部满足才停止）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

__all__ = [
    "TerminationRule",
    "PosteriorThreshold",
    "FixedLength",
    "StabilityRule",
    "AttributeConfidence",
    "PatternConfidence",
    "CompositeRule",
    "create_rule",
]


class TerminationRule(ABC):
    """终止规则基类。"""

    name: str = "base"
    rule_type: str = "max"     # 'min' 表示最小题量保护，'max' 表示正常终止

    @abstractmethod
    def should_stop(self, state: dict) -> bool:
        """判断是否应终止。

        Parameters
        ----------
        state : dict
            施测状态，包含键：
            - ``posterior``: 当前后验分布
            - ``marginal``: 属性边际概率
            - ``estimate``: 当前估计模式
            - ``n_answered``: 已作答题数
            - ``history``: 历史估计模式列表
            - ``n_items``: 题库规模
            - ``min_items``: 最少题量

        Returns
        -------
        bool
        """

    def reason(self, state: dict) -> str:
        """返回终止原因描述。"""
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PosteriorThreshold(TerminationRule):
    """后验概率阈值规则。

    Parameters
    ----------
    threshold : float
        置信度阈值，默认 0.8。
    mode : {'max', 'all_attrs'}
        - ``'max'``: 最大后验概率 > threshold 即停止（整体模式确定）。
        - ``'all_attrs'``: 所有属性边际概率都 > threshold 才停止（更严格）。
    """

    name = "PosteriorThreshold"

    def __init__(self, threshold: float = 0.8, mode: str = "all_attrs") -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold 需在 (0, 1] 区间")
        if mode not in {"max", "all_attrs"}:
            raise ValueError("mode 必须是 'max' 或 'all_attrs'")
        self.threshold = threshold
        self.mode = mode

    def should_stop(self, state: dict) -> bool:
        if state["n_answered"] == 0:
            return False
        if self.mode == "max":
            return bool(state["posterior"].max() >= self.threshold)
        marginal = np.asarray(state["marginal"])
        # 所有属性要么几乎掌握要么几乎未掌握
        return bool(np.all((marginal >= self.threshold) | (marginal <= 1 - self.threshold)))

    def reason(self, state: dict) -> str:
        if self.mode == "max":
            return f"最大后验概率达到 {self.threshold}"
        return f"所有属性边际概率达到 {self.threshold} 置信水平"

    def __repr__(self) -> str:
        return f"PosteriorThreshold(threshold={self.threshold}, mode='{self.mode}')"


class FixedLength(TerminationRule):
    """固定题量规则：答满 ``n_items`` 题即停止。"""

    name = "FixedLength"

    def __init__(self, n_items: int) -> None:
        if n_items < 1:
            raise ValueError("n_items 必须 >= 1")
        self.n_items = n_items

    def should_stop(self, state: dict) -> bool:
        return state["n_answered"] >= self.n_items

    def reason(self, state: dict) -> str:
        return f"达到固定题量 {self.n_items}"

    def __repr__(self) -> str:
        return f"FixedLength(n_items={self.n_items})"


class StabilityRule(TerminationRule):
    """估计稳定性规则：连续 ``n_stable`` 题估计模式不再变化即停止。"""

    name = "StabilityRule"

    def __init__(self, n_stable: int = 5) -> None:
        self.n_stable = n_stable

    def should_stop(self, state: dict) -> bool:
        hist: List = state.get("history", [])
        if len(hist) < self.n_stable + 1:
            return False
        recent = hist[-(self.n_stable + 1):]
        return all(int(p) == int(recent[-1]) for p in recent)

    def reason(self, state: dict) -> str:
        return f"估计模式连续 {self.n_stable} 题未变化"

    def __repr__(self) -> str:
        return f"StabilityRule(n_stable={self.n_stable})"


class AttributeConfidence(TerminationRule):
    """属性级置信规则：每个属性的边际概率都足够"确定"。

    与 ``PosteriorThreshold(mode='all_attrs')`` 的区别在于，
    这里用**属性熵**度量：每个属性的二元熵都低于给定水平。
    """

    name = "AttributeConfidence"

    def __init__(self, max_entropy: float = 0.3) -> None:
        self.max_entropy = max_entropy

    def should_stop(self, state: dict) -> bool:
        if state["n_answered"] == 0:
            return False
        marg = np.clip(np.asarray(state["marginal"]), 1e-9, 1 - 1e-9)
        ent = -(marg * np.log(marg) + (1 - marg) * np.log(1 - marg))
        ent_norm = ent / np.log(2)     # 归一化到 [0,1]
        return bool(np.all(ent_norm <= self.max_entropy))

    def reason(self, state: dict) -> str:
        return f"所有属性归一化熵 <= {self.max_entropy}"

    def __repr__(self) -> str:
        return f"AttributeConfidence(max_entropy={self.max_entropy})"


class CompositeRule(TerminationRule):
    """组合规则。

    Parameters
    ----------
    rules : list of TerminationRule
    how : {'any', 'all'}
        ``'any'`` 任一满足即停止；``'all'`` 全部满足才停止。
    min_items : int
        最小题量保护，未达到时绝不停止。
    max_items : int, optional
        最大题量硬上限，达到即停止。
    """

    name = "CompositeRule"

    def __init__(
        self,
        rules: List[TerminationRule],
        how: str = "any",
        min_items: int = 5,
        max_items: Optional[int] = None,
    ) -> None:
        if how not in {"any", "all"}:
            raise ValueError("how 必须是 'any' 或 'all'")
        self.rules = list(rules)
        self.how = how
        self.min_items = min_items
        self.max_items = max_items
        self._fired: List[str] = []

    def should_stop(self, state: dict) -> bool:
        n = state["n_answered"]
        if self.max_items is not None and n >= self.max_items:
            self._fired = [f"达到最大题量 {self.max_items}"]
            return True
        if n < self.min_items:
            return False
        flags = [r.should_stop(state) for r in self.rules]
        if self.how == "any":
            stop = any(flags)
        else:
            stop = all(flags) if flags else False
        self._fired = [r.reason(state) for r, f in zip(self.rules, flags) if f]
        return stop

    def reason(self, state: dict) -> str:
        if self._fired:
            return "; ".join(self._fired)
        return self.name

    def __repr__(self) -> str:
        return f"CompositeRule({len(self.rules)} rules, how='{self.how}', min={self.min_items}, max={self.max_items})"


class PatternConfidence(TerminationRule):
    """模式级置信规则：最大后验概率 AND 所有属性边际概率都达标。

    这是实践中最稳健的默认规则——单看属性边际（``all_attrs``）容易过早
    停止：各属性边际都 >0.8 并不意味着它们的**联合**模式已经确定
    （属性间后验可能仍存在组合不确定性）。同时要求模式级后验也达标，
    能显著提高模式判准率，代价是略增题量。
    """

    name = "PatternConfidence"

    def __init__(self, threshold: float = 0.8) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold 需在 (0, 1] 区间")
        self.threshold = threshold

    def should_stop(self, state: dict) -> bool:
        if state["n_answered"] == 0:
            return False
        post_ok = state["posterior"].max() >= self.threshold
        marginal = np.asarray(state["marginal"])
        attr_ok = np.all((marginal >= self.threshold) | (marginal <= 1 - self.threshold))
        return bool(post_ok and attr_ok)

    def reason(self, state: dict) -> str:
        return f"模式后验与属性边际均达到 {self.threshold}"

    def __repr__(self) -> str:
        return f"PatternConfidence(threshold={self.threshold})"


_RULE_REGISTRY = {
    "posterior": PosteriorThreshold,
    "posteriorthreshold": PosteriorThreshold,
    "fixed": FixedLength,
    "fixedlength": FixedLength,
    "stability": StabilityRule,
    "stabilityrule": StabilityRule,
    "confidence": AttributeConfidence,
    "attributeconfidence": AttributeConfidence,
    "patternconfidence": PatternConfidence,
    "pattern": PatternConfidence,
    "composite": CompositeRule,
}


def create_rule(name: str, **kwargs) -> TerminationRule:
    """按名称创建终止规则。"""
    key = name.strip().lower()
    if key not in _RULE_REGISTRY:
        raise ValueError(f"未知终止规则 '{name}'，可选: {sorted(set(_RULE_REGISTRY))}")
    return _RULE_REGISTRY[key](**kwargs)
