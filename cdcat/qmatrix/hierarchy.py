"""属性层级关系（Attribute Hierarchy Method, AHM）。

为什么需要它
------------
前面所有模块都默认 **2^K 个属性模式全部可能**。但教学实践中属性之间
几乎总有**先决关系**：不掌握"通分"就不可能掌握"异分母分数加减"；
不掌握"一元一次方程"就学不了"函数图像"。

把这类关系显式写进模型，会带来三个实打实的好处：

1. **合法模式数骤减**：K=6 时 64 个模式，若为线型层级只剩 7 个 ——
   后验只需在这 7 个里分配，判准率与收敛速度都会提升。
2. **PMR 上限重算**：可辨识性分析要在合法模式集合内做，
   原来的上限数字会变（通常变高，因为需要区分的模式少了）。
3. **诊断结果可解释**：不会出现"掌握了分数乘除但不会通分"这种
   教学上不可能的结论 —— 后验被约束在合法集合内。

约定
----
有向边 ``(a, b)`` 表示 **a 是 b 的先备属性**，即「掌握 b ⟹ 已掌握 a」。
因此线型层级 通分→约分→分数加减 写作 ``[(0,1), (1,2)]``。

合法模式的判定只需检查**直接边**：由链条传递性，
``α_c=1 ⟹ α_b=1 ⟹ α_a=1`` 自动成立。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import enumerate_attribute_patterns

__all__ = [
    "AttributeHierarchy",
    "HierarchyConsistencyReport",
    "check_hierarchy_consistency",
]


# ====================================================================== #
# 层级定义
# ====================================================================== #
@dataclass
class AttributeHierarchy:
    """属性层级结构。

    Parameters
    ----------
    n_attrs : int
        属性个数 K。
    edges : list of (int, int)
        有向边 ``(先备属性, 后继属性)``，即掌握后继必须先掌握先备。
    attr_names : list of str, optional
    structure : str
        结构类型标签（``none`` / ``linear`` / ``convergent`` / ``divergent`` / ``custom``）。

    Examples
    --------
    >>> h = AttributeHierarchy.linear(4)
    >>> h.n_legal_patterns
    5
    >>> h.legal_patterns()
    array([[0, 0, 0, 0],
           [1, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 1, 1, 0],
           [1, 1, 1, 1]], dtype=int8)
    """

    n_attrs: int
    edges: List[Tuple[int, int]] = field(default_factory=list)
    attr_names: Optional[List[str]] = None
    structure: str = "custom"

    def __post_init__(self) -> None:
        self.n_attrs = int(self.n_attrs)
        if self.n_attrs < 1:
            raise ValueError("n_attrs 必须 >= 1")
        cleaned: List[Tuple[int, int]] = []
        for a, b in self.edges:
            a, b = int(a), int(b)
            if not (0 <= a < self.n_attrs and 0 <= b < self.n_attrs):
                raise ValueError(f"边 ({a}, {b}) 超出属性范围 [0, {self.n_attrs})")
            if a == b:
                raise ValueError(f"不允许自环 ({a}, {b})")
            if (a, b) not in cleaned:
                cleaned.append((a, b))
        self.edges = cleaned
        self._cache: Dict[str, np.ndarray] = {}
        if self._has_cycle():
            raise ValueError("层级关系存在环，属性先决关系必须是 DAG")

    # ------------------------------------------------------------------ #
    # 构造快捷方式
    # ------------------------------------------------------------------ #
    @classmethod
    def none(cls, n_attrs: int, attr_names: Optional[Sequence[str]] = None) -> "AttributeHierarchy":
        """无层级（2^K 个模式全部合法，等同于此前的默认行为）。"""
        return cls(n_attrs, [], list(attr_names) if attr_names else None, "none")

    @classmethod
    def linear(cls, n_attrs: int,
               attr_names: Optional[Sequence[str]] = None) -> "AttributeHierarchy":
        """线型：A1 → A2 → … → AK，共 K+1 个合法模式。"""
        return cls(n_attrs, [(i, i + 1) for i in range(n_attrs - 1)],
                   list(attr_names) if attr_names else None, "linear")

    @classmethod
    def convergent(cls, n_attrs: int,
                   attr_names: Optional[Sequence[str]] = None) -> "AttributeHierarchy":
        """收敛型：A1..A(K-1) 全部先备于 AK。

        即末位属性是"综合能力"，掌握它意味着前面全部掌握。
        """
        if n_attrs < 2:
            return cls.none(n_attrs, attr_names)
        return cls(n_attrs, [(i, n_attrs - 1) for i in range(n_attrs - 1)],
                   list(attr_names) if attr_names else None, "convergent")

    @classmethod
    def divergent(cls, n_attrs: int,
                  attr_names: Optional[Sequence[str]] = None) -> "AttributeHierarchy":
        """发散型：A1 是所有其他属性的先备。

        即首位属性是"基础能力"，不会它后面的都免谈。
        """
        if n_attrs < 2:
            return cls.none(n_attrs, attr_names)
        return cls(n_attrs, [(0, i) for i in range(1, n_attrs)],
                   list(attr_names) if attr_names else None, "divergent")

    @classmethod
    def from_edges(cls, n_attrs: int, edges: Sequence[Tuple[int, int]],
                   attr_names: Optional[Sequence[str]] = None) -> "AttributeHierarchy":
        """直接给边表。"""
        return cls(n_attrs, list(edges), list(attr_names) if attr_names else None, "custom")

    @classmethod
    def from_adjacency(cls, matrix, attr_names: Optional[Sequence[str]] = None) -> "AttributeHierarchy":
        """由邻接矩阵构造（``A[a, b] = 1`` 表示 a 是 b 的先备）。"""
        m = np.asarray(matrix, dtype=int)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("邻接矩阵必须是方阵")
        k = m.shape[0]
        edges = [(int(a), int(b)) for a in range(k) for b in range(k) if m[a, b]]
        return cls(k, edges, list(attr_names) if attr_names else None, "custom")

    # ------------------------------------------------------------------ #
    # 结构量
    # ------------------------------------------------------------------ #
    def _has_cycle(self) -> bool:
        """拓扑排序判环。"""
        k = self.n_attrs
        indeg = [0] * k
        adj: List[List[int]] = [[] for _ in range(k)]
        for a, b in self.edges:
            adj[a].append(b)
            indeg[b] += 1
        stack = [i for i in range(k) if indeg[i] == 0]
        seen = 0
        while stack:
            n = stack.pop()
            seen += 1
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    stack.append(m)
        return seen != k

    @property
    def is_empty(self) -> bool:
        """是否无层级（所有模式合法）。"""
        return len(self.edges) == 0

    def direct_prerequisites(self, attr: int) -> List[int]:
        """该属性的**直接**先备属性。"""
        return [a for a, b in self.edges if b == attr]

    def direct_dependents(self, attr: int) -> List[int]:
        """以该属性为直接先备的后继属性。"""
        return [b for a, b in self.edges if a == attr]

    def reachability(self) -> np.ndarray:
        """可达矩阵 R，``R[a, b] = 1`` 表示 a 是 b 的（传递）先备。"""
        if "R" in self._cache:
            return self._cache["R"]
        k = self.n_attrs
        r = np.zeros((k, k), dtype=bool)
        for a, b in self.edges:
            r[a, b] = True
        # 布尔传递闭包（Warshall）
        for mid in range(k):
            r |= np.outer(r[:, mid], r[mid, :])
        np.fill_diagonal(r, False)
        self._cache["R"] = r
        return r

    def all_prerequisites(self, attr: int) -> List[int]:
        """该属性的**全部**（含传递）先备属性。"""
        return np.where(self.reachability()[:, attr])[0].tolist()

    @property
    def levels(self) -> np.ndarray:
        """每个属性的层级号（无先备者为 0，逐层递增）。"""
        k = self.n_attrs
        level = np.zeros(k, dtype=int)
        # 按拓扑序松弛，最多 K 轮
        for _ in range(k):
            changed = False
            for a, b in self.edges:
                if level[b] < level[a] + 1:
                    level[b] = level[a] + 1
                    changed = True
            if not changed:
                break
        return level

    # ------------------------------------------------------------------ #
    # 合法模式
    # ------------------------------------------------------------------ #
    def legal_mask(self) -> np.ndarray:
        """长度 2^K 的布尔掩码，标记哪些模式满足层级约束。"""
        if "mask" in self._cache:
            return self._cache["mask"]
        if self.is_empty:
            mask = np.ones(1 << self.n_attrs, dtype=bool)
            self._cache["mask"] = mask
            return mask
        patterns = enumerate_attribute_patterns(self.n_attrs).astype(bool)
        mask = np.ones(patterns.shape[0], dtype=bool)
        for a, b in self.edges:
            # 掌握 b 却未掌握 a -> 非法
            mask &= (~patterns[:, b]) | patterns[:, a]
        self._cache["mask"] = mask
        return mask

    def legal_indices(self) -> np.ndarray:
        """合法模式的编号数组。"""
        return np.where(self.legal_mask())[0]

    @property
    def n_legal_patterns(self) -> int:
        """合法模式个数。"""
        return int(self.legal_mask().sum())

    def legal_patterns(self) -> np.ndarray:
        """全部合法属性模式，形状 ``(n_legal, K)``。"""
        return enumerate_attribute_patterns(self.n_attrs)[self.legal_mask()]

    def is_legal(self, pattern: Sequence[int]) -> bool:
        """单个模式是否满足层级约束。"""
        p = np.asarray(pattern, dtype=int).ravel()
        for a, b in self.edges:
            if p[b] == 1 and p[a] == 0:
                return False
        return True

    def project(self, pattern: Sequence[int]) -> np.ndarray:
        """把任意模式**映射到最近的合法模式**（补全缺失的先备属性）。

        用于把外部传入的估计结果修正到合法集合内。
        """
        p = np.asarray(pattern, dtype=int).ravel().astype(np.int8).copy()
        r = self.reachability()
        for k in range(self.n_attrs):
            if p[k] == 1:
                p[r[:, k]] = 1
        return p

    # ------------------------------------------------------------------ #
    # 与 Q 矩阵的交互
    # ------------------------------------------------------------------ #
    def reduced_q_matrix(self, q_matrix):
        """生成 AHM 意义下的**简化 Q 矩阵**。

        若属性 b 以 a 为先备，则"考查 b"本身就蕴含"考查 a"，
        因此 Q 矩阵中同时标了 a 和 b 的条目里，a 是冗余的，可以删去。
        简化后的 Q 矩阵更简洁，且与层级假设一致。
        """
        from . import QMatrix

        m = np.array(q_matrix.matrix, dtype=np.int8, copy=True)
        r = self.reachability()
        for a, b in self.edges:
            # 对所有采纳了 b 的题目，把其（传递）先备 a 去掉
            rows = np.where(m[:, b] == 1)[0]
            m[np.ix_(rows, np.where(r[:, b])[0])] = 0
        return QMatrix(m, attr_names=q_matrix.attr_names, item_names=q_matrix.item_names)

    def to_dict(self) -> dict:
        return {
            "structure": self.structure,
            "n_attrs": self.n_attrs,
            "edges": [[int(a), int(b)] for a, b in self.edges],
            "attr_names": list(self.attr_names) if self.attr_names else None,
            "n_legal_patterns": self.n_legal_patterns,
            "levels": self.levels.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttributeHierarchy":
        return cls(
            int(data["n_attrs"]),
            [tuple(e) for e in data.get("edges", [])],
            data.get("attr_names"),
            data.get("structure", "custom"),
        )

    def summary(self, attr_names: Optional[Sequence[str]] = None) -> str:
        names = list(attr_names or self.attr_names or [f"A{i + 1}" for i in range(self.n_attrs)])
        lines = [
            "=" * 64,
            f"属性层级结构：{self.structure}",
            "=" * 64,
            f"  属性数 K          : {self.n_attrs}",
            f"  先决关系数        : {len(self.edges)}",
            f"  合法模式数        : {self.n_legal_patterns} / {1 << self.n_attrs}"
            f"（压缩 {(1 - self.n_legal_patterns / (1 << self.n_attrs)):.1%}）",
            f"  属性层级          : {self.levels.tolist()}",
        ]
        if self.edges:
            lines.append("  先决关系（先备 -> 后继）:")
            for a, b in self.edges:
                lines.append(f"    {names[a]} -> {names[b]}")
        else:
            lines.append("  无先决关系（全部 2^K 个模式合法）")
        lines.append("=" * 64)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AttributeHierarchy(structure={self.structure!r}, K={self.n_attrs}, "
            f"n_legal={self.n_legal_patterns})"
        )


# ====================================================================== #
# 层级与 Q 矩阵的一致性检查
# ====================================================================== #
@dataclass
class HierarchyConsistencyReport:
    """Q 矩阵与属性层级的一致性检查结果。"""

    n_items: int
    n_violations: int
    violations: List[dict] = field(default_factory=list)
    attr_names: List[str] = field(default_factory=list)
    item_names: List[str] = field(default_factory=list)
    structure: str = "custom"

    @property
    def is_consistent(self) -> bool:
        return self.n_violations == 0

    def to_dict(self) -> dict:
        return {
            "structure": self.structure,
            "n_items": self.n_items,
            "n_violations": self.n_violations,
            "is_consistent": self.is_consistent,
            "violations": self.violations,
        }

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.violations) if self.violations else pd.DataFrame(
            columns=["题号", "应补先备属性"]
        )

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "Q 矩阵与属性层级的一致性检查",
            "=" * 64,
            f"  题目数            : {self.n_items}",
            f"  不一致题目数      : {self.n_violations}",
        ]
        if self.violations:
            lines.append("")
            lines.append("  需要补充先备属性的题目:")
            for v in self.violations[:15]:
                lines.append(f"    {v['题号']:<10s} 应补: {v['应补先备属性']}")
            if len(self.violations) > 15:
                lines.append(f"    ... 另有 {len(self.violations) - 15} 题")
        else:
            lines.append("  所有题目均满足层级约束。")
        lines.append("=" * 64)
        return "\n".join(lines)


def check_hierarchy_consistency(q_matrix, hierarchy: AttributeHierarchy) -> HierarchyConsistencyReport:
    """检查 Q 矩阵是否与属性层级一致。

    一致性要求：某题若考查属性 b，则必须同时考查 b 的**全部**先备属性
    （否则该题在层级假设下不可能答对，属于题目与结构矛盾）。
    """
    m = np.asarray(q_matrix.matrix, dtype=int)
    k = hierarchy.n_attrs
    if q_matrix.n_attrs != k:
        raise ValueError(
            f"Q 矩阵属性数 ({q_matrix.n_attrs}) 与层级 ({k}) 不一致"
        )
    names = list(q_matrix.attr_names or [f"A{i + 1}" for i in range(k)])
    item_names = list(q_matrix.item_names or [f"I{i + 1}" for i in range(m.shape[0])])

    violations = []
    for it in range(m.shape[0]):
        need: List[int] = []
        for b in np.where(m[it] == 1)[0]:
            for a in np.where(hierarchy.reachability()[:, b])[0]:
                if m[it, a] == 0 and a not in need:
                    need.append(int(a))
        if need:
            violations.append({
                "题号": item_names[it],
                "item_idx": it,
                "应补先备属性": "、".join(names[a] for a in sorted(need)),
                "缺失属性索引": sorted(int(a) for a in need),
            })

    return HierarchyConsistencyReport(
        n_items=m.shape[0],
        n_violations=len(violations),
        violations=violations,
        attr_names=names,
        item_names=item_names,
        structure=hierarchy.structure,
    )
