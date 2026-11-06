"""Q 矩阵可辨识性分析 (Identifiability Analysis)。

**为什么这很重要**：CD-CAT 的判准率上限由 Q 矩阵的结构决定，而不是由
选题算法决定。如果两个属性模式在所有题目上的**理想反应完全相同**，
那么无论作答多少题、用多好的准则，都无法把二者区分开 —— 这部分判准率
损失是结构性的、算法无法弥补的。

例如 Q 矩阵中存在"某个属性不被任何题目单独考查，且总与另一属性同现"时，
这两个属性就永远无法分离。

本模块实现三类诊断：

1. **模式级可辨识** (``pattern_equivalence_classes``)
   检查理想反应矩阵是否有重复行。有重复 => 这些模式两两不可区分。

2. **属性级可辨识** (``attribute_distinguishability``)
   对每个属性 k，检查是否存在一个模式对，它们仅在第 k 位上不同。
   若不存在，说明属性 k 的"净效应"从未被单独测量过。

3. **滞后可辨识 / 完备 Q 矩阵** (``is_complete_qmatrix``)
   严格意义下（无 slip/guess）完备 Q 矩阵要求：
   - 每个属性至少被一题单独考查（该题只考查这一个属性），**或**
   - 属性之间存在足够交错的共现结构（"任意两属性都能通过某题区分"）。
   本实现采用文献常用的两条判定：
   (a) 每个属性至少被 1 道题考查；
   (b) 任意两个属性的考查题目集合不相同（否则二者可互换）。

4. **潜在损失估计** (``expected_pmr_ceiling``)
   在给定 Q 矩阵下，即使算法完美，PMR 的理论上限 =
   1 - (不可辨识模式对所占比例)。这是给使用者最有价值的参考数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils import pattern_to_str

__all__ = [
    "IdentifiabilityReport",
    "analyse_identifiability",
]


@dataclass
class IdentifiabilityReport:
    """Q 矩阵可辨识性诊断报告。"""

    n_items: int
    n_attrs: int
    n_patterns: int

    # 模式级
    n_distinct_response_vectors: int
    equivalence_classes: List[List[int]] = field(default_factory=list)

    # 属性级
    indistinguishable_attrs: List[int] = field(default_factory=list)
    interchangeable_attr_pairs: List[Tuple[int, int]] = field(default_factory=list)
    singleton_attrs: List[int] = field(default_factory=list)
    uncovered_attrs: List[int] = field(default_factory=list)

    # 完备性
    is_complete: bool = False

    # 理论判准率上限
    expected_pmr_ceiling: float = 1.0

    # 层级相关信息
    hierarchy_structure: str = ""
    n_legal_patterns: Optional[int] = None
    expected_pmr_ceiling_no_hierarchy: Optional[float] = None

    warnings: List[str] = field(default_factory=list)

    @property
    def n_unidentifiable_patterns(self) -> int:
        """处于"歧义等价组"中的模式总数（每组超过 1 个才计入）。"""
        return sum(len(g) for g in self.equivalence_classes if len(g) > 1)

    def summary(self, attr_names: Optional[List[str]] = None) -> str:
        names = attr_names or [f"A{i+1}" for i in range(self.n_attrs)]
        lines = [
            "=" * 64,
            "Q 矩阵可辨识性诊断",
            "=" * 64,
            f"规模: {self.n_items} 题 x {self.n_attrs} 属性 ({self.n_patterns} 个模式)",
            f"唯一理想反应向量数 : {self.n_distinct_response_vectors} / {self.n_patterns}",
            f"可辨识 PMR 理论上限: {self.expected_pmr_ceiling:.4f}",
        ]
        if self.hierarchy_structure:
            lines.append(
                f"属性层级            : {self.hierarchy_structure}"
                f"（合法模式 {self.n_legal_patterns} / {self.n_patterns}）"
            )
            if self.expected_pmr_ceiling_no_hierarchy is not None:
                gain = self.expected_pmr_ceiling - self.expected_pmr_ceiling_no_hierarchy
                lines.append(
                    f"  └ 无层级时上限    : "
                    f"{self.expected_pmr_ceiling_no_hierarchy:.4f}"
                    f"（层级假设提升 {gain:+.4f}）"
                )
        lines.append("-" * 64)

        if self.n_unidentifiable_patterns == 0:
            lines.append("[通过] 所有属性模式在理想反应上可区分")
        else:
            lines.append(
                f"[问题] {len(self.equivalence_classes)} 组模式不可区分，"
                f"共涉及 {self.n_unidentifiable_patterns} 个模式"
            )
            for grp in self.equivalence_classes[:8]:
                pats = [pattern_to_str(np.array([int(b) for b in bin(i)[2:].zfill(self.n_attrs)]))
                        for i in grp]
                lines.append(f"       等价组: {' ≡ '.join(pats)}")
            if len(self.equivalence_classes) > 8:
                lines.append(f"       ... 另有 {len(self.equivalence_classes) - 8} 组")

        lines.append("-" * 64)
        if self.singleton_attrs:
            lines.append(
                f"[提示] 有独立题目的属性(可单独测量): "
                f"{[names[i] for i in self.singleton_attrs]}"
            )
        if self.indistinguishable_attrs:
            lines.append(
                f"[问题] 完全无法测量的属性(掌握与否不改变任何作答): "
                f"{[names[i] for i in self.indistinguishable_attrs]}"
            )
        if self.interchangeable_attr_pairs:
            pairs = [(names[a], names[b]) for a, b in self.interchangeable_attr_pairs]
            lines.append(f"[问题] 考查题集完全相同、可互换的属性对: {pairs}")
        if self.uncovered_attrs:
            lines.append(f"[问题] 未被任何题目考查的属性: {[names[i] for i in self.uncovered_attrs]}")

        lines.append(f"\n完备 Q 矩阵(严格意义): {'是' if self.is_complete else '否'}")
        for w in self.warnings:
            lines.append(f"[建议] {w}")
        lines.append("=" * 64)
        return "\n".join(lines)


def _mode_bits(idx: int, n_attrs: int) -> str:
    return bin(idx)[2:].zfill(n_attrs)


def _equivalence_classes(e: np.ndarray) -> List[List[int]]:
    """按理想反应向量分组，返回重复组（按组内最小索引排序）。"""
    seen: Dict[tuple, List[int]] = {}
    for i in range(e.shape[0]):
        seen.setdefault(tuple(int(x) for x in e[i]), []).append(i)
    groups = [sorted(v) for v in seen.values() if len(v) > 1]
    groups.sort(key=lambda g: g[0])
    return groups


def analyse_identifiability(
    q_matrix,
    ideal_mode: str = "conjunctive",
    hierarchy=None,
) -> IdentifiabilityReport:
    """对 Q 矩阵做完整可辨识性分析。

    Parameters
    ----------
    q_matrix : QMatrix
    ideal_mode : {'conjunctive', 'disjunctive'}
        与所用响应模型一致的理想反应模式；DINA 用 conjunctive，DINO 用 disjunctive。
    hierarchy : AttributeHierarchy, optional
        属性层级结构。给定时，分析只在**合法模式集合**内进行：
        需要区分的模式更少，因此 PMR 上限通常会提高
        （原先"教学上不可能出现"的模式不再计入分母）。

    Returns
    -------
    IdentifiabilityReport
    """
    from ..qmatrix import QMatrix

    q = q_matrix if isinstance(q_matrix, QMatrix) else QMatrix(q_matrix)
    j, k = q.n_items, q.n_attrs
    if hierarchy is not None and hierarchy.n_attrs != k:
        raise ValueError(
            f"层级结构属性数 ({hierarchy.n_attrs}) 与 Q 矩阵 ({k}) 不一致"
        )

    all_patterns = q.attribute_patterns()
    e_all = q.ideal_response(ideal_mode)          # (2^K, J)

    # 有层级时只在合法模式上分析
    if hierarchy is not None and not hierarchy.is_empty:
        keep = hierarchy.legal_mask()
        patterns = all_patterns[keep]
        e = e_all[keep]
        idx_map = np.where(keep)[0].tolist()
    else:
        patterns = all_patterns
        e = e_all
        idx_map = list(range(q.n_patterns))
    n_eval = len(patterns)

    # ---------------- 1. 模式级可辨识 ----------------
    seen: Dict[tuple, List[int]] = {}
    for pos in range(n_eval):
        seen.setdefault(tuple(int(x) for x in e[pos]), []).append(idx_map[pos])
    equivalence_classes = [sorted(v) for v in seen.values() if len(v) > 1]
    equivalence_classes.sort(key=lambda g: g[0])
    n_distinct = len(seen)

    # ---------------- 2. 属性级可辨识 ----------------
    # 某属性 k 可辨识 <=> 存在一对**合法**模式，它们仅在第 k 位上不同，
    # 且二者的理想反应不同（即翻转该属性会改变至少一道题的作答）。
    legal_index_set = set(idx_map)
    indistinguishable_attrs: List[int] = []
    for attr in range(k):
        found = False
        for i in idx_map:
            partner = i ^ (1 << (k - 1 - attr))     # 翻转第 attr 位
            if partner <= i or partner not in legal_index_set:
                continue
            if not np.array_equal(e_all[i], e_all[partner]):
                found = True                        # 该属性的净效应被观测到了
                break
        if not found:
            # 在层级约束下，无论该属性掌握与否，作答都一样 => 无法测量该属性
            indistinguishable_attrs.append(attr)

    # 考查题集完全相同的属性对（可互换）
    item_sets = {attr: frozenset(q.items_measuring(attr)) for attr in range(k)}
    interchangeable: List[Tuple[int, int]] = []
    for a, b in combinations(range(k), 2):
        if item_sets[a] == item_sets[b] and len(item_sets[a]) > 0:
            interchangeable.append((a, b))

    # 独立考查某属性的题目（该题的 Q 向量只有这一个 1）
    singleton_attrs = [
        attr for attr in range(k)
        if any(int(q.matrix[item].sum()) == 1 for item in q.items_measuring(attr))
    ]

    uncovered = [attr for attr in range(k) if len(item_sets[attr]) == 0]

    # ---------------- 3. 完备性 ----------------
    # 采用"模式级可辨识"作为完备性的最终判定：
    # 即使每个属性的净效应都能被观测到，只要存在理想反应完全相同的
    # 模式组（如三属性总同现），仍然无法把这些模式分开。
    is_complete = (
        len(uncovered) == 0
        and len(interchangeable) == 0
        and n_distinct == n_eval
    )

    # ---------------- 4. 理论上限 ----------------
    # 处于歧义等价组的模式：在"完美算法"下也只能猜到组内某一个。
    # 分母用**当前结构下需要区分的模式数**（有层级时为合法模式数）。
    distinguishable = n_eval - sum(len(g) - 1 for g in equivalence_classes)
    ceiling = distinguishable / n_eval

    # 同时给出不考虑层级时的上限，便于对比"层级假设带来的收益"
    no_hier_ceiling = None
    if hierarchy is not None and not hierarchy.is_empty:
        distinguishable_all = q.n_patterns - sum(
            len(g) - 1 for g in _equivalence_classes(e_all)
        )
        no_hier_ceiling = distinguishable_all / q.n_patterns

    # ---------------- 5. 建议 ----------------
    warnings: List[str] = []
    if uncovered:
        names = [f"A{i+1}" for i in uncovered]
        warnings.append(f"为 {names} 各增加至少一道考查题目。")
    if interchangeable:
        for a, b in interchangeable:
            warnings.append(
                f"属性 A{a+1} 与 A{b+1} 的考查题集完全相同，无法区分；"
                f"请为其中至少一个属性增加一道只考查它的题目。"
            )
    if indistinguishable_attrs:
        names = [f"A{i+1}" for i in indistinguishable_attrs]
        warnings.append(
            f"属性 {names} 的掌握与否不改变任何题目的作答，完全无法测量；"
            f"请增加只考查这些属性的题目。"
        )
    if equivalence_classes and not (interchangeable or indistinguishable_attrs):
        for grp in equivalence_classes[:3]:
            pats = [_mode_bits(i, k) for i in grp]
            diff_bits = [pos for pos in range(k) if len({p[pos] for p in pats}) > 1]
            if diff_bits:
                attrs = [f"A{b+1}" for b in diff_bits]
                warnings.append(
                    f"模式 {'/'.join(pats)} 的理想作答完全相同（涉及 {attrs}）；"
                    f"这些属性总是同现，请增加能把它们分开的题目。"
                )
    if not warnings:
        warnings.append("Q 矩阵结构良好，所有属性模式均可完全辨识。")

    return IdentifiabilityReport(
        n_items=j,
        n_attrs=k,
        n_patterns=q.n_patterns,
        n_distinct_response_vectors=n_distinct,
        equivalence_classes=equivalence_classes,
        indistinguishable_attrs=indistinguishable_attrs,
        interchangeable_attr_pairs=interchangeable,
        singleton_attrs=singleton_attrs,
        uncovered_attrs=uncovered,
        is_complete=is_complete,
        expected_pmr_ceiling=float(ceiling),
        hierarchy_structure=(hierarchy.structure if hierarchy is not None else ""),
        n_legal_patterns=(n_eval if hierarchy is not None else None),
        expected_pmr_ceiling_no_hierarchy=(
            float(no_hier_ceiling) if no_hier_ceiling is not None else None
        ),
        warnings=warnings,
    )
