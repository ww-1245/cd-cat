"""Q 矩阵：构建、校验、诊断与解析。

Q 矩阵是认知诊断测验的基石，形状为 ``(J, K)``：
第 ``j`` 行第 ``k`` 列为 1 表示第 ``j`` 题考查第 ``k`` 个属性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils import enumerate_attribute_patterns, conjunctive_ideal, random_q_matrix
from .identifiability import IdentifiabilityReport, analyse_identifiability

__all__ = [
    "QMatrix",
    "QMatrixDiagnostics",
    "IdentifiabilityReport",
    "analyse_identifiability",
]


@dataclass
class QMatrixDiagnostics:
    """Q 矩阵诊断结果。"""

    n_items: int
    n_attrs: int
    attr_exposure: np.ndarray            # 每个属性被考查的次数
    items_per_attr: Dict[int, List[int]]  # 每个属性被哪些题目考查
    uncovered_attrs: List[int]            # 从未被考查的属性
    duplicate_rows: Dict[tuple, List[int]]  # 完全相同的 Q 向量组
    is_connected: bool                     # 属性关联图是否连通
    connected_components: List[List[int]]
    attr_correlations: np.ndarray          # 属性共现相关矩阵
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Q 矩阵诊断: {self.n_items} 题 x {self.n_attrs} 属性",
            f"  属性考查次数: {self.attr_exposure.tolist()}",
            f"  关联图连通: {'是' if self.is_connected else '否'}",
        ]
        if self.connected_components and len(self.connected_components) > 1:
            lines.append(f"  连通分量: {self.connected_components}")
        if self.uncovered_attrs:
            lines.append(f"  [警告] 未被考查的属性: {self.uncovered_attrs}")
        if self.duplicate_rows:
            dup_desc = {k: v for k, v in list(self.duplicate_rows.items())[:5]}
            lines.append(f"  [提示] 重复 Q 向量组 (样本): {dup_desc}")
        for w in self.warnings:
            lines.append(f"  [警告] {w}")
        return "\n".join(lines)


class QMatrix:
    """Q 矩阵封装类，提供校验、诊断、查询与可视化支持。

    Parameters
    ----------
    matrix : array-like, shape (J, K)
        0/1 矩阵。
    attr_names : sequence of str, optional
        属性名称，长度需为 K；默认 ``["A1", "A2", ...]``。
    item_names : sequence of str, optional
        题目名称，长度需为 J；默认 ``["I1", "I2", ...]``。

    Examples
    --------
    >>> q = QMatrix(np.array([[1, 0], [1, 1], [0, 1]]))
    >>> q.n_items, q.n_attrs
    (3, 2)
    >>> q.attr_names
    ['A1', 'A2']
    """

    def __init__(
        self,
        matrix,
        attr_names: Optional[Sequence[str]] = None,
        item_names: Optional[Sequence[str]] = None,
    ) -> None:
        arr = np.asarray(matrix)
        if arr.ndim != 2:
            raise ValueError(f"Q 矩阵必须是二维矩阵，当前维度为 {arr.ndim}")
        if not np.isin(arr, [0, 1]).all():
            raise ValueError("Q 矩阵元素必须为 0 或 1")
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("Q 矩阵不能有空维度")

        self.matrix = arr.astype(np.int8)
        j, k = self.matrix.shape

        self.attr_names = list(attr_names) if attr_names is not None else [f"A{i+1}" for i in range(k)]
        self.item_names = list(item_names) if item_names is not None else [f"I{i+1}" for i in range(j)]
        if len(self.attr_names) != k:
            raise ValueError(f"属性名数量 {len(self.attr_names)} 与矩阵列数 {k} 不符")
        if len(self.item_names) != j:
            raise ValueError(f"题目名数量 {len(self.item_names)} 与矩阵行数 {j} 不符")

        self._ideal_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # 基本属性
    # ------------------------------------------------------------------ #
    @property
    def n_items(self) -> int:
        """题目数量 J。"""
        return self.matrix.shape[0]

    @property
    def n_attrs(self) -> int:
        """属性数量 K。"""
        return self.matrix.shape[1]

    @property
    def n_patterns(self) -> int:
        """属性掌握模式总数 2^K。"""
        return 1 << self.n_attrs

    def __len__(self) -> int:
        return self.n_items

    def __repr__(self) -> str:
        return f"QMatrix(n_items={self.n_items}, n_attrs={self.n_attrs}, n_patterns={self.n_patterns})"

    # ------------------------------------------------------------------ #
    # 构造方法
    # ------------------------------------------------------------------ #
    @classmethod
    def from_random(
        cls,
        n_items: int,
        n_attrs: int,
        min_attrs_per_item: int = 1,
        max_attrs_per_item: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "QMatrix":
        """随机生成一个 Q 矩阵（保证每个属性都被考查）。"""
        return cls(
            random_q_matrix(n_items, n_attrs, min_attrs_per_item, max_attrs_per_item, seed),
            **kwargs,
        )

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "QMatrix":
        """从 CSV 文件读取 Q 矩阵（无表头或首行为表头均可）。"""
        import pandas as pd

        df = pd.read_csv(path)
        # 若首列看起来像题目名则把它作为索引
        maybe_names = None
        if df.shape[1] > 1 and not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            maybe_names = df.iloc[:, 0].astype(str).tolist()
            df = df.iloc[:, 1:]
        arr = df.to_numpy()
        if maybe_names is not None:
            kwargs.setdefault("item_names", maybe_names)
        kwargs.setdefault("attr_names", [str(c) for c in df.columns])
        return cls(arr, **kwargs)

    # ------------------------------------------------------------------ #
    # 派生量
    # ------------------------------------------------------------------ #
    def attribute_patterns(self) -> np.ndarray:
        """返回全部 2^K 个属性掌握模式，形状 ``(2^K, K)``。"""
        return enumerate_attribute_patterns(self.n_attrs)

    def ideal_response(self, mode: str = "conjunctive") -> np.ndarray:
        """返回理想反应矩阵 ``E``，形状 ``(2^K, J)``。

        ``E[i, j]`` = 属性模式 i 在第 j 题上的理想作答 (1/0)。
        DINA 使用 conjunctive，DINO 使用 disjunctive。
        """
        if mode in self._ideal_cache:
            return self._ideal_cache[mode]

        patterns = self.attribute_patterns()
        e = np.zeros((self.n_patterns, self.n_items), dtype=np.int8)
        for j in range(self.n_items):
            q_row = self.matrix[j]
            if mode == "conjunctive":
                e[:, j] = conjunctive_ideal(patterns, q_row)
            elif mode == "disjunctive":
                required = q_row.astype(bool)
                if required.any():
                    e[:, j] = patterns[:, required].astype(bool).any(axis=1).astype(np.int8)
                else:
                    e[:, j] = 1
            else:
                raise ValueError(f"未知理想反应模式: {mode}")
        self._ideal_cache[mode] = e
        return e

    def required_attrs(self, item_idx: int) -> List[int]:
        """返回某题考查的属性索引列表。"""
        return np.where(self.matrix[item_idx] == 1)[0].tolist()

    def items_measuring(self, attr_idx: int) -> List[int]:
        """返回考查某属性的题目索引列表。"""
        return np.where(self.matrix[:, attr_idx] == 1)[0].tolist()

    def attribute_exposure(self) -> np.ndarray:
        """每个属性被考查的次数，形状 ``(K,)``。"""
        return self.matrix.sum(axis=0).astype(np.int64)

    # ------------------------------------------------------------------ #
    # 校验与诊断
    # ------------------------------------------------------------------ #
    def validate(self) -> List[str]:
        """基础校验，返回问题列表（空列表表示通过）。"""
        issues: List[str] = []
        uncovered = np.where(self.matrix.sum(axis=0) == 0)[0].tolist()
        if uncovered:
            names = [self.attr_names[i] for i in uncovered]
            issues.append(f"以下属性从未被任何题目考查: {names}")
        empty_items = np.where(self.matrix.sum(axis=1) == 0)[0].tolist()
        if empty_items:
            names = [self.item_names[i] for i in empty_items]
            issues.append(f"以下题目不考查任何属性（无法提供诊断信息）: {names}")
        return issues

    def diagnose(self) -> QMatrixDiagnostics:
        """完整诊断：属性覆盖、重复题、关联图连通性、属性共现。"""
        j, k = self.n_items, self.n_attrs
        exposure = self.attribute_exposure()
        items_per_attr = {i: np.where(self.matrix[:, i] == 1)[0].tolist() for i in range(k)}
        uncovered = np.where(exposure == 0)[0].tolist()

        # 重复 Q 向量分组
        groups: Dict[tuple, List[int]] = {}
        for i in range(j):
            key = tuple(int(x) for x in self.matrix[i])
            groups.setdefault(key, []).append(i)
        duplicate_rows = {k2: v for k2, v in groups.items() if len(v) > 1}

        # 属性关联图：两个属性若共同被某题考查则相连
        adjacency = (self.matrix.T.astype(np.int32) @ self.matrix.astype(np.int32)) > 0
        np.fill_diagonal(adjacency, False)

        components = self._connected_components(adjacency)

        # 属性共现相关
        if k > 1:
            co = self.matrix.astype(np.float64)
            # 皮尔逊相关 (若某属性方差为 0 则相关定义为 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(co.T)
            corr = np.nan_to_num(corr, nan=0.0)
        else:
            corr = np.ones((1, 1))

        warnings: List[str] = []
        if len(components) > 1:
            warnings.append(
                f"属性关联图不连通，存在 {len(components)} 个独立分块，"
                "可能导致跨分块属性无法联合诊断"
            )
        if uncovered:
            warnings.append("存在未被考查的属性，请检查 Q 矩阵设计")

        return QMatrixDiagnostics(
            n_items=j,
            n_attrs=k,
            attr_exposure=exposure,
            items_per_attr=items_per_attr,
            uncovered_attrs=uncovered,
            duplicate_rows=duplicate_rows,
            is_connected=len(components) == 1,
            connected_components=components,
            attr_correlations=corr,
            warnings=warnings,
        )

    @staticmethod
    def _connected_components(adjacency: np.ndarray) -> List[List[int]]:
        """基于邻接矩阵的连通分量 (BFS)。"""
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        comps: List[List[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp: List[int] = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in np.where(adjacency[node])[0]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(int(nb))
            comps.append(sorted(comp))
        return comps

    # ------------------------------------------------------------------ #
    # 可辨识性分析
    # ------------------------------------------------------------------ #
    def analyse_identifiability(self, ideal_mode: str = "conjunctive") -> IdentifiabilityReport:
        """分析 Q 矩阵的可辨识性，并给出判准率的理论上限。

        这是**上线前最重要的检查**：如果两个属性模式的理想反应完全相同，
        那么无论用什么选题算法、答多少题，都无法把它们区分开。
        此时 PMR 的上限严格小于 1。

        Examples
        --------
        >>> q = QMatrix(np.array([[1, 0], [1, 0]]))
        >>> rep = q.analyse_identifiability()
        >>> rep.n_unidentifiable_patterns > 0
        True
        """
        return analyse_identifiability(self, ideal_mode)

    def is_identifiable(self, ideal_mode: str = "conjunctive") -> bool:
        """Q 矩阵是否可完全辨识所有属性模式。"""
        return self.analyse_identifiability(ideal_mode).n_unidentifiable_patterns == 0

    # ------------------------------------------------------------------ #
    # 转换
    # ------------------------------------------------------------------ #
    def to_dataframe(self):
        """转成 pandas DataFrame（带题目名与属性名）。"""
        import pandas as pd

        return pd.DataFrame(self.matrix, index=self.item_names, columns=self.attr_names)

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "attr_names": self.attr_names,
            "item_names": self.item_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QMatrix":
        return cls(data["matrix"], data.get("attr_names"), data.get("item_names"))

    def copy(self) -> "QMatrix":
        return QMatrix(self.matrix.copy(), list(self.attr_names), list(self.item_names))


# ---------------------------------------------------------------------- #
# Q 矩阵校验与自动修正（放在文件末尾，避免与 QMatrix 定义形成循环导入）
# ---------------------------------------------------------------------- #
from .hierarchy import (  # noqa: E402
    AttributeHierarchy,
    HierarchyConsistencyReport,
    check_hierarchy_consistency,
)
from .refinement import (  # noqa: E402
    QSuggestion,
    QValidationReport,
    validate_q_matrix,
)

__all__ += [
    "QSuggestion",
    "QValidationReport",
    "validate_q_matrix",
    "AttributeHierarchy",
    "HierarchyConsistencyReport",
    "check_hierarchy_consistency",
]
