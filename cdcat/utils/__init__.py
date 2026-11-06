"""通用工具函数：属性模式枚举、理想反应模式、数据生成等。"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "enumerate_attribute_patterns",
    "ideal_response_pattern",
    "conjunctive_ideal",
    "disjunctive_ideal",
    "random_item_params",
    "random_q_matrix",
    "build_test_items",
    "pattern_to_str",
    "str_to_pattern",
]


def enumerate_attribute_patterns(n_attrs: int, dtype=np.int8) -> np.ndarray:
    """枚举全部 2^K 个属性掌握模式。

    返回形状为 ``(2**K, K)`` 的矩阵，第 ``i`` 行是模式 ``i`` 的二进制展开。
    行索引即模式编号：行 0 为全零 (000...)，行 2^K-1 为全一 (111...)。

    Parameters
    ----------
    n_attrs : int
        属性个数 K。K > 20 时会返回超大矩阵，请谨慎使用。
    dtype : numpy dtype
        输出矩阵数据类型。

    Examples
    --------
    >>> enumerate_attribute_patterns(3)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1]], dtype=int8)
    """
    if n_attrs < 1:
        raise ValueError("n_attrs 必须 >= 1")
    n_patterns = 1 << n_attrs
    # 按位展开，保证行 i 的二进制位从高位属性到低位属性排列
    bits = ((np.arange(n_patterns)[:, None] >> np.arange(n_attrs - 1, -1, -1)) & 1)
    return bits.astype(dtype)


def ideal_response_pattern(alpha: np.ndarray, q_row: np.ndarray, mode: str = "conjunctive") -> int:
    """计算单个属性模式在某道题上的理想反应 (无失误无猜测)。

    Parameters
    ----------
    alpha : ndarray
        单个被试的属性掌握模式，长度 K。
    q_row : ndarray
        某道题的 Q 向量，长度 K。
    mode : {'conjunctive', 'disjunctive'}
        - conjunctive (DINA): 只有掌握全部所需属性才答对。
        - disjunctive (DINO): 掌握任一所需属性即可答对。

    Returns
    -------
    int
        1 表示理想答对，0 表示理想答错。
    """
    required = q_row.astype(bool)
    if not required.any():
        # 该题不考查任何属性 —— 视为恒答对 (非诊断题)
        return 1
    mastered = alpha.astype(bool)[required]
    if mode == "conjunctive":
        return int(mastered.all())
    if mode == "disjunctive":
        return int(mastered.any())
    raise ValueError(f"未知的理想反应模式: {mode}")


def conjunctive_ideal(alphas: np.ndarray, q_row: np.ndarray) -> np.ndarray:
    """向量化计算全部属性模式在一道题上的连接型理想反应。

    Parameters
    ----------
    alphas : ndarray, shape (n_patterns, K)
    q_row : ndarray, shape (K,)

    Returns
    -------
    ndarray, shape (n_patterns,)  dtype=int8
    """
    required = q_row.astype(bool)
    if not required.any():
        return np.ones(alphas.shape[0], dtype=np.int8)
    return alphas[:, required].astype(bool).all(axis=1).astype(np.int8)


def disjunctive_ideal(alphas: np.ndarray, q_row: np.ndarray) -> np.ndarray:
    """向量化计算全部属性模式在一道题上的析取型理想反应。"""
    required = q_row.astype(bool)
    if not required.any():
        return np.ones(alphas.shape[0], dtype=np.int8)
    return alphas[:, required].astype(bool).any(axis=1).astype(np.int8)


def random_q_matrix(
    n_items: int,
    n_attrs: int,
    min_attrs_per_item: int = 1,
    max_attrs_per_item: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """随机生成 Q 矩阵，并保证每个属性至少被考查一次。

    Parameters
    ----------
    n_items : int
        题目数量 J。
    n_attrs : int
        属性数量 K。
    min_attrs_per_item : int
        每道题最少考查的属性数。
    max_attrs_per_item : int, optional
        每道题最多考查的属性数，默认为 ``min(n_attrs, 3)``。
    seed : int, optional
        随机种子。

    Returns
    -------
    ndarray, shape (J, K)
    """
    rng = np.random.default_rng(seed)
    if max_attrs_per_item is None:
        max_attrs_per_item = min(n_attrs, 3)
    if not (1 <= min_attrs_per_item <= max_attrs_per_item <= n_attrs):
        raise ValueError("需要满足 1 <= min_attrs_per_item <= max_attrs_per_item <= n_attrs")

    q = np.zeros((n_items, n_attrs), dtype=np.int8)
    for j in range(n_items):
        k = rng.integers(min_attrs_per_item, max_attrs_per_item + 1)
        cols = rng.choice(n_attrs, size=int(k), replace=False)
        q[j, cols] = 1

    # 修复从未被考查的属性：随机挂到某道题上
    uncovered = np.where(q.sum(axis=0) == 0)[0]
    for attr in uncovered:
        j = int(rng.integers(0, n_items))
        q[j, attr] = 1
    return q


def random_item_params(
    q_matrix: np.ndarray,
    slip_range: tuple = (0.05, 0.25),
    guess_range: tuple = (0.05, 0.25),
    seed: Optional[int] = None,
) -> dict:
    """为给定 Q 矩阵生成随机题目参数 (DINA/DINO 的 slip & guess)。

    Returns
    -------
    dict with keys 'slip' and 'guess', each shape (J,)
    """
    rng = np.random.default_rng(seed)
    j = q_matrix.shape[0]
    slip = rng.uniform(*slip_range, size=j)
    guess = rng.uniform(*guess_range, size=j)
    return {"slip": slip, "guess": guess}


def build_test_items(
    n_items: int,
    n_attrs: int,
    min_attrs_per_item: int = 1,
    max_attrs_per_item: Optional[int] = None,
    slip_range: tuple = (0.05, 0.25),
    guess_range: tuple = (0.05, 0.25),
    seed: Optional[int] = None,
) -> dict:
    """一步生成题库：Q 矩阵 + 题目参数 + 分值。

    Returns
    -------
    dict
        ``{"q_matrix": ndarray, "slip": ndarray, "guess": ndarray, "scores": ndarray}``
    """
    q = random_q_matrix(n_items, n_attrs, min_attrs_per_item, max_attrs_per_item, seed)
    params = random_item_params(q, slip_range, guess_range, seed)
    return {
        "q_matrix": q,
        "slip": params["slip"],
        "guess": params["guess"],
        "scores": np.ones(n_items, dtype=np.float64),
    }


def pattern_to_str(pattern: np.ndarray, sep: str = "") -> str:
    """把属性模式转成 '10110' 形式的字符串。"""
    return sep.join(str(int(b)) for b in np.asarray(pattern).ravel())


def str_to_pattern(text: str) -> np.ndarray:
    """把 '10110' 形式的字符串转成属性模式数组。"""
    cleaned = [c for c in text.strip() if c in "01"]
    if not cleaned:
        raise ValueError(f"无法解析属性模式字符串: {text!r}")
    return np.array([int(c) for c in cleaned], dtype=np.int8)
