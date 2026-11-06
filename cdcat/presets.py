"""内置示例题库定义。

放在库内部（而不是 ``demo/``）是为了让命令行入口 ``cdcat`` 能独立使用：
``cdcat simulate --preset "分数运算 (4 属性)"`` 无需任何外部数据文件即可跑通。

Web 界面通过 ``demo.bank_utils`` 复用这里的定义，两边行为完全一致。
"""

from __future__ import annotations

from typing import Dict, List

from .qmatrix import QMatrix
from .utils import build_test_items

__all__ = ["PRESET_BANKS", "list_presets", "build_preset_bank"]


#: 内置示例题库（贴近真实学科的属性设定）
PRESET_BANKS: Dict[str, dict] = {
    "分数运算 (4 属性)": {
        "n_items": 48,
        "attrs": ["通分", "约分", "分数加减", "分数乘除"],
        "description": "小学分数运算诊断测验，4 个属性、48 道题。",
        "n_attrs": 4,
        "max_per_item": 2,
        "seed": 101,
    },
    "初中代数 (5 属性)": {
        "n_items": 60,
        "attrs": ["一元一次方程", "不等式", "因式分解", "分式方程", "函数图像"],
        "description": "初中代数诊断测验，5 个属性、60 道题。",
        "n_attrs": 5,
        "max_per_item": 2,
        "seed": 202,
    },
    "英语语法 (5 属性)": {
        "n_items": 60,
        "attrs": ["时态", "语态", "从句", "非谓语", "虚拟语气"],
        "description": "英语语法诊断测验，5 个属性、60 道题。",
        "n_attrs": 5,
        "max_per_item": 2,
        "seed": 303,
    },
    "统计基础 (6 属性)": {
        "n_items": 72,
        "attrs": ["描述统计", "概率分布", "抽样分布", "假设检验", "回归分析", "方差分析"],
        "description": "统计学基础诊断测验，6 个属性、72 道题。",
        "n_attrs": 6,
        "max_per_item": 2,
        "seed": 404,
    },
}


def list_presets() -> List[str]:
    """返回内置题库名称列表。"""
    return list(PRESET_BANKS.keys())


def build_preset_bank(name: str) -> dict:
    """构建内置题库（Q 矩阵 + 题目参数 + 属性名）。

    Returns
    -------
    dict
        ``{"q": QMatrix, "slip": ndarray, "guess": ndarray, "name": str,
        "description": str, "attrs": list}``

    Examples
    --------
    >>> bank = build_preset_bank("分数运算 (4 属性)")
    >>> bank["q"].n_items, bank["q"].n_attrs
    (48, 4)
    """
    if name not in PRESET_BANKS:
        raise KeyError(
            f"未知内置题库 '{name}'，可选: {list(PRESET_BANKS)}"
        )
    spec = PRESET_BANKS[name]
    q = QMatrix.from_random(
        n_items=spec["n_items"],
        n_attrs=spec["n_attrs"],
        min_attrs_per_item=1,
        max_attrs_per_item=spec["max_per_item"],
        seed=spec["seed"],
        attr_names=spec["attrs"],
    )
    items = build_test_items(
        n_items=spec["n_items"],
        n_attrs=spec["n_attrs"],
        min_attrs_per_item=1,
        max_attrs_per_item=spec["max_per_item"],
        slip_range=(0.05, 0.25),
        guess_range=(0.05, 0.25),
        seed=spec["seed"],
    )
    return {
        "q": q,
        "slip": items["slip"],
        "guess": items["guess"],
        "name": name,
        "description": spec["description"],
        "attrs": spec["attrs"],
    }
