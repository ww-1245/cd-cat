"""CD-CAT 系统的共享工具：题库管理、演示数据、状态持久化。

Streamlit 界面与 CLI 演示脚本都从这里取题库与模型，避免重复逻辑。

内置题库定义已下沉到 :mod:`cdcat.presets`（命令行入口需要独立可用），
这里只做转发，保证 Web 界面与 CLI 读到的是同一份题库。
"""

from __future__ import annotations

import json

import numpy as np

from cdcat import QMatrix
from cdcat.presets import PRESET_BANKS, build_preset_bank, list_presets

__all__ = [
    "PRESET_BANKS",
    "build_preset_bank",
    "list_presets",
    "load_qmatrix_from_upload",
    "bank_to_json",
    "bank_from_json",
]



def load_qmatrix_from_upload(file_obj, has_header: bool = True) -> QMatrix:
    """从上传的 CSV 文件读取 Q 矩阵。"""
    import pandas as pd

    df = pd.read_csv(file_obj, header=0 if has_header else None)
    item_names = None
    if df.shape[1] > 1 and not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        item_names = df.iloc[:, 0].astype(str).tolist()
        df = df.iloc[:, 1:]
    attr_names = [str(c) for c in df.columns]
    return QMatrix(df.to_numpy(), attr_names=attr_names, item_names=item_names)


def bank_to_json(bank: dict) -> str:
    """把题库序列化成 JSON 字符串。"""
    return json.dumps(
        {
            "name": bank["name"],
            "description": bank.get("description", ""),
            "attrs": bank["attrs"],
            "q_matrix": np.asarray(bank["q"].matrix).tolist(),
            "item_names": bank["q"].item_names,
            "slip": np.asarray(bank["slip"]).tolist(),
            "guess": np.asarray(bank["guess"]).tolist(),
        },
        ensure_ascii=False,
        indent=2,
    )


def bank_from_json(text: str) -> dict:
    """从 JSON 字符串还原题库。"""
    data = json.loads(text)
    q = QMatrix(data["q_matrix"], attr_names=data["attrs"], item_names=data.get("item_names"))
    return {
        "q": q,
        "slip": np.array(data["slip"], dtype=float),
        "guess": np.array(data["guess"], dtype=float),
        "name": data.get("name", "导入题库"),
        "description": data.get("description", ""),
        "attrs": data["attrs"],
    }
