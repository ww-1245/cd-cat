"""纯 Python 数据 I/O 层：Q 矩阵 / 作答矩阵 / 题目参数的读写。

这一层**不依赖 streamlit**，供命令行入口 (``cdcat.cli``) 与 Web 界面共用，
保证"同一份 CSV 在两边读出来完全一致"。

约定
----
Q 矩阵 CSV：首行可选表头（属性名），首列可选题号；其余为 0/1。
    示例（带表头与题号）::

        题号,通分,约分,分数加减
        I1,1,0,1
        I2,0,1,1

作答矩阵 CSV：行 = 被试，列 = 题目（顺序与 Q 矩阵一致）。
    首行可选表头，首列可选被试 ID；缺失值用空串 / ``NA`` / ``NaN`` 表示。
    多级记分题直接填整数分数（0,1,2,...）。

题目参数 JSON：``{"model": "DINA", "slip": [...], "guess": [...]}``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .qmatrix import QMatrix

__all__ = [
    "read_q_matrix",
    "read_responses",
    "read_params",
    "read_bank",
    "write_q_matrix",
    "write_responses",
    "write_params",
    "write_bank",
    "write_table",
    "read_delimited",
    "MISSING_TOKENS",
]

#: 视为缺失值的文本记号（大小写不敏感）
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", ".", "-", "?"}


# ---------------------------------------------------------------------- #
# 通用读取
# ---------------------------------------------------------------------- #
def read_delimited(path: str) -> Tuple[List[List[str]], Optional[List[str]]]:
    """读取 CSV/TSV，返回 ``(数据行, 表头或 None)``。

    自动嗅探分隔符；自动跳过 UTF-8 BOM。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(fh, dialect) if any(str(c).strip() for c in r)]
    if not rows:
        raise ValueError(f"文件为空: {path}")

    header: Optional[List[str]] = None

    # 表头判定：首行**不像数据行**，而第二行像数据行。
    #
    # "像数据行"的定义：若首列可能承载行标签（题号/学号），则要求**除首列外**
    # 的每个单元格都能解析为数值或缺失标记；单列文件则要求该列本身可解析。
    # 这样 ``S01,1,0,NA`` 会被正确认作数据（首列学号 + 一个缺失值），
    # 而 ``sid,I1,I2,I3`` 不会——I1/I2/I3 既不是数值也不是缺失标记。
    def _is_value(token: object) -> bool:
        tok = str(token).strip().lower()
        if tok in MISSING_TOKENS:
            return True
        try:
            float(tok)
            return True
        except ValueError:
            return False

    def _looks_like_data(cells: Sequence[str]) -> bool:
        if not cells:
            return False
        tail = list(cells[1:]) if len(cells) > 1 else list(cells)
        return bool(tail) and all(_is_value(c) for c in tail)

    if len(rows) > 1 and not _looks_like_data(rows[0]) and _looks_like_data(rows[1]):
        header = [str(c).strip() for c in rows[0]]
        rows = rows[1:]
    return rows, header


def _to_float(token: str) -> float:
    tok = str(token).strip().lower()
    if tok in MISSING_TOKENS:
        return float("nan")
    return float(tok)


# ---------------------------------------------------------------------- #
# Q 矩阵
# ---------------------------------------------------------------------- #
def read_q_matrix(
    path: str,
    attr_names: Optional[Sequence[str]] = None,
) -> QMatrix:
    """从 CSV/TSV 读取 Q 矩阵。

    首列若为非数字（如题号 ``I1``）会被识别为题目名；首行若为非数字
    会被识别为属性名。也可用 ``attr_names`` 显式覆盖属性名。
    """
    rows, header = read_delimited(path)
    item_names: Optional[List[str]] = None

    # 判定首列是否为“题号”列
    first_col_numeric = True
    for r in rows:
        try:
            float(str(r[0]).strip())
        except (ValueError, IndexError):
            first_col_numeric = False
            break
    if not first_col_numeric:
        item_names = [str(r[0]).strip() for r in rows]
        rows = [r[1:] for r in rows]

    if not rows:
        raise ValueError("Q 矩阵没有数据行")

    width = len(rows[0])
    for r in rows:
        if len(r) != width:
            raise ValueError(f"Q 矩阵各行列数不一致：期望 {width}，实际 {len(r)}")

    raw = np.array([[_to_float(c) for c in r] for r in rows], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Q 矩阵含缺失值或非法数值，不允许")
    if not np.isin(raw, (0.0, 1.0)).all():
        raise ValueError("Q 矩阵只能包含 0 和 1")
    matrix = raw.astype(np.int8)

    names = list(attr_names) if attr_names else (
        header[1:] if (header and len(header) == width + 1) else (
            header if (header and len(header) == width) else None
        )
    )
    return QMatrix(matrix, attr_names=names, item_names=item_names)


def write_q_matrix(q: QMatrix, path: str) -> str:
    """把 Q 矩阵写成带表头与题号的 CSV。"""
    attrs = q.attr_names
    items = q.item_names or [f"I{i + 1}" for i in range(q.n_items)]
    rows = [["item", *attrs]]
    for name, row in zip(items, np.asarray(q.matrix)):
        rows.append([str(name), *[str(int(v)) for v in row]])
    return write_table(rows, path)


# ---------------------------------------------------------------------- #
# 作答矩阵
# ---------------------------------------------------------------------- #
def read_responses(
    path: str,
    n_items: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """从 CSV/TSV 读取作答矩阵。

    Returns
    -------
    responses : ndarray, shape (n_examinees, n_items), dtype float
        缺失值以 ``np.nan`` 标记（不做插补，交由下游处理）。
    examinee_ids : list of str or None
        首列若为非数字（如学号）会被识别为被试 ID。
    """
    rows, header = read_delimited(path)
    ids: Optional[List[str]] = None

    first_col_numeric = True
    for r in rows:
        try:
            float(str(r[0]).strip())
        except (ValueError, IndexError):
            first_col_numeric = False
            break
    if not first_col_numeric:
        ids = [str(r[0]).strip() for r in rows]
        rows = [r[1:] for r in rows]

    width = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != width:
            raise ValueError(f"第 {i + 1} 行作答数列数不一致：期望 {width}，实际 {len(r)}")

    if header is not None and not first_col_numeric:
        header = header[1:]
    resp = np.array([[_to_float(c) for c in r] for r in rows], dtype=np.float64)

    if n_items is not None and resp.shape[1] != n_items:
        raise ValueError(
            f"作答矩阵列数 ({resp.shape[1]}) 与 Q 矩阵题目数 ({n_items}) 不一致"
        )
    return resp, ids


def write_responses(
    responses: np.ndarray,
    path: str,
    examinee_ids: Optional[Sequence[str]] = None,
    item_names: Optional[Sequence[str]] = None,
) -> str:
    """把作答矩阵写成 CSV（缺失值留空）。"""
    resp = np.asarray(responses, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responses 必须是二维数组")
    items = list(item_names) if item_names else [f"I{i + 1}" for i in range(resp.shape[1])]
    rows: List[list] = [["examinee", *map(str, items)]]
    for i in range(resp.shape[0]):
        sid = str(examinee_ids[i]) if examinee_ids is not None else f"E{i + 1}"
        cells = [sid]
        for v in resp[i]:
            cells.append("" if not np.isfinite(v) else (
                str(int(v)) if float(v).is_integer() else f"{float(v):.6g}"
            ))
        rows.append(cells)
    return write_table(rows, path)


# ---------------------------------------------------------------------- #
# 题目参数
# ---------------------------------------------------------------------- #
def read_params(path: str) -> Tuple[str, Dict[str, Any]]:
    """读取题目参数 JSON，返回 ``(model_name, params_dict)``。

    也接受 ``cdcat`` 的题库 JSON（含 ``q_matrix`` / ``attrs``），
    此时可用 :func:`read_bank`。
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("参数文件必须是 JSON 对象")
    model = str(data.get("model", "DINA"))
    params = {k: v for k, v in data.items() if k != "model"}
    for key in ("slip", "guess", "pi_star", "beta"):
        if key in params:
            params[key] = np.asarray(params[key], dtype=np.float64).ravel()
    if "gamma" in params:
        params["gamma"] = [np.asarray(g, dtype=np.float64).ravel() for g in params["gamma"]]
    if "r_star" in params:
        params["r_star"] = [np.asarray(g, dtype=np.float64).ravel() for g in params["r_star"]]
    return model, params


def write_params(model_name: str, model, path: str) -> str:
    """把模型参数写成 JSON（接受任意 ``CDModel`` 实例）。"""
    payload: Dict[str, Any] = {"model": model_name}
    for key, value in model.get_params().items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
            payload[key] = [v.tolist() for v in value]
        else:
            payload[key] = value
    return write_json(payload, path)


# ---------------------------------------------------------------------- #
# 完整题库（Q 矩阵 + 参数 + 属性名，对应 Web 界面导出的 JSON）
# ---------------------------------------------------------------------- #
def read_bank(path: str) -> Dict[str, Any]:
    """读取完整题库 JSON（Web 界面「导出题库」产出的格式）。

    Returns
    -------
    dict
        ``{"q": QMatrix, "slip": ndarray, "guess": ndarray,
        "name": str, "description": str, "attrs": list}``
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if "q_matrix" not in data:
        raise ValueError(
            f"{path} 不是题库文件（缺少 q_matrix 字段）。"
            "若这是题目参数文件，请改用 read_params()。"
        )
    attrs = data.get("attrs")

    # 兼容两种存法：嵌套的 q_matrix，或扁平的 matrix
    q = QMatrix(
        data["q_matrix"],
        attr_names=attrs,
        item_names=data.get("item_names"),
    )
    n_items = q.n_items

    def _pad(key: str) -> np.ndarray:
        vals = np.asarray(data.get(key, []), dtype=np.float64).ravel()
        if vals.size == 0:
            return np.full(n_items, 0.1)
        return vals

    return {
        "q": q,
        "slip": _pad("slip"),
        "guess": _pad("guess"),
        "name": data.get("name", "导入题库"),
        "description": data.get("description", ""),
        "attrs": attrs if attrs else q.attr_names,
    }


def write_bank(bank: Dict[str, Any], path: str) -> str:
    """写出完整题库 JSON（与 Web 界面导出格式兼容）。"""
    payload = {
        "name": bank.get("name", "题库"),
        "description": bank.get("description", ""),
        "attrs": list(bank.get("attrs") or bank["q"].attr_names),
        "q_matrix": np.asarray(bank["q"].matrix).tolist(),
        "item_names": bank["q"].item_names,
        "slip": np.asarray(bank["slip"], dtype=float).tolist(),
        "guess": np.asarray(bank["guess"], dtype=float).tolist(),
    }
    return write_json(payload, path)


# ---------------------------------------------------------------------- #
# 通用写出
# ---------------------------------------------------------------------- #
def write_json(obj: Any, path: str) -> str:
    """写出 JSON（自动建目录、保留中文）。"""
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=_json_default)
    return path


def write_table(rows: Sequence[Sequence[Any]], path: str) -> str:
    """写出 CSV 表格（``rows[0]`` 视为表头，自动建目录）。"""
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(row)
    return path


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"无法序列化类型: {type(obj)!r}")
