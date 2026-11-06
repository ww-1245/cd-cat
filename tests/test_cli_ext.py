"""新增 CLI 子命令的端到端测试。

覆盖 ``ppmc / hier-test / assemble / online-calib / dif / report-html``
六条命令：既检查退出码，也检查落盘产物，避免"命令跑通但没写出东西"。

数据统一用一个 4 属性 × 20 题的小题库，保证整套跑完在十几秒以内。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdcat import DINA, QMatrix  # noqa: E402
from cdcat.cli import main  # noqa: E402
from cdcat.io import write_q_matrix, write_responses  # noqa: E402


# ---------------------------------------------------------------------- #
def _fixture(tmp_path, n_items: int = 20, n: int = 400, seed: int = 5):
    """造一套 DINA 数据并落盘，返回 (q, 路径字典)。"""
    rng = np.random.default_rng(seed)
    m = np.zeros((n_items, 4), dtype=np.int8)
    for j in range(n_items):
        m[j, rng.choice(4, size=1 + (j % 2), replace=False)] = 1
    q = QMatrix(m, attr_names=[f"A{i + 1}" for i in range(4)])
    model = DINA(q, slip=np.full(n_items, 0.12), guess=np.full(n_items, 0.16))
    alphas = rng.integers(0, 2, size=(n, 4))
    resp = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)

    qp = str(tmp_path / "q.csv")
    rp = str(tmp_path / "resp.csv")
    gp = str(tmp_path / "groups.csv")
    pp = str(tmp_path / "params.json")
    write_q_matrix(q, qp)
    write_responses(resp, rp, item_names=q.item_names,
                    examinee_ids=[f"E{i + 1}" for i in range(n)])
    with open(gp, "w", encoding="utf-8-sig") as fh:
        fh.write("id,group\n")
        for i in range(n):
            fh.write(f"E{i + 1},{'A' if i % 2 == 0 else 'B'}\n")
    with open(pp, "w", encoding="utf-8") as fh:
        json.dump({"model": "DINA", "slip": model.slip.tolist(),
                   "guess": model.guess.tolist()}, fh)
    return q, {"q": qp, "resp": rp, "groups": gp, "params": pp}


# ====================================================================== #
# ppmc
# ====================================================================== #
class TestCLIPPMC:
    def test_ppmc_writes_table(self, tmp_path, capsys):
        _, p = _fixture(tmp_path)
        out = str(tmp_path / "pp")
        rc = main([
            "ppmc", "-q", p["q"], "-r", p["resp"], "-o", out,
            "--replicates", "40", "--max-items-pair", "20",
        ])
        assert rc == 0
        text = capsys.readouterr().out
        assert "后验预测检验" in text
        for stat in ("pair_srmsr", "q3_max", "joint_max", "score_sd"):
            assert stat in text

        path = os.path.join(out, "ppmc.csv")
        assert os.path.exists(path)
        with open(path, encoding="utf-8-sig") as fh:
            head = fh.readline()
        assert "统计量" in head or "stat" in head.lower()

    def test_ppmc_no_refit_flag(self, tmp_path, capsys):
        """--no-refit 应跳过复制侧重估并正常出结果。"""
        _, p = _fixture(tmp_path, n=250)
        rc = main([
            "ppmc", "-q", p["q"], "-r", p["resp"],
            "-o", str(tmp_path / "pp2"),
            "--replicates", "20", "--no-refit", "--max-items-pair", "20",
        ])
        assert rc == 0
        assert "复制侧重估" in capsys.readouterr().out


# ====================================================================== #
# hier-test
# ====================================================================== #
class TestCLIHierarchy:
    def test_hierarchy_comparison(self, tmp_path, capsys):
        _, p = _fixture(tmp_path)
        out = str(tmp_path / "hier")
        rc = main([
            "hier-test", "-q", p["q"], "-r", p["resp"], "-o", out,
            "--structures", "none", "linear", "--max-iter", "200",
        ])
        assert rc == 0
        text = capsys.readouterr().out
        assert "属性层级" in text and "BIC" in text

        path = os.path.join(out, "hierarchy_comparison.csv")
        assert os.path.exists(path)
        with open(os.path.join(out, "hierarchy_comparison.json") if False else path,
                  encoding="utf-8-sig") as fh:
            lines = fh.read().strip().splitlines()
        assert len(lines) == 3           # 表头 + 两个候选

    def test_hierarchy_json(self, tmp_path):
        _, p = _fixture(tmp_path, n=250)
        out = str(tmp_path / "hier2")
        rc = main([
            "hier-test", "-q", p["q"], "-r", p["resp"], "-o", out,
            "--structures", "none", "linear", "--max-iter", "150", "--json",
        ])
        assert rc == 0
        with open(os.path.join(out, "hierarchy_comparison.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert "candidates" in data or "best" in data


# ====================================================================== #
# assemble
# ====================================================================== #
class TestCLIAssemble:
    def test_assemble_basic(self, tmp_path, capsys):
        q, p = _fixture(tmp_path)
        out = str(tmp_path / "asm")
        rc = main([
            "assemble", "-q", p["q"], "--params", p["params"],
            "-k", "8", "-o", out, "--export-bank",
        ])
        assert rc == 0
        text = capsys.readouterr().out
        assert "期望模式判准率" in text and "属性覆盖" in text

        assert os.path.exists(os.path.join(out, "assembly.csv"))
        with open(os.path.join(out, "assembly.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["selected"] and len(data["selected"]) == 8
        assert data["n_items"] == 8

        with open(os.path.join(out, "assembled_items.txt"), encoding="utf-8") as fh:
            names = [ln for ln in fh.read().splitlines() if ln.strip()]
        assert len(names) == 8

    def test_assemble_must_include_and_bounds(self, tmp_path, capsys):
        q, p = _fixture(tmp_path)
        out = str(tmp_path / "asm2")
        rc = main([
            "assemble", "-q", p["q"], "--params", p["params"],
            "-k", "8", "--must", "1", "2", "--max-per-attr", "3",
            "-o", out, "--no-search",
        ])
        assert rc == 0
        with open(os.path.join(out, "assembly.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        # 必含题必须在结果里（题名 I1 / I2）
        assert {"I1", "I2"} <= set(data["selected_names"])
        assert max(data["attr_coverage"].values()) <= 3

    def test_assemble_infeasible_constraints_exit(self, tmp_path):
        q, p = _fixture(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main([
                "assemble", "-q", p["q"], "--params", p["params"],
                "-k", "20", "--max-per-attr", "2", "-o", str(tmp_path / "bad"),
            ])
        assert "约束" in str(exc.value) or "不可行" in str(exc.value)

    def test_assemble_bad_must_index(self, tmp_path):
        _, p = _fixture(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main([
                "assemble", "-q", p["q"], "--params", p["params"],
                "--must", "999", "-o", str(tmp_path / "bad2"),
            ])
        assert "超出范围" in str(exc.value)


# ====================================================================== #
# online-calib
# ====================================================================== #
class TestCLIOnlineCalibration:
    def test_online_calib(self, tmp_path, capsys):
        _, p = _fixture(tmp_path)
        out = str(tmp_path / "oc")
        rc = main([
            "online-calib", "-q", p["q"], "-r", p["resp"],
            "--new-items", "1", "2", "3", "-o", out,
        ])
        assert rc == 0
        text = capsys.readouterr().out
        assert "新题在线标定" in text and "掌握组人数" in text
        assert os.path.exists(os.path.join(out, "online_calibration.csv"))

    def test_online_calib_bad_index(self, tmp_path):
        _, p = _fixture(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main([
                "online-calib", "-q", p["q"], "-r", p["resp"],
                "--new-items", "0", "-o", str(tmp_path / "bad"),
            ])
        assert "超出范围" in str(exc.value)


# ====================================================================== #
# dif
# ====================================================================== #
class TestCLIDIF:
    def test_dif_requires_groups(self, tmp_path):
        _, p = _fixture(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main(["dif", "-q", p["q"], "-r", p["resp"], "-o", str(tmp_path / "d0")])
        assert "groups" in str(exc.value)

    def test_dif_runs(self, tmp_path, capsys):
        _, p = _fixture(tmp_path)
        out = str(tmp_path / "dif")
        rc = main([
            "dif", "-q", p["q"], "-r", p["resp"], "--groups", p["groups"],
            "-o", out,
        ])
        assert rc == 0
        text = capsys.readouterr().out
        assert "DIF" in text
        assert os.path.exists(os.path.join(out, "dif.csv"))


# ====================================================================== #
# report-html
# ====================================================================== #
class TestCLIReportHTML:
    def test_self_contained_report(self, tmp_path, capsys):
        _, p = _fixture(tmp_path, n=300)
        out = str(tmp_path / "rep")
        rc = main([
            "report-html", "-q", p["q"], "-r", p["resp"], "-o", out,
            "--models", "DINA", "--no-figures", "--title", "测试报告",
        ])
        assert rc == 0
        text = capsys.readouterr().out
        assert "最优模型" in text

        path = os.path.join(out, "report.html")
        assert os.path.exists(path)
        html = Path(path).read_text(encoding="utf-8")
        assert "测试报告" in html
        # 自包含：不允许外链样式表或脚本
        assert html.count("http://") == 0
        assert html.count("https://") == 0
        assert "<link" not in html
        assert "<script" not in html
