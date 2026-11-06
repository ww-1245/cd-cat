"""蒙特卡洛仿真引擎。

``CDCATSimulator`` 负责：

1. **单被试施测**（``administer``）：真实 α -> 循环 {选题 -> 作答 -> 更新后验 -> 终止判定}。
2. **固定题量仿真**（``simulate_fixed_length``）：忽略终止规则，固定 N 题，用于研究收敛曲线。
3. **蒙特卡洛批量仿真**（``run``）：重复 R 次，每次 N 个被试，支持并行。
4. **多准则对比**（``compare_criteria``）：同一批被试在不同准则下施测，做公平对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..estimation import AttributeEstimator, EstimationHistory
from ..models import CDModel, create_model
from ..qmatrix import QMatrix
from ..selection import SelectionCriterion, create_selector
from ..termination import TerminationRule, CompositeRule, FixedLength
from ..utils import build_test_items

__all__ = ["CDCATSimulator", "SimulationResult", "ExamineeRecord"]


@dataclass
class ExamineeRecord:
    """单个被试的施测记录。"""

    examinee_id: int
    replication: int
    true_pattern: np.ndarray
    true_pattern_idx: int
    est_pattern: Optional[np.ndarray] = None
    est_pattern_idx: Optional[int] = None
    marginal_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    administered: List[int] = field(default_factory=list)
    responses: List[int] = field(default_factory=list)
    posterior_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_answered: int = 0
    stop_reason: str = ""
    max_posterior: float = 0.0

    # ---- 派生指标 ----
    @property
    def pattern_correct(self) -> bool:
        return bool(np.array_equal(self.est_pattern, self.true_pattern))

    @property
    def attribute_accuracy(self) -> float:
        if self.est_pattern is None:
            return 0.0
        return float(np.mean(self.est_pattern == self.true_pattern))

    def to_dict(self) -> dict:
        return {
            "examinee_id": self.examinee_id,
            "replication": self.replication,
            "true_pattern": np.asarray(self.true_pattern).tolist(),
            "est_pattern": None if self.est_pattern is None else np.asarray(self.est_pattern).tolist(),
            "administered": self.administered,
            "responses": self.responses,
            "n_answered": self.n_answered,
            "stop_reason": self.stop_reason,
            "pattern_correct": self.pattern_correct,
            "attribute_accuracy": self.attribute_accuracy,
            "max_posterior": self.max_posterior,
        }


@dataclass
class SimulationResult:
    """仿真结果容器，含原始记录与统计摘要。"""

    records: List[ExamineeRecord] = field(default_factory=list)
    config: Dict = field(default_factory=dict)
    n_patterns: int = 0
    n_attrs: int = 0
    n_items: int = 0
    q_matrix: Optional[QMatrix] = None
    hierarchy: Optional[object] = None

    @property
    def q(self) -> Optional[QMatrix]:
        """题库 Q 矩阵（用于可辨识性对比等分析）。"""
        return self.q_matrix

    # ------------------------------------------------------------------ #
    # 汇总统计
    # ------------------------------------------------------------------ #
    @property
    def n_examinees(self) -> int:
        return len(self.records)

    def pmr(self) -> float:
        """模式级判准率 (Pattern Match Ratio)：整条模式完全正确的比例。"""
        if not self.records:
            return 0.0
        return float(np.mean([r.pattern_correct for r in self.records]))

    def amr(self) -> float:
        """属性级判准率 (Attribute Match Ratio)：逐属性一致的平均比例。"""
        if not self.records:
            return 0.0
        return float(np.mean([r.attribute_accuracy for r in self.records]))

    def avg_items(self) -> float:
        """平均用题量。"""
        if not self.records:
            return 0.0
        return float(np.mean([r.n_answered for r in self.records]))

    def item_usage(self) -> np.ndarray:
        """题目曝光次数，长度 J。"""
        usage = np.zeros(self.n_items, dtype=np.int64)
        for r in self.records:
            for i in r.administered:
                usage[i] += 1
        return usage

    def item_exposure_rate(self) -> np.ndarray:
        """题目曝光率（相对被试数）。"""
        if self.n_examinees == 0:
            return np.zeros(self.n_items)
        return self.item_usage() / self.n_examinees

    def test_utilization(self) -> float:
        """题库利用率：至少被使用过一次的题目比例。"""
        if self.n_items == 0:
            return 0.0
        return float(np.mean(self.item_usage() > 0))

    def exposure_uniformity(self) -> float:
        """曝光均匀度：曝光率标准差越小越均匀（返回标准差）。"""
        return float(np.std(self.item_exposure_rate()))

    def attribute_accuracy_by_attr(self) -> np.ndarray:
        """各属性的判准率，长度 K。"""
        if not self.records or self.n_attrs == 0:
            return np.zeros(self.n_attrs)
        acc = np.zeros(self.n_attrs)
        for r in self.records:
            acc += (r.est_pattern == r.true_pattern).astype(float)
        return acc / len(self.records)

    def pmr_by_pattern(self) -> Dict[int, dict]:
        """按真实模式分组的判准情况。"""
        groups: Dict[int, List[ExamineeRecord]] = {}
        for r in self.records:
            groups.setdefault(r.true_pattern_idx, []).append(r)
        out = {}
        for pid, recs in groups.items():
            out[pid] = {
                "n": len(recs),
                "pmr": float(np.mean([x.pattern_correct for x in recs])),
                "amr": float(np.mean([x.attribute_accuracy for x in recs])),
                "avg_items": float(np.mean([x.n_answered for x in recs])),
            }
        return out

    def marginal_convergence_curve(self) -> np.ndarray:
        """平均属性边际概率的收敛曲线，形状 ``(max_len, K)``。

        每个被试不足最大长度的部分用最后一步填充。
        """
        curves = [r.marginal_history for r in self.records if r.marginal_history.size > 0]
        if not curves:
            return np.zeros((0, self.n_attrs))
        max_len = max(c.shape[0] for c in curves)
        acc = np.zeros((max_len, self.n_attrs))
        counts = np.zeros(max_len)
        for c in curves:
            for t in range(max_len):
                idx = min(t, c.shape[0] - 1)
                acc[t] += c[idx]
                counts[t] += 1
        return acc / counts[:, None]

    def stop_reason_distribution(self) -> Dict[str, int]:
        """终止原因分布。"""
        dist: Dict[str, int] = {}
        for r in self.records:
            dist[r.stop_reason] = dist.get(r.stop_reason, 0) + 1
        return dist

    def summary(self) -> str:
        """人类可读的摘要。"""
        lines = [
            "=" * 62,
            "CD-CAT 仿真结果摘要",
            "=" * 62,
            f"被试数 x 重复次数 : {self.config.get('n_examinees', 0)} x {self.config.get('n_replications', 1)}",
            f"题库 / 属性      : {self.n_items} 题 / {self.n_attrs} 属性 ({self.n_patterns} 模式)",
            f"响应模型         : {self.config.get('model', '-')}",
            f"选题准则         : {self.config.get('selector', '-')}",
            f"估计方法         : {self.config.get('estimator', '-')}",
            f"曝光控制         : {self.config.get('exposure_control', '无控制')}",
            "-" * 62,
            f"模式判准率 PMR   : {self.pmr():.4f}",
            f"属性判准率 AMR   : {self.amr():.4f}",
            f"平均用题量       : {self.avg_items():.2f}",
            f"题库利用率       : {self.test_utilization():.4f}",
            f"曝光率标准差     : {self.exposure_uniformity():.4f}",
            "-" * 62,
        ]
        by_attr = self.attribute_accuracy_by_attr()
        lines.append("各属性判准率: " + "  ".join(f"A{k+1}={v:.3f}" for k, v in enumerate(by_attr)))
        lines.append("-" * 62)
        dist = self.stop_reason_distribution()
        lines.append("终止原因分布:")
        for reason, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"  {cnt:5d} ({cnt/self.n_examinees:5.1%})  {reason}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def pmr_vs_ceiling(self, ideal_mode: str = "conjunctive") -> dict:
        """把实测 PMR 与 Q 矩阵可辨识性理论上限做对比。

        如果实测 PMR 已接近上限，说明**算法已做到最好**，
        继续提升必须修改 Q 矩阵而非调整算法。

        Returns
        -------
        dict
            含 ``pmr``, ``ceiling``, ``gap``, ``efficiency``, ``verdict``。
        """
        from ..qmatrix import analyse_identifiability

        rep = analyse_identifiability(self.q, ideal_mode, hierarchy=self.hierarchy) \
            if hasattr(self, "q") else None
        ceiling = rep.expected_pmr_ceiling if rep is not None else 1.0
        pmr = self.pmr()
        gap = ceiling - pmr
        efficiency = pmr / ceiling if ceiling > 0 else 0.0

        if ceiling >= 0.999:
            verdict = "Q 矩阵可完全辨识，PMR 提升空间来自算法与样本量"
        elif efficiency >= 0.95:
            verdict = "已达到结构上限，继续提升必须修改 Q 矩阵（增加可区分属性的题目）"
        elif efficiency >= 0.85:
            verdict = "接近结构上限，算法仍有小幅优化空间"
        else:
            verdict = "距结构上限较远，算法或终止规则有较大优化空间"

        return {
            "pmr": pmr,
            "ceiling": float(ceiling),
            "gap": float(gap),
            "efficiency": float(efficiency),
            "n_unidentifiable": rep.n_unidentifiable_patterns if rep is not None else 0,
            "verdict": verdict,
        }

    def to_dataframe(self):
        """转成 pandas DataFrame（每个被试一行）。"""
        import pandas as pd

        rows = [r.to_dict() for r in self.records]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "n_patterns": self.n_patterns,
            "n_attrs": self.n_attrs,
            "n_items": self.n_items,
            "summary": {
                "pmr": self.pmr(),
                "amr": self.amr(),
                "avg_items": self.avg_items(),
                "test_utilization": self.test_utilization(),
                "exposure_sd": self.exposure_uniformity(),
                "attribute_accuracy": self.attribute_accuracy_by_attr().tolist(),
            },
            "records": [r.to_dict() for r in self.records],
        }


def _run_replication(
    sim: "CDCATSimulator",
    rep: int,
    seed: int,
    n_examinees: int,
    pattern_dist: Optional[Sequence[float]],
    record_posterior: bool,
) -> List["ExamineeRecord"]:
    """跑完一次重复，返回该次全部被试记录。

    模块级函数，便于 joblib 多进程 pickle。**串行与并行走同一条代码路径**，
    因此 ``n_jobs>1`` 的结果与 ``n_jobs=1`` 逐位一致。

    每次调用先深拷贝仿真器：曝光计数、选择器与终止规则都带内部状态，
    必须保证各重复之间互不污染。
    """
    import copy

    sim = copy.deepcopy(sim)
    rng = np.random.default_rng(int(seed))
    truths = sim._sample_true_patterns(n_examinees, rng, pattern_dist)
    # 曝光计数只在"单次重复内部"累积（同一批被试共享题库）
    sim.exposure_control.reset()
    return [
        sim.administer(
            truths[eid],
            examinee_id=eid,
            replication=rep,
            rng=rng,
            record_posterior=record_posterior,
        )
        for eid in range(n_examinees)
    ]


class CDCATSimulator:
    """CD-CAT 选题与仿真引擎。

    Parameters
    ----------
    q_matrix : QMatrix
        题库的 Q 矩阵。
    model : str or CDModel
        响应模型，可以是名称（'DINA' / 'DINO' / 'G-DINA' / 'RRUM'）或已标定模型实例。
    model_params : dict, optional
        模型参数，如 ``{"slip": [...], "guess": [...]}``。
    selector : str or SelectionCriterion
        选题准则名称或实例。
    estimator : str
        属性估计方法：'MAP' / 'EAP' / 'MLE'。
    termination : TerminationRule, optional
        终止规则；默认 (后验阈值 0.8 + 最大 40 题)。
    min_items : int
        最小题量。
    max_items : int, optional
        最大题量硬上限。
    true_model : str or CDModel, optional
        仿真中被试的真实作答模型；默认为 ``model``（即参数真值已知）。
        可设置为与 ``model`` 不同的模型以模拟"模型错配"情景。

    Examples
    --------
    >>> q = QMatrix.from_random(60, 5, seed=42)
    >>> items = build_test_items(60, 5, seed=42)
    >>> sim = CDCATSimulator(q, model='DINA',
    ...                      model_params={'slip': items['slip'], 'guess': items['guess']},
    ...                      selector='PWKL')
    >>> result = sim.run(n_examinees=200, n_replications=5, seed=0)
    >>> result.pmr() > 0.5
    True
    """

    def __init__(
        self,
        q_matrix: QMatrix,
        model=("DINA",),
        model_params: Optional[dict] = None,
        selector=("PWKL",),
        estimator: str = "MAP",
        termination: Optional[TerminationRule] = None,
        min_items: int = 5,
        max_items: Optional[int] = 40,
        true_model=None,
        true_model_params: Optional[dict] = None,
        prior=None,
        exposure_control=None,
        hierarchy=None,
        verbose: bool = False,
    ) -> None:
        if not isinstance(q_matrix, QMatrix):
            q_matrix = QMatrix(q_matrix)
        self.q = q_matrix

        # ---- 属性层级（可选）：约束后验 + 只在合法模式抽样被试 ----
        self.hierarchy = hierarchy
        if hierarchy is not None and getattr(hierarchy, "n_attrs", None) != q_matrix.n_attrs:
            raise ValueError(
                f"层级结构属性数 ({hierarchy.n_attrs}) 与 Q 矩阵 ({q_matrix.n_attrs}) 不一致"
            )

        # ---- 模型（用于估计/选题） ----
        self.model_name, self.model = self._resolve_model(model, model_params, q_matrix)
        # ---- 真实模型（用于生成作答） ----
        if true_model is None:
            self.true_model = self.model
            self.true_model_name = self.model_name
        else:
            self.true_model_name, self.true_model = self._resolve_model(true_model, true_model_params, q_matrix)

        # ---- 选题准则 ----
        if isinstance(selector, str):
            self.selector_name = selector.upper()
            self.selector: SelectionCriterion = create_selector(selector, self.model)
        else:
            self.selector = selector
            self.selector_name = selector.__class__.__name__

        # ---- 估计器 ----
        self.estimator_name = estimator.upper()
        self.estimator = AttributeEstimator(
            self.model, prior=prior, method=estimator, hierarchy=hierarchy
        )

        # ---- 终止规则 ----
        self.min_items = min_items
        self.max_items = max_items
        if termination is None:
            from ..termination import PatternConfidence

            termination = CompositeRule(
                [PatternConfidence(threshold=0.8)],
                how="any",
                min_items=min_items,
                max_items=max_items,
            )
        self.termination = termination
        self.verbose = verbose

        # ---- 曝光控制 ----
        if isinstance(exposure_control, str):
            from ..selection import create_exposure_control

            self.exposure_control = create_exposure_control(exposure_control, self.q.n_items)
            self.exposure_control_name = exposure_control
        elif exposure_control is None:
            from ..selection import NoExposureControl

            self.exposure_control = NoExposureControl(self.q.n_items)
            self.exposure_control_name = "无控制"
        else:
            self.exposure_control = exposure_control
            self.exposure_control_name = exposure_control.__class__.__name__

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_model(model, params, q_matrix):
        if isinstance(model, CDModel):
            return model.__class__.__name__, model
        name = str(model)
        return name.upper(), create_model(name, q_matrix, **(params or {}))

    def _rebuild_selector(self) -> None:
        """模型参数变化后重建选题准则缓存。"""
        if isinstance(self.selector, str):
            self.selector = create_selector(self.selector_name, self.model)
        else:
            self.selector.clear_cache()

    def set_model_params(self, **params) -> "CDCATSimulator":
        """更新题目参数并刷新依赖缓存。"""
        self.model.set_params(**params)
        self._rebuild_selector()
        self.estimator.model = self.model
        self.estimator.clear_cache() if hasattr(self.estimator, "clear_cache") else None
        return self

    # ------------------------------------------------------------------ #
    # 单被试施测
    # ------------------------------------------------------------------ #
    def administer(
        self,
        true_pattern: np.ndarray,
        examinee_id: int = 0,
        replication: int = 0,
        rng: Optional[np.random.Generator] = None,
        record_posterior: bool = False,
    ) -> ExamineeRecord:
        """对单个被试执行一次完整自适应施测。

        Parameters
        ----------
        true_pattern : array-like, shape (K,)
            被试真实属性掌握模式。
        examinee_id : int
        replication : int
            重复编号（用于记录）。
        rng : numpy Generator, optional
        record_posterior : bool
            是否保存每步完整后验分布（内存占用大，默认只存边际概率）。

        Returns
        -------
        ExamineeRecord
        """
        if rng is None:
            rng = np.random.default_rng()
        true_pattern = np.asarray(true_pattern, dtype=np.int8).ravel()
        if len(true_pattern) != self.q.n_attrs:
            raise ValueError(f"真实模式长度需为 {self.q.n_attrs}")

        k = self.q.n_attrs
        true_idx = int(sum(int(a) << (k - 1 - i) for i, a in enumerate(true_pattern)))

        rec = ExamineeRecord(
            examinee_id=examinee_id,
            replication=replication,
            true_pattern=true_pattern.copy(),
            true_pattern_idx=true_idx,
        )

        administered: List[int] = []
        responses: List[int] = []
        est_history: List[int] = []
        marginal_hist: List[np.ndarray] = []
        post_hist: List[np.ndarray] = []

        # 初始估计（无作答）
        init = self.estimator.estimate(np.array([]), np.array([], dtype=int))
        marginal = init.marginal
        posterior = init.posterior
        est_idx = init.pattern_idx
        marginal_hist.append(marginal.copy())
        est_history.append(est_idx)
        if record_posterior:
            post_hist.append(posterior.copy())

        max_items = self.max_items if self.max_items is not None else self.q.n_items

        while len(administered) < min(max_items, self.q.n_items):
            state = {
                "posterior": posterior,
                "marginal": marginal,
                "estimate": self.q.attribute_patterns()[est_idx],
                "n_answered": len(administered),
                "history": est_history[1:],
                "n_items": self.q.n_items,
                "min_items": self.min_items,
            }
            if self.termination.should_stop(state):
                rec.stop_reason = self.termination.reason(state)
                break

            # ---- 选题（经曝光控制） ----
            item = self._select_item(posterior, est_idx, administered, rng)

            # ---- 作答（用真实模型生成） ----
            resp = int(self.true_model.simulate(true_pattern, [item], rng)[0])
            administered.append(item)
            responses.append(resp)
            self.exposure_control.update(item, True)

            # ---- 更新后验 ----
            res = self.estimator.estimate(
                np.array(responses, dtype=float),
                np.array(administered, dtype=int),
            )
            posterior = res.posterior
            marginal = res.marginal
            est_idx = res.pattern_idx
            marginal_hist.append(marginal.copy())
            est_history.append(est_idx)
            if record_posterior:
                post_hist.append(posterior.copy())

        else:
            rec.stop_reason = rec.stop_reason or f"达到最大题量 {max_items}"

        # 最终结果
        final = self.estimator.estimate(
            np.array(responses, dtype=float),
            np.array(administered, dtype=int),
        )
        rec.est_pattern = final.pattern
        rec.est_pattern_idx = final.pattern_idx
        rec.administered = administered
        rec.responses = responses
        rec.n_answered = len(administered)
        rec.max_posterior = final.max_posterior
        rec.marginal_history = np.vstack(marginal_hist)
        if record_posterior:
            rec.posterior_history = np.vstack(post_hist)
        self.exposure_control.end_examinee()
        return rec

    def _select_item(
        self,
        posterior: np.ndarray,
        est_idx: int,
        administered: List[int],
        rng: np.random.Generator,
    ) -> int:
        """统一的选题入口：准则评分 + 曝光控制。

        先让选题准则对所有未作答题打分，再交由曝光控制器决定最终选哪道。
        """
        cand = np.array(
            [i for i in range(self.q.n_items) if i not in set(administered)], dtype=np.int64
        )
        if len(cand) == 0:
            raise RuntimeError("题库已耗尽，无题可选")

        if hasattr(self.selector, "set_estimate"):
            self.selector.set_estimate(est_idx)
        values = self.selector.compute(posterior, cand)
        return self.exposure_control.pick(cand, values, rng)

    # ------------------------------------------------------------------ #
    # 固定题量仿真（研究收敛曲线）
    # ------------------------------------------------------------------ #
    def simulate_fixed_length(
        self,
        true_pattern: np.ndarray,
        n_items: int,
        rng: Optional[np.random.Generator] = None,
        examinee_id: int = 0,
    ) -> ExamineeRecord:
        """固定题量施测：无论后验如何，都答满 ``n_items`` 题。

        用于绘制"判准率随题量增长"的收敛曲线。
        """
        if rng is None:
            rng = np.random.default_rng()
        true_pattern = np.asarray(true_pattern, dtype=np.int8).ravel()
        k = self.q.n_attrs
        true_idx = int(sum(int(a) << (k - 1 - i) for i, a in enumerate(true_pattern)))

        rec = ExamineeRecord(examinee_id, 0, true_pattern.copy(), true_idx)
        administered: List[int] = []
        responses: List[int] = []
        marginal_hist: List[np.ndarray] = []

        init = self.estimator.estimate(np.array([]), np.array([], dtype=int))
        posterior, est_idx, marginal = init.posterior, init.pattern_idx, init.marginal
        marginal_hist.append(marginal.copy())

        for _ in range(min(n_items, self.q.n_items)):
            item = self._select_item(posterior, est_idx, administered, rng)
            resp = int(self.true_model.simulate(true_pattern, [item], rng)[0])
            administered.append(item)
            responses.append(resp)
            self.exposure_control.update(item, True)
            res = self.estimator.estimate(
                np.array(responses, dtype=float), np.array(administered, dtype=int)
            )
            posterior, est_idx, marginal = res.posterior, res.pattern_idx, res.marginal
            marginal_hist.append(marginal.copy())

        final = self.estimator.estimate(
            np.array(responses, dtype=float), np.array(administered, dtype=int)
        )
        rec.est_pattern = final.pattern
        rec.est_pattern_idx = final.pattern_idx
        rec.administered = administered
        rec.responses = responses
        rec.n_answered = len(administered)
        rec.max_posterior = final.max_posterior
        rec.marginal_history = np.vstack(marginal_hist)
        rec.stop_reason = f"固定 {rec.n_answered} 题"
        self.exposure_control.end_examinee()
        return rec

    # ------------------------------------------------------------------ #
    # 蒙特卡洛批量仿真
    # ------------------------------------------------------------------ #
    def _sample_true_patterns(
        self,
        n_examinees: int,
        rng: np.random.Generator,
        pattern_dist: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """抽样真实属性模式。默认均匀分布，可指定先验分布。

        若设置了属性层级，只在**合法模式**上抽样 —— 教学上不可能出现的
        组合不应进入仿真，否则会低估实际判准率。
        """
        all_p = self.q.attribute_patterns()
        if self.hierarchy is not None and not self.hierarchy.is_empty:
            legal = np.asarray(self.hierarchy.legal_indices(), dtype=np.int64)
        else:
            legal = np.arange(self.q.n_patterns)

        if pattern_dist is None:
            idxs = legal[rng.integers(0, len(legal), size=n_examinees)]
        else:
            dist = np.asarray(pattern_dist, dtype=np.float64)
            if dist.shape[0] == self.q.n_patterns and len(legal) != self.q.n_patterns:
                dist = dist[legal]          # 层级模式下把质量压到合法模式上
            dist = dist / dist.sum()
            idxs = rng.choice(legal, size=n_examinees, p=dist)
        return all_p[idxs]

    def run(
        self,
        n_examinees: int = 500,
        n_replications: int = 10,
        seed: Optional[int] = None,
        pattern_dist: Optional[Sequence[float]] = None,
        n_jobs: int = 1,
        progress: bool = False,
        record_posterior: bool = False,
    ) -> SimulationResult:
        """执行蒙特卡洛仿真。

        Parameters
        ----------
        n_examinees : int
            每次重复的被试数。
        n_replications : int
            重复次数（每次独立抽样被试）。
        seed : int, optional
            随机种子，保证可复现。
        pattern_dist : sequence, optional
            真实属性模式的分布（长度 2^K）；默认均匀。
        n_jobs : int
            并行进程数。``1`` 为串行；``>1`` 使用 joblib 多进程；
            ``-1`` 表示使用全部 CPU 核心。并行结果与串行**逐位一致**
            （每个重复由独立种子驱动，互不干扰）。
        progress : bool
            是否显示进度条。
        record_posterior : bool
            是否记录完整后验轨迹。

        Returns
        -------
        SimulationResult
        """
        master = np.random.default_rng(seed)
        seeds = master.integers(0, 2**31 - 1, size=n_replications)

        result = SimulationResult(
            config={
                "n_examinees": n_examinees,
                "n_replications": n_replications,
                "model": self.model_name,
                "true_model": self.true_model_name,
                "selector": self.selector_name,
                "estimator": self.estimator_name,
                "termination": repr(self.termination),
                "min_items": self.min_items,
                "max_items": self.max_items,
                "exposure_control": self.exposure_control_name,
                "seed": seed,
                "n_jobs": int(n_jobs),
            },
            n_patterns=self.q.n_patterns,
            n_attrs=self.q.n_attrs,
            n_items=self.q.n_items,
            q_matrix=self.q,
            hierarchy=self.hierarchy,
        )

        all_records: List[ExamineeRecord] = []
        reps = list(range(n_replications))

        if int(n_jobs) != 1 and n_replications > 1:
            batches = self._run_parallel(
                reps, seeds, n_examinees, pattern_dist,
                n_jobs=int(n_jobs), record_posterior=record_posterior,
                progress=progress,
            )
            for batch in batches:
                all_records.extend(batch)
        else:
            iterator = reps
            if progress:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="replications")
            for rep in iterator:
                all_records.extend(
                    _run_replication(
                        self, rep, int(seeds[rep]), n_examinees,
                        pattern_dist, record_posterior,
                    )
                )

        result.records = all_records
        return result

    # ------------------------------------------------------------------ #
    def _run_parallel(
        self,
        reps: List[int],
        seeds: np.ndarray,
        n_examinees: int,
        pattern_dist: Optional[Sequence[float]],
        n_jobs: int,
        record_posterior: bool,
        progress: bool,
    ) -> List[List[ExamineeRecord]]:
        """用 joblib 多进程跑各次重复；结果顺序与串行一致。

        未安装 joblib 时自动退回串行（不报错，仅提示一次）。
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:  # pragma: no cover
            import warnings

            warnings.warn(
                "未安装 joblib，n_jobs 已忽略、退回串行。"
                "安装：pip install joblib",
                RuntimeWarning,
                stacklevel=2,
            )
            return [
                _run_replication(
                    self, rep, int(seeds[rep]), n_examinees,
                    pattern_dist, record_posterior,
                )
                for rep in reps
            ]

        return Parallel(n_jobs=n_jobs, verbose=10 if progress else 0)(
            delayed(_run_replication)(
                self, rep, int(seeds[rep]), n_examinees,
                pattern_dist, record_posterior,
            )
            for rep in reps
        )

    # ------------------------------------------------------------------ #
    # 准则对比
    # ------------------------------------------------------------------ #
    def compare_criteria(
        self,
        criteria: Sequence[str],
        n_examinees: int = 500,
        n_replications: int = 10,
        seed: int = 0,
        fixed_length: Optional[int] = None,
        progress: bool = False,
    ) -> Dict[str, SimulationResult]:
        """在同一批"被试"上对比多个选题准则，保证公平性。

        通过固定随机种子，让每个准则面对完全相同的被试模式与作答随机流。

        Parameters
        ----------
        criteria : sequence of str
            要对比的准则名称列表，如 ``['KL', 'PWKL', 'MPWKL', 'GDI', 'SHE', 'CDI']``。
        fixed_length : int, optional
            若给定，则用固定题量模式对比（消除终止规则差异）。

        Returns
        -------
        dict
            准则名 -> SimulationResult
        """
        out: Dict[str, SimulationResult] = {}
        for name in criteria:
            if fixed_length is not None:
                out[name] = self._run_fixed_length_batch(
                    name, fixed_length, n_examinees, n_replications, seed, progress
                )
            else:
                orig = self.selector
                try:
                    self.selector = create_selector(name, self.model)
                    self.selector_name = name.upper()
                    out[name] = self.run(
                        n_examinees=n_examinees,
                        n_replications=n_replications,
                        seed=seed,
                        progress=progress,
                    )
                finally:
                    self.selector = orig
        return out

    def _run_fixed_length_batch(
        self,
        criterion: str,
        n_items: int,
        n_examinees: int,
        n_replications: int,
        seed: int,
        progress: bool = False,
    ) -> SimulationResult:
        """固定题量下的批量子仿真（用于准则对比）。"""
        master = np.random.default_rng(seed)
        seeds = master.integers(0, 2**31 - 1, size=n_replications)

        result = SimulationResult(
            config={
                "n_examinees": n_examinees,
                "n_replications": n_replications,
                "model": self.model_name,
                "selector": criterion.upper(),
                "estimator": self.estimator_name,
                "termination": f"固定 {n_items} 题",
                "min_items": n_items,
                "max_items": n_items,
                "seed": seed,
            },
            n_patterns=self.q.n_patterns,
            n_attrs=self.q.n_attrs,
            n_items=self.q.n_items,
            q_matrix=self.q,
            hierarchy=self.hierarchy,
        )
        orig_selector = self.selector
        self.selector = create_selector(criterion, self.model)

        iterator = range(n_replications)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"{criterion} replications")

        records: List[ExamineeRecord] = []
        for rep in iterator:
            rng = np.random.default_rng(int(seeds[rep]))
            truths = self._sample_true_patterns(n_examinees, rng, None)
            self.exposure_control.reset()
            for eid in range(n_examinees):
                rec = self.simulate_fixed_length(truths[eid], n_items, rng, eid)
                rec.replication = rep
                records.append(rec)
        self.selector = orig_selector
        result.records = records
        return result

    # ------------------------------------------------------------------ #
    # 收敛曲线（判准率 vs 题量）
    # ------------------------------------------------------------------ #
    def convergence_curve(
        self,
        n_examinees: int = 300,
        max_items: int = 30,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """计算 PMR / AMR 随题量增长的曲线（固定题量仿真）。

        Returns
        -------
        dict
            含 ``'pmr'``, ``'amr'``, ``'n_items'`` 三个数组。
        """
        rng = np.random.default_rng(seed)
        truths = self._sample_true_patterns(n_examinees, rng, None)

        pmr = np.zeros(max_items + 1)
        amr = np.zeros(max_items + 1)
        counts = np.zeros(max_items + 1)

        orig_selector = self.selector
        # 用累积估计而非重跑，效率更高
        for i in range(n_examinees):
            rec = self.simulate_fixed_length(truths[i], max_items, rng, i)
            # 重放每一步的边际概率，得到逐步估计
            marg = rec.marginal_history              # (max_items+1, K)
            for t in range(0, max_items + 1):
                est = (marg[t] >= 0.5).astype(np.int8)
                pmr[t] += int(np.array_equal(est, truths[i]))
                amr[t] += float(np.mean(est == truths[i]))
                counts[t] += 1
        self.selector = orig_selector
        return {
            "pmr": pmr / counts,
            "amr": amr / counts,
            "n_items": np.arange(max_items + 1),
        }
