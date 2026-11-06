"""题库信息量分析。

回答"哪道题最有诊断价值"这个问题，用于题库优化与精简。

提供三类指标：

1. **题目区分度** (``item_discrimination``)
   连接型：$1 - s_j - g_j$；值越大，掌握与未掌握者越拉得开。

2. **题目信息量** (``item_information``)
   在后验分布下，该题"预期能提供多少信息"（用 Shannon 熵缩减度量）：

       I_j = H(π) − E_x[ H(π(α | X_j = x)) ]

   这与 ``SHE`` 选题准则同源，但这里用于**离线评估整个题库**，
   而非在线选题。

3. **题目必要性与冗余** (``item_redundancy``)
   两类冗余都要检测：

   - **结构冗余**：两道题的 Q 向量完全相同 —— 它们考查同样一组属性，
     提供的诊断信息在期望意义上等价，是**确定性的**冗余，应优先处理。
   - **统计冗余**：两道题在模拟作答上的相关系数超过阈值 —— 即使 Q 向量
     不同，若实际作答高度相关，说明它们在实际测量中没提供额外信息。

   注意：Q 向量相同的题目由于每题有独立的 slip/guess 噪声，
   其**实测作答相关通常只有 0.5~0.7**，所以只看相关会漏检。
   本实现以 Q 向量比对作为主判据，相关系数作为补充。

4. **题库信息曲线** (``test_information_curve``)
   逐题累加信息量，用于判断"题库是否已经饱和"——如果新增题目
   带来的信息增益趋于 0，说明题库存在冗余。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..models import CDModel

__all__ = [
    "ItemAnalysisResult",
    "analyse_items",
]


@dataclass
class ItemAnalysisResult:
    """题库信息量分析结果。"""

    item_names: List[str]
    discrimination: np.ndarray            # (J,) 区分度
    information: np.ndarray               # (J,) 信息量（均匀先验下）
    discrimination_rank: np.ndarray       # (J,) 按区分度降序的题目索引
    information_rank: np.ndarray          # (J,) 按信息量降序的题目索引
    redundant_pairs: List[tuple] = field(default_factory=list)   # (i, j, corr)
    structural_duplicates: List[List[int]] = field(default_factory=list)  # 相同 Q 向量的题目组
    weakest_items: List[int] = field(default_factory=list)       # 区分度最低
    strongest_items: List[int] = field(default_factory=list)     # 区分度最高
    test_information_curve: Optional[np.ndarray] = None          # 累加信息曲线

    def top_items(self, n: int = 10, by: str = "discrimination") -> List[int]:
        """返回信息量/区分度最高的前 n 道题。"""
        if by == "information":
            return self.information_rank[:n].tolist()
        return self.discrimination_rank[:n].tolist()

    def summary(self, n_show: int = 10) -> str:
        lines = [
            "=" * 64,
            "题库信息量分析",
            "=" * 64,
            f"题目数: {len(self.item_names)}",
            f"区分度: 均值={self.discrimination.mean():.4f}, "
            f"范围=[{self.discrimination.min():.4f}, {self.discrimination.max():.4f}]",
            f"信息量: 均值={self.information.mean():.4f}, "
            f"范围=[{self.information.min():.4f}, {self.information.max():.4f}]",
            "-" * 64,
            f"区分度最高的 {n_show} 道题:",
        ]
        for i in self.discrimination_rank[:n_show]:
            lines.append(
                f"  {self.item_names[i]:>8s}  区分度={self.discrimination[i]:.4f}  "
                f"信息量={self.information[i]:.4f}"
            )
        lines.append("-" * 64)
        weakest_n = min(5, len(self.weakest_items))
        lines.append(f"区分度最低的 {weakest_n} 道题（建议修订或淘汰）:")
        for i in self.weakest_items[:weakest_n]:
            lines.append(
                f"  {self.item_names[i]:>8s}  区分度={self.discrimination[i]:.4f}"
            )
        if self.structural_duplicates:
            lines.append("-" * 64)
            lines.append(
                f"[重要] {len(self.structural_duplicates)} 组题目 Q 向量完全相同"
                "（诊断信息等价，建议保留区分度最高的一道）:"
            )
            for grp in self.structural_duplicates[:5]:
                names = " = ".join(self.item_names[i] for i in grp)
                lines.append(f"  {names}")
            if len(self.structural_duplicates) > 5:
                lines.append(f"  ... 另有 {len(self.structural_duplicates) - 5} 组")

        stat_pairs = [(i, jj, c) for i, jj, c in self.redundant_pairs if c < 0.999]
        if stat_pairs:
            lines.append("-" * 64)
            lines.append(f"作答高度相关的题目对（r > 阈值，统计冗余）: {len(stat_pairs)} 组")
            for i, jj, c in stat_pairs[:5]:
                lines.append(f"  {self.item_names[i]} <-> {self.item_names[jj]}  r={c:.4f}")
        lines.append("=" * 64)
        return "\n".join(lines)


def analyse_items(
    model: CDModel,
    prior: Optional[np.ndarray] = None,
    n_simulate: int = 0,
    redundancy_threshold: float = 0.95,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> ItemAnalysisResult:
    """对题库做信息量分析。

    Parameters
    ----------
    model : CDModel
        已标定的响应模型。
    prior : array-like, optional
        属性模式先验；默认均匀先验。
    n_simulate : int
        若 > 0，则额外用模拟作答计算题目间冗余相关。
    redundancy_threshold : float
        判定高冗余的相关系数阈值。
    seed : int, optional
        随机种子（与 ``rng`` 二选一）。
    rng : numpy Generator, optional

    Returns
    -------
    ItemAnalysisResult
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    q = model.q
    j = q.n_items
    if prior is None:
        prior = np.full(q.n_patterns, 1.0 / q.n_patterns)
    else:
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()

    P = np.clip(model.prob_matrix(), 1e-12, 1 - 1e-12)     # (S, J)

    # ---------------- 区分度 ----------------
    if hasattr(model, "slip") and hasattr(model, "guess"):
        discrimination = 1.0 - np.asarray(model.slip) - np.asarray(model.guess)
    else:
        # 通用定义：全掌握组与全未掌握组的概率差（用理想反应分组）
        e = q.ideal_response(model.ideal_mode).astype(bool)
        disc = np.zeros(j)
        for k in range(j):
            mastered = e[:, k]
            if mastered.any() and (~mastered).any():
                disc[k] = P[mastered, k].mean() - P[~mastered, k].mean()
        discrimination = np.maximum(disc, 0.0)

    # ---------------- 信息量（熵缩减） ----------------
    def entropy(dist: np.ndarray, axis: int) -> np.ndarray:
        return -np.sum(np.where(dist > 0, dist * np.log(dist), 0.0), axis=axis)

    h_prior = float(entropy(prior, axis=0))
    post = prior[:, None]
    q_obs = post * P
    p_item = np.clip(q_obs.sum(axis=0), 1e-12, 1 - 1e-12)
    post1 = q_obs / p_item[None, :]
    post0 = (post * (1 - P)) / (1 - p_item)[None, :]
    h1 = entropy(post1, axis=0)
    h0 = entropy(post0, axis=0)
    information = h_prior - (p_item * h1 + (1 - p_item) * h0)
    information = np.maximum(information, 0.0)

    # ---------------- 冗余分析 ----------------
    redundant_pairs: List[tuple] = []

    # (a) 结构冗余：Q 向量完全相同的题目组（确定性判据，优先）
    q_groups: Dict[tuple, List[int]] = {}
    for idx in range(j):
        key = tuple(int(x) for x in q.matrix[idx])
        q_groups.setdefault(key, []).append(idx)
    structural_duplicates = [g for g in q_groups.values() if len(g) > 1]
    structural_duplicates.sort(key=lambda g: g[0])
    for grp in structural_duplicates:
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                redundant_pairs.append((grp[a], grp[b], 1.0))

    # (b) 统计冗余：模拟作答相关（补充判据，需 Q 向量不同才有意义）
    if n_simulate > 0:
        alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, n_simulate)]
        X = np.array([model.simulate(a, rng=rng) for a in alphas], dtype=float)  # (n, J)
        std = X.std(axis=0)
        valid = np.where(std > 0)[0]
        if len(valid) > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(X[:, valid].T)
            corr = np.nan_to_num(corr, nan=0.0)
            existing = {(min(i, jj), max(i, jj)) for i, jj, _ in redundant_pairs}
            for a in range(len(valid)):
                for b in range(a + 1, len(valid)):
                    ia, ib = int(valid[a]), int(valid[b])
                    if (min(ia, ib), max(ia, ib)) in existing:
                        continue
                    if corr[a, b] >= redundancy_threshold:
                        redundant_pairs.append((ia, ib, float(corr[a, b])))
            redundant_pairs.sort(key=lambda t: -t[2])

    # ---------------- 排序 ----------------
    disc_rank = np.argsort(-discrimination)
    info_rank = np.argsort(-information)
    weakest = np.argsort(discrimination)[: max(5, j // 10)].tolist()
    strongest = disc_rank[: max(5, j // 10)].tolist()

    # ---------------- 累加信息曲线 ----------------
    ordered_info = information[info_rank]
    cum = np.cumsum(ordered_info)
    total = cum[-1] if cum[-1] > 0 else 1.0
    curve = cum / total        # 归一化到 [0,1]

    return ItemAnalysisResult(
        item_names=list(q.item_names),
        discrimination=discrimination,
        information=information,
        discrimination_rank=disc_rank,
        information_rank=info_rank,
        redundant_pairs=redundant_pairs,
        structural_duplicates=structural_duplicates,
        weakest_items=weakest,
        strongest_items=strongest,
        test_information_curve=curve,
    )
