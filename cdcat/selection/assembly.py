"""题库装配优化：给定题量预算，从题库里挑出一份"最不容易把考生分错类"的卷子。

为什么需要它
------------
自适应测验是逐题挑的，但很多场景必须**先组一份固定卷**：纸笔测验、大规模
统一考试、或者 CD-CAT 的初始阶段（前几题没有作答信息，只能按预设顺序出题）。
题库往往有几百道题，凭经验挑题很难保证这份卷子对各个知识状态都有区分力。

本模块把"挑卷子"变成一个带约束的组合优化问题。

目标函数：最小成对 KL（maximin）
--------------------------------
对两个真实属性模式 :math:`\\alpha_s, \\alpha_{s'}`，一份卷子能把它们分开的
难易由两者作答分布的 KL 散度决定：

.. math::

    D(s, s') = \\sum_{j \\in \\text{卷}} \\left[
        p_{sj} \\ln \\frac{p_{sj}}{p_{s'j}}
        + (1-p_{sj}) \\ln \\frac{1-p_{sj}}{1-p_{s'j}} \\right]

这里 :math:`p_{sj} = P(X_j = 1 \\mid \\alpha_s)`。它可加、依赖题目参数、并且
直接决定"把 s 判成 s'"的渐近错误率上界（Stein 引理）。取**所有模式对里最小的
那个**作为目标，等于专门保护"最难分开的那一对"——这正是判准率 PMR 的短板所在。

KL 不对称，取 Jeffreys 对称化 :math:`(D(s,s') + D(s',s))/2`。

顺带给出理论判准率
------------------
``expected_pmr``：把每个真实模式按模型抽一批作答，走一遍 MAP 分类，统计判对
的比例。这是"真值已知、只受测量误差影响"时的判准率，也就是这份卷子的**理论上限**
（真实 PMR 还要再打折，因为属性本身是估出来的）。

这里不能改用"理想作答"（每题取模）：只要题目区分得开，理想作答必然全判对，
结果恒为 1.0，衡量不了任何东西。``pattern_accuracy`` 会指出这份卷子最容易混淆
的几个模式，方便针对性补题。

算法
----
1. **必含题**先入选；
2. **覆盖优先**：若某属性覆盖题数不足 ``min_per_attr``，每步优先补覆盖；
3. **贪心加题**：在剩余名额与覆盖上限内，每次加入使目标函数最大的题；
4. **成对交换局部搜索**：反复尝试"换出一道、换入一道"，只要有改进就接受，
   直到一轮无改进或达到轮数上限。

贪心 + 交换是这类子模型目标的标准做法：单步最优不保证全局最优，但交换能把
贪心解的 1-1/e 近似比再往上推，实测能追平穷举解。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..models import CDModel, DINA
from ..qmatrix import QMatrix

__all__ = ["AssemblyResult", "assemble_test_bank"]

_EPS = 1e-10


# ---------------------------------------------------------------------- #
# 结果
# ---------------------------------------------------------------------- #
@dataclass
class AssemblyResult:
    """题库装配结果。"""

    selected: np.ndarray                    # 入选题目索引（升序）
    objective: str                          # 目标函数名
    score: float                            # 目标函数值（最小成对 KL）
    mean_pair_kl: float                     # 平均成对 KL（辅助指标）
    expected_pmr: float                     # 模型隐含的期望模式判准率（理论上限）
    expected_amr: np.ndarray                # 各属性的期望判准率
    pattern_accuracy: np.ndarray            # 各真实模式的判准率
    attr_coverage: np.ndarray               # 各属性被多少道入选题测量
    attr_share: np.ndarray                  # 覆盖占比（归一化）
    history: List[float] = field(default_factory=list)
    n_swaps: int = 0
    n_items_bank: int = 0
    n_attrs: int = 0
    must_include: List[int] = field(default_factory=list)
    min_per_attr: int = 1
    max_per_attr: Optional[int] = None
    item_names: List[str] = field(default_factory=list)
    attr_names: List[str] = field(default_factory=list)

    # ---- 便捷属性 ----
    @property
    def n_items(self) -> int:
        return int(self.selected.size)

    @property
    def selected_names(self) -> List[str]:
        if not self.item_names:
            return [f"I{i + 1}" for i in self.selected]
        return [self.item_names[i] for i in self.selected]

    def coverage_ok(self) -> bool:
        return bool(np.all(self.attr_coverage >= self.min_per_attr))

    # ---- 输出 ----
    def to_dataframe(self):
        import pandas as pd

        names = self.attr_names or [f"A{k + 1}" for k in range(self.n_attrs)]
        rows = []
        for i in self.selected:
            rows.append({
                "item_idx": int(i),
                "item": (self.item_names[i] if self.item_names else f"I{i + 1}"),
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        names = self.attr_names or [f"A{k + 1}" for k in range(self.n_attrs)]
        return {
            "n_items": self.n_items,
            "n_items_bank": self.n_items_bank,
            "objective": self.objective,
            "score": round(float(self.score), 6),
            "mean_pair_kl": round(float(self.mean_pair_kl), 6),
            "expected_pmr": round(float(self.expected_pmr), 6),
            "expected_amr": {
                n: round(float(v), 4) for n, v in zip(names, self.expected_amr)
            },
            "weakest_patterns": [int(i) for i in np.argsort(self.pattern_accuracy)[:5]],
            "attr_coverage": {
                n: int(v) for n, v in zip(names, self.attr_coverage)
            },
            "selected": [int(i) for i in self.selected],
            "selected_names": self.selected_names,
            "n_swaps": int(self.n_swaps),
            "must_include": [int(i) for i in self.must_include],
        }

    def summary(self) -> str:
        names = self.attr_names or [f"A{k + 1}" for k in range(self.n_attrs)]
        lines = [
            "=" * 68,
            "题库装配结果",
            "=" * 68,
            f"  题库题量 / 入选题量 : {self.n_items_bank} / {self.n_items}",
            f"  目标函数            : {self.objective}",
            f"  最小成对 KL         : {self.score:.4f}",
            f"  平均成对 KL         : {self.mean_pair_kl:.4f}",
            f"  理论判准率上限 PMR  : {self.expected_pmr:.4f}",
            f"  局部搜索交换次数    : {self.n_swaps}",
            "",
            "  各属性覆盖:",
        ]
        for n, c, s in zip(names, self.attr_coverage, self.attr_share):
            bar = "█" * int(round(s * 20))
            lines.append(f"    {n:<8}{c:>4} 题  {bar:<20}{s:>6.1%}")
        if not self.coverage_ok():
            lack = [names[k] for k in np.where(self.attr_coverage < self.min_per_attr)[0]]
            lines.append(f"  ! 覆盖不足（要求 ≥{self.min_per_attr} 题）: {', '.join(lack)}")
        lines += ["", "  入选题目: " + ", ".join(self.selected_names)]
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 目标函数内部工具
# ---------------------------------------------------------------------- #
def _item_kl_matrix(p: np.ndarray) -> np.ndarray:
    """单题的对称 KL 贡献矩阵，形状 ``(S, S)``。

    ``p[s] = P(X_j=1 | α_s)``；结果对 ``(s, s')`` 与 ``(s', s)`` 相同。
    """
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    q = 1.0 - p
    # D(s→s')
    a = p[:, None] * np.log(p[:, None] / p[None, :]) + \
        q[:, None] * np.log(q[:, None] / q[None, :])
    return 0.5 * (a + a.T)


def _min_pair(acc: np.ndarray) -> float:
    iu = np.triu_indices(acc.shape[0], k=1)
    if iu[0].size == 0:
        return float("inf")
    return float(acc[iu[0], iu[1]].min())


def _mean_pair(acc: np.ndarray) -> float:
    iu = np.triu_indices(acc.shape[0], k=1)
    if iu[0].size == 0:
        return float("nan")
    return float(acc[iu[0], iu[1]].mean())


def _expected_accuracy(prob: np.ndarray, selected: np.ndarray,
                       prior: np.ndarray, n_sim: int = 40,
                       seed: int = 11) -> tuple:
    """模型隐含的期望判准率（理论上限）。

    对每个真实模式按 ``P(X|α)`` 抽 ``n_sim`` 份作答，走一遍 MAP 分类，统计判对
    比例。这是"真值已知、只受测量误差影响"时的判准率，也就是该卷子的理论上限：
    真实 PMR 还要再打一次折（属性是估出来的，不是给的）。

    用理想作答（每题取模）来算没有意义——只要题目够区分，理想作答必然判对，
    得到的永远是 1.0，衡量不了任何东西。

    返回 ``(PMR, 各属性 AMR, 各模式判准率)``。
    """
    s_total = int(prob.shape[0])
    sub = np.asarray(prob[:, selected], dtype=np.float64)
    j_sel = int(sub.shape[1])
    if j_sel == 0:
        return 0.0, np.zeros(1), np.zeros(s_total)

    rng = np.random.default_rng(seed)
    draws = (rng.random((s_total, n_sim, j_sel)) < sub[:, None, :]).astype(np.float64)
    draws = draws.reshape(s_total * n_sim, j_sel)

    log_p = np.log(np.clip(sub, _EPS, 1.0))
    log_q = np.log(np.clip(1.0 - sub, _EPS, 1.0))
    ll = draws @ log_p.T + (1.0 - draws) @ log_q.T
    ll = ll + np.log(np.clip(prior, 1e-300, None))[None, :]
    pred = ll.argmax(axis=1).reshape(s_total, n_sim)
    acc_by_pattern = (pred == np.arange(s_total)[:, None]).mean(axis=1)
    pmr = float(acc_by_pattern.mean())

    k = int(round(np.log2(s_total))) if s_total > 0 and (1 << int(round(np.log2(s_total)))) == s_total else 0
    if k <= 0:
        return pmr, np.array([pmr]), acc_by_pattern
    bits = (np.arange(s_total)[:, None] >> np.arange(k - 1, -1, -1)) & 1
    hit = (bits[pred] == bits[:, None, :])
    amr = hit.mean(axis=(0, 1))
    return pmr, amr, acc_by_pattern


# ---------------------------------------------------------------------- #
# 主函数
# ---------------------------------------------------------------------- #
def assemble_test_bank(
    q_matrix,
    model: Optional[CDModel] = None,
    n_items: Optional[int] = None,
    *,
    must_include: Optional[Sequence[int]] = None,
    min_per_attr: int = 1,
    max_per_attr: Optional[int] = None,
    prior: Optional[np.ndarray] = None,
    local_search: bool = True,
    max_rounds: int = 200,
    item_names: Optional[Sequence[str]] = None,
    attr_names: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> AssemblyResult:
    """从题库中挑选一份固定长度的测验。

    Parameters
    ----------
    q_matrix : QMatrix or array-like
    model : CDModel, optional
        提供题目参数（slip/guess 等）。不给则按 ``DINA(slip=0.10, guess=0.20)``
        的"中等质量题库"假定——用于在还没有标定结果时做题库设计。
    n_items : int, optional
        目标题量。默认取 ``min(题库题量, max(2K, K*min_per_attr))``。
    must_include : sequence of int, optional
        必含题目索引（锚题、真题等）。
    min_per_attr : int
        每个属性至少被多少道入选题测量，默认 1。
    max_per_attr : int, optional
        每个属性最多被多少道入选题测量，用于避免卷子过度偏向某几个属性。
    prior : ndarray, optional
        属性模式先验（长度 ``2^K``），默认均匀。
    local_search : bool
        是否在贪心之后做交换式局部搜索，默认开启。
    max_rounds : int
        局部搜索轮数上限。
    item_names, attr_names : sequence of str, optional
        仅用于展示。
    verbose : bool

    Returns
    -------
    AssemblyResult

    Examples
    --------
    >>> res = assemble_test_bank(q, model, n_items=12)   # doctest: +SKIP
    >>> res.selected                                     # doctest: +SKIP
    array([ 0,  3,  7, 11, ...])
    >>> print(res.summary())                             # doctest: +SKIP
    """
    q = q_matrix if isinstance(q_matrix, QMatrix) else QMatrix(q_matrix)
    if model is None:
        model = DINA(q, slip=np.full(q.n_items, 0.10), guess=np.full(q.n_items, 0.20))
    if model.n_items != q.n_items:
        raise ValueError(
            f"模型的题目数 ({model.n_items}) 与 Q 矩阵 ({q.n_items}) 不一致"
        )

    bank = q.n_items
    k = q.n_attrs
    s_total = q.n_patterns
    if s_total < 2:
        raise ValueError("属性数至少为 2 才能比较模式之间的可分性")

    min_per_attr = int(min_per_attr)
    if min_per_attr < 0:
        raise ValueError("min_per_attr 不能为负")

    if n_items is None:
        n_items = min(bank, max(2 * k, k * max(min_per_attr, 1)))
    n_items = int(n_items)
    if not 1 <= n_items <= bank:
        raise ValueError(f"n_items 必须在 [1, {bank}] 之间，收到 {n_items}")

    must = sorted({int(i) for i in (must_include or [])})
    if any(i < 0 or i >= bank for i in must):
        raise ValueError(f"must_include 含越界索引，题库索引范围 [0, {bank})")
    if len(must) > n_items:
        raise ValueError(f"必含题 ({len(must)}) 多于目标题量 ({n_items})")

    Qm = np.asarray(q.matrix, dtype=np.int8)
    attr_of = [np.where(Qm[j] == 1)[0] for j in range(bank)]

    # 可行性：每个属性至少要有 min_per_attr 道题可用
    for kk in range(k):
        avail = int((Qm[:, kk] == 1).sum())
        if avail < min_per_attr:
            raise ValueError(
                f"属性 {kk} 在题库中只有 {avail} 道题，无法满足 min_per_attr={min_per_attr}"
            )
    if max_per_attr is not None:
        max_per_attr = int(max_per_attr)
        if max_per_attr < min_per_attr:
            raise ValueError("max_per_attr 不能小于 min_per_attr")
        # 每道题至少覆盖 1 个属性 ⇒ 总覆盖数 ≥ n_items，这是必要条件
        if n_items > max_per_attr * k:
            raise ValueError(
                f"约束不可行：{n_items} 道题、每个属性至多 {max_per_attr} 题，"
                f"最多只能容纳 {max_per_attr * k} 个属性位，而每题至少占 1 个。"
                "请放宽 max_per_attr 或减少目标题量。"
            )

    prob = np.asarray(model._get_prob(), dtype=np.float64)      # (S, J)
    if prior is None:
        prior_vec = np.full(s_total, 1.0 / s_total)
    else:
        prior_vec = np.clip(np.asarray(prior, dtype=np.float64).ravel(), 0.0, None)
        if prior_vec.size != s_total:
            raise ValueError(f"prior 长度应为 {s_total}")
        prior_vec = prior_vec / prior_vec.sum()

    # 单题 KL 贡献：按需计算，避免 (J, S, S) 常驻内存
    def _kl_col(j: int) -> np.ndarray:
        return _item_kl_matrix(prob[:, j])

    def _kl_batch(items: Sequence[int]) -> np.ndarray:
        """多道题的 KL 矩阵堆叠，形状 ``(len(items), S, S)``。"""
        if len(items) == 0:
            return np.zeros((0, s_total, s_total))
        return np.stack([_kl_col(j) for j in items])

    def _best_add(acc: np.ndarray, cand: np.ndarray) -> int:
        """在候选里挑一道使目标函数最大的题。

        max-min 在题量少时会大面积并列（很多模式对还没分开），此时用平均 KL
        破平：平均 KL 随题量单调上升，能给出有意义的顺序。
        """
        if cand.size == 0:
            return -1
        mats = acc[None, :, :] + _kl_batch(cand)          # (C, S, S)
        iu = np.triu_indices(s_total, k=1)
        pair_vals = mats[:, iu[0], iu[1]]
        mins = pair_vals.min(axis=1)
        means = pair_vals.mean(axis=1)
        order = np.lexsort((-means, -mins))
        return int(cand[order[0]])

    # ---- 1. 必含题 ----
    selected: List[int] = list(must)
    remaining = [j for j in range(bank) if j not in set(must)]
    cover = np.zeros(k, dtype=int)
    for j in selected:
        cover[attr_of[j]] += 1

    acc = np.zeros((s_total, s_total))
    for j in selected:
        acc += _kl_col(j)
    history: List[float] = [_min_pair(acc)] if selected else []

    def _over_cap(j: int, cov: np.ndarray) -> bool:
        if max_per_attr is None:
            return False
        a = attr_of[j]
        return bool(len(a)) and bool(np.any(cov[a] + 1 > max_per_attr))

    # ---- 2. 覆盖阶段：先把每个属性的最低覆盖补够 ----
    # 单独成阶段，而不是与目标函数混在一起排序：max-min 目标在题量少时会大面积
    # 并列 0（很多模式对还分不开），混排会让贪心退化成"按题号顺序取题"。
    while len(selected) < n_items and np.any(cover < min_per_attr):
        lacking = np.where(cover < min_per_attr)[0]
        pool = [j for j in remaining
                if any(a in lacking for a in attr_of[j]) and not _over_cap(j, cover)]
        if not pool:
            pool = [j for j in remaining if any(a in lacking for a in attr_of[j])]
        if not pool:
            break

        def _fill_key(j: int) -> tuple:
            a = attr_of[j]
            fill = int(sum(1 for x in a if cover[x] < min_per_attr))
            trial = acc + _kl_col(j)
            return (fill, _min_pair(trial), _mean_pair(trial))

        best_j = max(pool, key=_fill_key)
        selected.append(best_j)
        remaining.remove(best_j)
        cover[attr_of[best_j]] += 1
        acc += _kl_col(best_j)
        history.append(_min_pair(acc))
        if verbose:
            print(f"  + 第 {len(selected):>3} 题 → I{best_j + 1}  [补覆盖]")

    # ---- 3. 目标阶段：剩余名额按 max-min KL 贪心 ----
    while len(selected) < n_items:
        cand = np.array(remaining, dtype=int)
        if cand.size == 0:
            break
        if max_per_attr is not None:
            ok = np.array([not _over_cap(j, cover) for j in cand])
            if not ok.any():
                # 硬约束不放宽：宁可报错，也不要交出一份违规的卷子
                raise ValueError(
                    f"无法在 max_per_attr={max_per_attr} 下选满 {n_items} 题："
                    f"已选 {len(selected)} 题、当前覆盖 {cover.tolist()}，"
                    "剩余题目都会让某个属性超过上限。请放宽上限或减少题量。"
                )
            cand = cand[ok]
        j = _best_add(acc, cand)
        if j < 0:
            break
        selected.append(j)
        remaining.remove(j)
        cover[attr_of[j]] += 1
        acc += _kl_col(j)
        history.append(_min_pair(acc))
        if verbose:
            print(f"  + 第 {len(selected):>3} 题 → I{j + 1}  最小KL={_min_pair(acc):.4f}")

    # ---- 3b. 名额已满但覆盖仍不足：换出"富余"题腾位置 ----
    while np.any(cover < min_per_attr):
        lacking = np.where(cover < min_per_attr)[0]
        pool = [j for j in remaining if any(a in lacking for a in attr_of[j])]
        if not pool:
            break
        surplus = [i for i in selected
                   if i not in set(must) and all(cover[a] > min_per_attr for a in attr_of[i])]
        if not surplus:
            break
        best_j = max(pool, key=lambda j: int(sum(1 for x in attr_of[j] if cover[x] < min_per_attr)))
        if _over_cap(best_j, cover):
            break
        out = min(surplus, key=lambda i: _min_pair(acc - _kl_col(i)))
        selected.remove(out)
        remaining.append(out)
        cover[attr_of[out]] -= 1
        acc -= _kl_col(out)
        selected.append(best_j)
        remaining.remove(best_j)
        cover[attr_of[best_j]] += 1
        acc += _kl_col(best_j)

    # ---- 4. 交换式局部搜索 ----
    n_swaps = 0
    if local_search and remaining and len(selected) > 0:
        frozen = set(must)          # 必含题不参与换出
        for _ in range(int(max_rounds)):
            if not remaining:
                break
            base = _min_pair(acc)
            best_gain, best_pair = 1e-12, None
            cand_in = np.array(remaining, dtype=int)
            mats_in = _kl_batch(cand_in)                   # (C, S, S)
            iu = np.triu_indices(s_total, k=1)
            for i in selected:
                if i in frozen:
                    continue
                acc_minus = acc - _kl_col(i)
                # 换出 i 后覆盖
                cov_minus = cover.copy()
                cov_minus[attr_of[i]] -= 1
                ok = np.ones(cand_in.size, dtype=bool)
                if max_per_attr is not None:
                    for c, j in enumerate(cand_in):
                        if _over_cap(j, cov_minus):
                            ok[c] = False
                trials = acc_minus[None, :, :] + mats_in
                mins = trials[:, iu[0], iu[1]].min(axis=1)
                mins = np.where(ok, mins, -np.inf)
                c = int(np.argmax(mins))
                gain = float(mins[c]) - base
                if np.isfinite(gain) and gain > best_gain:
                    # 交换后仍需满足覆盖要求
                    cov_new = cov_minus.copy()
                    cov_new[attr_of[int(cand_in[c])]] += 1
                    if np.all(cov_new >= min_per_attr):
                        best_gain, best_pair = gain, (i, int(cand_in[c]))
            if best_pair is None:
                break
            out, inn = best_pair
            selected.remove(out)
            remaining.append(out)
            cover[attr_of[out]] -= 1
            acc -= _kl_col(out)
            selected.append(inn)
            remaining.remove(inn)
            cover[attr_of[inn]] += 1
            acc += _kl_col(inn)
            n_swaps += 1
            history.append(_min_pair(acc))

    selected_arr = np.array(sorted(selected), dtype=int)
    # 目标函数按升序重算，避免累加顺序造成的舍入差异
    acc_final = np.zeros((s_total, s_total))
    for j in selected_arr:
        acc_final += _kl_col(int(j))
    score = _min_pair(acc_final)
    mean_kl = _mean_pair(acc_final)
    pmr, amr, pat_acc = _expected_accuracy(prob, selected_arr, prior_vec)
    cov_final = np.zeros(k, dtype=int)
    for j in selected_arr:
        cov_final[attr_of[int(j)]] += 1
    share = cov_final / cov_final.sum() if cov_final.sum() else np.zeros(k)

    # 硬约束最后再核一遍：宁可报错，也不要悄悄交出违规的卷子
    if max_per_attr is not None and np.any(cov_final > max_per_attr):
        raise ValueError(
            f"装配结果违反 max_per_attr={max_per_attr}：覆盖 {cov_final.tolist()}。"
            "题库结构可能不足以支撑该约束，请放宽上限或减少题量。"
        )

    if verbose:
        print(f"完成：{selected_arr.size} 题，最小KL={score:.4f}，理论PMR={pmr:.4f}")

    return AssemblyResult(
        selected=selected_arr,
        objective="maximin_pair_kl",
        score=score,
        mean_pair_kl=mean_kl,
        expected_pmr=pmr,
        expected_amr=np.asarray(amr, dtype=np.float64),
        pattern_accuracy=np.asarray(pat_acc, dtype=np.float64),
        attr_coverage=cov_final,
        attr_share=share,
        history=history,
        n_swaps=n_swaps,
        n_items_bank=bank,
        n_attrs=k,
        must_include=must,
        min_per_attr=min_per_attr,
        max_per_attr=max_per_attr,
        item_names=list(item_names) if item_names else [],
        attr_names=list(attr_names) if attr_names else list(q.attr_names or []),
    )
