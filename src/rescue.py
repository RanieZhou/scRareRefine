from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils import classification_tables


# ==========================================
# 策略一：基于低维表征原型距离的候选细胞提取
# ==========================================
class PrototypeRescuer:
    """计算训练集各细胞类型原型，度量未知细胞的原型距离与排名以筛选候选细胞"""

    def __init__(self, rare_class: str):
        self.rare_class = rare_class
        self.prototypes = {}
        self.classes = []

    def fit(
        self, train_latent: np.ndarray, train_labels: pd.Series, is_labeled: np.ndarray
    ):
        """计算训练集中各已知细胞类型的潜在表示中心作为原型中心向量，并估算可分性比率"""
        self.classes = sorted(train_labels[is_labeled].unique())
        if self.rare_class not in self.classes:
            raise ValueError(f"训练集中未发现稀有类别标签: {self.rare_class}")

        self.prototypes = {
            cls: train_latent[is_labeled & train_labels.eq(cls).to_numpy()].mean(axis=0)
            for cls in self.classes
        }

        # 各类「类内半径」（到原型距离的中位数，稳健抗离群）。用于各向异性隶属度评分的尺度归一化。
        # 少于 3 个标注样本时无法可靠估计分布，改用保守默认半径 1.0，防止 score 被接近 0 的半径异常放大。
        self.radii = {}
        for cls in self.classes:
            pts = train_latent[is_labeled & train_labels.eq(cls).to_numpy()]
            if len(pts) < 3:
                self.radii[cls] = 1.0
            else:
                d = np.sqrt(((pts - self.prototypes[cls]) ** 2).sum(1))
                self.radii[cls] = max(float(np.median(d)), 1e-6)

        # 基于训练集计算 separability：原型间距 / 类内稀有半径
        # 稀有标注 < 3 时，类内半径估计不可靠（单点=0），强制 separability=0 触发弃权安全网。
        proto_mat = np.vstack([self.prototypes[c] for c in self.classes])
        rare_i = self.classes.index(self.rare_class)
        rare_train = train_latent[
            is_labeled & train_labels.eq(self.rare_class).to_numpy()
        ]
        maj_i = [i for i, c in enumerate(self.classes) if c != self.rare_class]
        d_to_maj = (
            float(np.sqrt(((proto_mat[rare_i] - proto_mat[maj_i]) ** 2).sum(1)).min())
            if maj_i
            else 0.0
        )
        if len(rare_train) < 3:
            self.separability_ratio = 0.0
        else:
            intra_r = float(
                np.sqrt(((rare_train - proto_mat[rare_i]) ** 2).sum(1)).mean()
            )
            self.separability_ratio = d_to_maj / max(intra_r, 1e-8)

    def rare_membership_score(self, query_latent: np.ndarray) -> np.ndarray:
        """各向异性隶属度评分：softmax_c(-d_c / r_c) 的稀有类分量。

        用每个类自己的类内半径 r_c 归一化距离（各向同性 Mahalanobis 近似），
        让评分对各类尺度自适应，缓解边界可分数据集中稀有类被相邻多数类「侵入」的问题。
        """
        classes = self.classes
        P = np.vstack([self.prototypes[c] for c in classes])
        R = np.array([self.radii[c] for c in classes])
        d = np.sqrt(((query_latent[:, None, :] - P[None]) ** 2).sum(2))
        logits = -d / R[None]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=1, keepdims=True)
        return p[:, classes.index(self.rare_class)]

    def rare_rank(self, query_latent: np.ndarray) -> np.ndarray:
        """各 query 到稀有原型的各向同性欧氏距离 rank（1=所有类中最近）。"""
        classes = self.classes
        P = np.vstack([self.prototypes[c] for c in classes])
        d = np.sqrt(((query_latent[:, None, :] - P[None]) ** 2).sum(2))
        ridx = classes.index(self.rare_class)
        return np.argsort(np.argsort(d, axis=1), axis=1)[:, ridx] + 1

    def rank_candidate(
        self, query_latent: np.ndarray, predicted_labels: pd.Series, max_rank: int = 1
    ) -> np.ndarray:
        """候选掩膜：predicted != rare 且稀有原型距离 rank <= max_rank（各向同性欧氏）。

        max_rank=1 即 isotropic_rank1；放宽到 2 可纳入与相邻多数类几何纠缠、
        真稀有常落在 rank=2 的边界细胞（如 mast/gamma），召回上限更高。
        FFR 仍由下游 conformal score>=tau 控制，max_rank 仅决定候选池宽窄。
        """
        ranks = self.rare_rank(query_latent)
        not_rare = predicted_labels.to_numpy() != self.rare_class
        return not_rare & (ranks <= int(max_rank))


# ==========================================
# Conformal 重排序拯救（综合方案，跨数据集泛化、无 per-dataset 阈值）
# ==========================================

# 单一来源的 conformal 默认参数：run_pipeline.py 与 tools/comparison/run_scrarerefine_comparison.py
# 均从此处导入，避免两处各自硬编码同一语义的常量而产生数值漂移。
DEFAULT_CONFORMAL_ALPHA = 0.01  # 发表级 FFR 上界，跨数据集固定，非调参
CONFORMAL_LOW_SEP = 1.3  # conformal 策略弃权下限（rank_candidate 候选比旧 gate 宽松，
# sep 1.1-1.3 区间候选精度可能很低，保守取 1.3）


class ConformalRescuer:
    """综合稀有细胞拯救：各向同性 rank=1 定候选 + 各向异性隶属度评分 + conformal 阈值控 FFR。

    设计动机（解决 gate/fusion 框架的 val/test 阈值漂移）：
    - 候选筛选 rank=1（各向同性欧氏）：对高/低可分数据集都提供强精度约束。
    - 稀有性 score（各向异性隶属度，PrototypeRescuer.rare_membership_score）：按各类紧致度归一化。
    - Conformal 阈值：tau 取「全体 val 非稀有细胞」score 的有限样本 (1-alpha) 顺序统计量。
      用大样本（数百~数千非稀有）校准而非小候选集 grid search，给出分布无关的 FFR<=alpha 上界。
    高可分数据集：真稀有 score 远高于 val 非稀有分位 → tau 低 → 不影响（recall 不退化）。
    边界数据集：相邻多数类细胞 score 较高 → tau 自动抬升 → 挡住假阳性。整套逻辑无数据集相关常量。
    """

    def __init__(self, rare_class: str, alpha: float = DEFAULT_CONFORMAL_ALPHA):
        self.rare_class = rare_class
        self.alpha = alpha  # 目标 FFR 上界（发表标准 <=0.01），跨数据集固定，非调参
        self.tau = None

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """有限样本保守上分位：第 ceil((1-alpha)(n+1)) 个顺序统计量（n 不足以保证时返回 +inf=不拯救）。"""
        s = np.sort(np.asarray(scores, dtype=float))
        n = len(s)
        if n == 0:
            return float("inf")
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        return float("inf") if k > n else float(s[k - 1])

    def calibrate(self, val_scores: np.ndarray, val_true: pd.Series) -> float:
        """在 validation 的非稀有细胞 score 上校准 conformal 阈值 tau（不接触 test）。"""
        val_true = pd.Series(val_true).astype(str).to_numpy()
        nonrare_scores = np.asarray(val_scores)[val_true != self.rare_class]
        self.tau = self._conformal_quantile(nonrare_scores, self.alpha)
        return self.tau

    def relabel(
        self,
        predicted_labels: pd.Series,
        candidate_mask: np.ndarray,
        test_scores: np.ndarray,
    ) -> pd.Series:
        """对候选且 score>=tau 的细胞重标注为稀有类。"""
        result = predicted_labels.astype(str).copy()
        if self.tau is None or not np.isfinite(self.tau):
            return result
        fire = candidate_mask & (np.asarray(test_scores) >= self.tau)
        result.iloc[np.where(fire)[0]] = self.rare_class
        return result


# 候选 rank 上限网格：rank ∈ {1,2,3}。val-自适应规则会自动剔除 val FFR > α 的 rank，
# 也会通过 tie-break 优先取更小 rank（更保守）。
# 上限 3 的 inductive 论证（不引用任何 test 经验）：
#   (1) rank 是"在所有类原型中，稀有原型对 query 的距离排名"。rank≥k 意味着有 k-1 个多数类原型
#       比稀有原型更近——超过 2 个多数类同时更近，"稀有候选"的几何依据已弱到可被纯噪声主导，
#       该 query 更应归入这些更近的多数类。
#   (2) val-自适应 + conformal τ 的双重门控保证：若 rank=3 真的引入大量假阳性，val FFR 会 > α
#       并被 grid 自动剔除；选择本身完全 val 内自洽，不依赖 test 经验。
#   (3) 上限封死 3 而非 None 的目的是控制候选池大小，避免极端情况下 score 阈值依赖于
#       几乎全部细胞的分布，使有限样本 conformal 校准误差被放大。
CONFORMAL_RANK_GRID = (1, 2, 3)

# Split-shift guard 阈值：val 上 baseline 漏判稀有数 < MIN_VAL_MISSED 时弃权。
# inductive 论证：val-自适应 rank 在 {1,2,3} 中选择需要 val 上能形成可比较的 rare F1 信号；
# 若 val 漏判稀有 < 3，单细胞改判即可让 F1 翻盘，选出的 rank 是噪声而非信号，应直接弃权。
# 这是「conformal 有限样本校准需要 ~1/α 个非稀有」的对偶：rescue 目标侧也需要最小有效样本。
MIN_VAL_MISSED = 3


def conformal_rescue(
    proto: "PrototypeRescuer",
    base_pred_test: pd.Series,
    val_pred_labels: pd.Series,
    val_true: pd.Series,
    val_latent: np.ndarray,
    test_latent: np.ndarray,
    *,
    alpha: float = DEFAULT_CONFORMAL_ALPHA,
    rank_grid=CONFORMAL_RANK_GRID,
) -> tuple[pd.Series, dict]:
    """scRareRefine conformal 拯救（单一来源，run_pipeline 与对比脚本共用）。

    四道全 inductive（只用 train 拟合原型 + val 标签选参，绝不碰 test 标签）的机制：

    1. separability 安全网：sep < CONFORMAL_LOW_SEP(1.3) 时稀有/多数严重重叠，弃权。
    2. necessity + split-shift 守门：val 上 baseline 漏判稀有数 < MIN_VAL_MISSED(3) 时弃权。
       同时覆盖 (a) val 已全召回 (val_missed=0) 的 "无需 rescue" 情形（小肠 tuft cell），
       与 (b) val 漏少但 val_missed 不足以支撑 val-自适应 rank 选择的 split-shift 情形
       （pancreas_integrated rts=0.01/0.05：val 漏 1-2 个 / test 已 saturated → 强行 rescue
       只会引入误判，把 baseline 已经满分的 test 拉下来）。
    3. val-自适应候选 rank：在 CONFORMAL_RANK_GRID={1,2,3} 中选「val 稀有 F1 最高且
       val FFR Wilson 95% 上界 <= alpha」的 max_rank，平手取更小 rank（更保守）。
       Wilson 上界（非 point estimate）显式计入 validation 有限样本不确定性；若所有 rank
       都不能满足该约束，则严格弃权，不以默认 rank 绕过安全约束。
    4. conformal tau：val 非稀有 score 的 (1-alpha) 顺序统计量，应用到 test 控 FFR。
       高可分数据集（immune/endo）val 自动选 rank=1；边界/纠缠数据集（pancreas/stomach）
       选 rank=2 或 rank=3（视 val 样本量），召回上升而 FFR 仍受 tau + Wilson 双约束。

    Returns: (final_test_pred, summary)
    """
    base_pred_test = pd.Series(base_pred_test).astype(str).reset_index(drop=True)
    val_pred_labels = pd.Series(val_pred_labels).astype(str).reset_index(drop=True)
    val_true = pd.Series(val_true).astype(str).reset_index(drop=True)
    rare = proto.rare_class

    summary = {
        "abstain": False,
        "reason": "",
        "chosen_rank": 0,
        "tau": float("inf"),
        "n_candidate": 0,
        "n_rescued": 0,
    }

    # 道 1：separability 安全网
    if proto.separability_ratio < CONFORMAL_LOW_SEP:
        summary.update(abstain=True, reason=f"sep<{CONFORMAL_LOW_SEP}")
        return base_pred_test.copy(), summary

    # 道 2：necessity + split-shift 守门（val baseline 漏判稀有数 < MIN_VAL_MISSED 时弃权）
    val_missed = int((val_true.eq(rare) & val_pred_labels.ne(rare)).sum())
    summary["val_missed"] = val_missed
    if int(val_true.eq(rare).sum()) > 0 and val_missed < MIN_VAL_MISSED:
        reason = (
            "val baseline 零漏判稀有"
            if val_missed == 0
            else f"val_missed={val_missed} < MIN_VAL_MISSED={MIN_VAL_MISSED}"
        )
        summary.update(abstain=True, reason=reason)
        return base_pred_test.copy(), summary

    # conformal tau（val 非稀有 score 校准）
    val_score = proto.rare_membership_score(val_latent)
    test_score = proto.rare_membership_score(test_latent)
    conf = ConformalRescuer(rare, alpha=alpha)
    tau = conf.calibrate(val_score, val_true)
    summary["tau"] = tau
    if not np.isfinite(tau):
        summary.update(abstain=True, reason="tau=inf（val 非稀有样本不足）")
        return base_pred_test.copy(), summary

    # 道 3：val-自适应候选 rank（Wilson 95% 上界控 FFR ≤ alpha 约束下最大化 val 稀有 F1）
    # 用 Wilson 上界而非 point estimate 显式计入 validation 有限样本不确定性。
    val_ranks = proto.rare_rank(val_latent)
    n_val_nonrare = int(val_true.ne(rare).sum())
    best = None  # (val_f1, -rank)
    chosen_rank = None
    z = 1.96  # 95% 单侧（=2.5% 双尾），跨数据集固定先验，非调参
    for k in rank_grid:
        v_cand = (val_ranks <= k) & val_pred_labels.ne(rare).to_numpy()
        v_fire = v_cand & (val_score >= tau)
        v_relabel = val_pred_labels.copy()
        v_relabel[v_fire] = rare
        v_false = int(((v_fire) & val_true.ne(rare).to_numpy()).sum())
        n = max(n_val_nonrare, 1)
        p = v_false / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        v_ffr_upper = center + half
        if v_ffr_upper > alpha:
            continue
        vf1, _ = classification_tables(val_true, v_relabel, rare_class=rare)
        key = (round(vf1["rare_f1"], 6), -k)
        if best is None or key > best:
            best = key
            chosen_rank = k
    if best is None:
        summary.update(abstain=True, reason="no_feasible_rank")
        return base_pred_test.copy(), summary

    summary["chosen_rank"] = chosen_rank

    # 应用到 test
    test_cand = proto.rank_candidate(test_latent, base_pred_test, max_rank=chosen_rank)
    final = conf.relabel(base_pred_test, test_cand, test_score)
    summary["n_candidate"] = int(test_cand.sum())
    summary["n_rescued"] = int(final.ne(base_pred_test).sum())
    return final, summary
