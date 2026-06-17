from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from src.utils import classification_tables, log1p_cpm

# ==========================================
# 辅助函数：快速截取并转化 CPM 基因表达量
# ==========================================
def _load_expression_subset(adata, cell_ids: list[str], genes: list[str]) -> np.ndarray:
    """ 依指定的细胞ID与基因列表，提取子集并计算 CPM 标准化与 log1p 对数化表达值 """
    idx = adata.obs_names.isin(cell_ids)
    sub = adata[idx]
    
    # 按照输入 cell_ids 对子集样本进行对齐排序，防止发生样本错位
    id_pos = {cid: i for i, cid in enumerate(sub.obs_names)}
    ordered = [id_pos[c] for c in cell_ids if c in id_pos]
    sub = sub[ordered]
    
    var_idx = [sub.var_names.get_loc(g) for g in genes if g in sub.var_names]
    X = sub.X
    if sparse.issparse(X):
        X = X.toarray()
        
    return log1p_cpm(np.asarray(X, dtype=np.float32)[:, var_idx])

# ==========================================
# 策略一：基于低维表征原型距离的候选细胞提取
# ==========================================
class PrototypeRescuer:
    """ 计算训练集各细胞类型原型，度量未知细胞的原型距离与排名以筛选候选细胞 """
    def __init__(self, rare_class: str):
        self.rare_class = rare_class
        self.prototypes = {}
        self.classes = []

    def fit(self, train_latent: np.ndarray, train_labels: pd.Series, is_labeled: np.ndarray):
        """ 计算训练集中各已知细胞类型的潜在表示中心作为原型中心向量，并估算可分性比率 """
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
        rare_train = train_latent[is_labeled & train_labels.eq(self.rare_class).to_numpy()]
        maj_i = [i for i, c in enumerate(self.classes) if c != self.rare_class]
        d_to_maj = float(np.sqrt(((proto_mat[rare_i] - proto_mat[maj_i]) ** 2).sum(1)).min()) if maj_i else 0.0
        if len(rare_train) < 3:
            self.separability_ratio = 0.0
        else:
            intra_r = float(np.sqrt(((rare_train - proto_mat[rare_i]) ** 2).sum(1)).mean())
            self.separability_ratio = d_to_maj / max(intra_r, 1e-8)

    def rare_membership_score(self, query_latent: np.ndarray) -> np.ndarray:
        """ 各向异性隶属度评分：softmax_c(-d_c / r_c) 的稀有类分量。

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

    def isotropic_rank1(self, query_latent: np.ndarray, predicted_labels: pd.Series) -> np.ndarray:
        """ 候选掩膜：predicted != rare 且各向同性欧氏距离下稀有原型最近 (rank==1)。 """
        classes = self.classes
        P = np.vstack([self.prototypes[c] for c in classes])
        d = np.sqrt(((query_latent[:, None, :] - P[None]) ** 2).sum(2))
        rank1 = d.argmin(axis=1) == classes.index(self.rare_class)
        not_rare = predicted_labels.to_numpy() != self.rare_class
        return not_rare & rank1

    # 可分性弃权线（CLAUDE.md 既定先验：separability < 1.1 时稀有类与多数类严重重叠，prototype 无结构优势）。
    # 注意：此值是数据集无关的先验，不从 test 集选取；候选筛选的 margin_quantile / dratio_threshold 改由 validation 校准。
    LOW_SEP = 1.1

    def predict_scores(
        self,
        query_latent: np.ndarray,
        predicted_labels: pd.Series,
        margin: np.ndarray,
        margin_quantile: float = 0.25,
        dratio_threshold: float = 1.0,
        margin_threshold: float | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """ 计算测试细胞到各原型的距离，评估其距离稀有类的排名 (rank)，并标记候选人。

        候选条件（参数化，阈值由 validation 校准，见 select_gate_params_on_val）：
            not_rare AND rank==1 AND margin_ok AND (dist_ratio < dratio_threshold)
        - margin_threshold: 由 val 上预先计算的固定数值（优先于 margin_quantile，inductive 合规）。
        - margin_quantile: 仅当 margin_threshold=None 时有效；在当前 split 上重算分位点（内部调试用）。
        - dratio_threshold=1.0 表示不做几何过滤（rank==1 已隐含 dist_ratio<1）。
        - separability < LOW_SEP 时直接弃权（安全网，数据集无关先验）。
        """
        # 将各原型转化为堆叠矩阵
        proto_vecs = np.vstack([self.prototypes[cls] for cls in self.classes])
        diff = query_latent[:, None, :] - proto_vecs[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))

        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        rare_idx = class_to_idx[self.rare_class]

        # 度量到预测类与稀有类的原型距离
        pred_dist = np.array([
            distances[i, class_to_idx[pred]] if pred in class_to_idx else np.nan
            for i, pred in enumerate(predicted_labels)
        ])
        rare_dist = distances[:, rare_idx]

        # 计算细胞在距离稀有类别原型上的距离由近到远排名 (rank=1代表最接近)
        ranks = np.argsort(np.argsort(distances, axis=1), axis=1)[:, rare_idx] + 1

        non_rare_idx = [i for i, cls in enumerate(self.classes) if cls != self.rare_class]
        d_nearest_majority = distances[:, non_rare_idx].min(axis=1) if non_rare_idx else np.full(len(query_latent), np.nan)
        dist_ratio = np.where(d_nearest_majority > 1e-10, rare_dist / d_nearest_majority, np.nan)

        sep = getattr(self, "separability_ratio", 1.0)
        not_rare = predicted_labels.to_numpy() != self.rare_class
        rank1 = ranks == 1

        if sep < self.LOW_SEP:
            # 不可分：弃权安全网
            candidates = np.zeros(len(query_latent), dtype=bool)
        else:
            if margin_threshold is not None:
                # 使用 validation 上预先计算的固定阈值（inductive 合规：不依赖当前 split 分布）
                margin_ok = (margin <= margin_threshold) if np.isfinite(margin_threshold) else np.ones(len(query_latent), dtype=bool)
            elif margin_quantile >= 1.0:
                margin_ok = np.ones(len(query_latent), dtype=bool)
            else:
                threshold = float(np.quantile(margin, margin_quantile))
                margin_ok = margin <= threshold
            dratio_ok = (dist_ratio < dratio_threshold) | np.isnan(dist_ratio)
            candidates = not_rare & rank1 & margin_ok & dratio_ok

        if verbose:
            print(f"   [候选筛选] separability={sep:.3f}  margin_q={margin_quantile}  dratio_th={dratio_threshold}  候选数={int(candidates.sum())}")

        return pd.DataFrame({
            f"distance_to_{self.rare_class}": rare_dist,
            "distance_to_pred": pred_dist,
            f"prototype_rank_{self.rare_class}": ranks,
            f"d_pred_minus_d_{self.rare_class}": pred_dist - rare_dist,
            "d_nearest_majority": d_nearest_majority,
            f"dist_ratio_{self.rare_class}": dist_ratio,
            "prototype_rescue_candidate": candidates,
        }, index=predicted_labels.index)

    def select_gate_params_on_val(
        self,
        val_latent: np.ndarray,
        val_predicted: pd.Series,
        val_true: pd.Series,
        val_margin: np.ndarray,
        max_false_rescue_rate: float = 0.001,
    ) -> dict:
        """ 在 validation 集上用 FFR 约束 grid search 候选门控阈值 (margin_quantile, dratio_threshold)。

        Inductive：只用 val 的标签选阈值，不接触 test。返回的阈值之后同时应用于 val 与 test。
        目标：在 false rescue rate <= max_false_rescue_rate 约束下，最大化 val 候选对真稀有的召回。
        若 separability < LOW_SEP 或无任何组合带来正候选，则返回最严格组合（实际等于弃权）。
        """
        val_true = pd.Series(val_true).astype(str).reset_index(drop=True)
        val_predicted = pd.Series(val_predicted).astype(str).reset_index(drop=True)
        rare = self.rare_class
        # val 中被 baseline 误判的真稀有（候选筛选的召回目标）与非稀有总数（FFR 分母）
        missed_rare = (val_true.eq(rare) & val_predicted.ne(rare)).to_numpy()
        non_rare = val_true.ne(rare).to_numpy()
        n_nonrare = int(non_rare.sum())
        n_missed = int(missed_rare.sum())

        # 弃权 fallback：dratio_threshold=0.0 保证 dist_ratio<0 不可能 → 0 候选
        default = {"margin_threshold": None, "dratio_threshold": 0.0}
        if getattr(self, "separability_ratio", 1.0) < self.LOW_SEP or n_missed == 0 or n_nonrare == 0:
            return default

        margin_grid = [0.25, 0.5, 0.75, 1.0]   # 1.0 = 不过滤 margin
        dratio_grid = [0.90, 0.95, 1.0]        # 1.0 = 不过滤几何
        best = None  # (recall, -ffr, mq, dt)
        for mq in margin_grid:
            for dt in dratio_grid:
                scores = self.predict_scores(val_latent, val_predicted, val_margin,
                                             margin_quantile=mq, dratio_threshold=dt, verbose=False)
                cand = scores["prototype_rescue_candidate"].to_numpy(bool)
                n_cand = int(cand.sum())
                if n_cand == 0:
                    continue
                false_rescues = int((cand & non_rare).sum())
                ffr = false_rescues / n_nonrare
                if ffr > max_false_rescue_rate:
                    continue
                recall = int((cand & missed_rare).sum()) / n_missed
                key = (recall, -ffr, mq, dt)  # 优先高 recall，平手时优先低 FFR
                if best is None or key > best:
                    best = key
        if best is None:
            return default
        mq, dt = best[2], best[3]
        # 把 quantile level 转成 val 上的固定数值阈值；test 只使用此固定值，不依赖 test 自身分布
        val_margin_th = float("inf") if mq >= 1.0 else float(np.quantile(val_margin, mq))
        return {"margin_threshold": val_margin_th, "dratio_threshold": dt}

# ==========================================
# 策略二：基于 Marker 表达丰度验证的精细拯救
# ==========================================
class MarkerRescuer:
    """ 计算类别 Marker 签名，在验证集上搜寻并筛选最佳表达量 Margin 阈值进行测试集 relabel """
    def __init__(self, rare_class: str, max_false_rescue_rate: float = 0.001):
        self.rare_class = rare_class
        self.max_false_rescue_rate = max_false_rescue_rate
        self.signatures = {}

    def compute_marker_signatures(
        self,
        expression: np.ndarray,
        gene_names: list[str],
        labels: pd.Series,
        is_labeled: np.ndarray,
        top_n: int = 25,
        min_cells: int = 5,
    ):
        """ 基于表达矩阵均值差异计算各已知细胞类型的正向差异表达基因签名 """
        labels = pd.Series(labels).astype(str).reset_index(drop=True)
        is_labeled = np.asarray(is_labeled, dtype=bool)
        expr = np.asarray(expression, dtype=float)
        
        for label in sorted(labels[is_labeled].unique()):
            in_class = is_labeled & labels.eq(label).to_numpy()
            out_class = is_labeled & ~labels.eq(label).to_numpy()
            if int(in_class.sum()) < min_cells or int(out_class.sum()) == 0:
                continue
            diff = expr[in_class].mean(axis=0) - expr[out_class].mean(axis=0)
            top_idx = np.argsort(-diff)[:top_n]
            self.signatures[label] = [gene_names[i] for i in top_idx if diff[i] > 0]

    def score_candidates(
        self,
        expression: np.ndarray,
        candidates: pd.DataFrame,
        gene_names: list[str],
    ) -> pd.DataFrame:
        """ 计算候选人细胞在稀有类 Marker 基因上的均值，和在初步预测类 Marker 上的均值及 Margin """
        expr = np.asarray(expression, dtype=float)
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        
        rare_genes = [gene_to_idx[g] for g in self.signatures.get(self.rare_class, []) if g in gene_to_idx]
        rows = []
        for row_num, (_, row) in enumerate(candidates.iterrows()):
            pred = str(row["predicted_label"])
            pred_genes = [gene_to_idx[g] for g in self.signatures.get(pred, []) if g in gene_to_idx]
            
            rare_score = float(expr[row_num, rare_genes].mean()) if rare_genes else 0.0
            pred_score = float(expr[row_num, pred_genes].mean()) if pred_genes else 0.0
            margin = rare_score - pred_score
            rows.append({
                f"marker_score_{self.rare_class}": rare_score,
                "marker_score_predicted": pred_score,
                "marker_margin": margin,
                "marker_verified": margin > 0,
            })
        return pd.DataFrame(rows, index=candidates.index)

    def select_threshold_on_val(
        self,
        predictions: pd.DataFrame,
        scored_candidates: pd.DataFrame,
    ) -> float:
        """ 在验证集上跑网格搜索，选取使正常细胞被误拯救率低于 max_false_rescue_rate 的最适阈值 """
        y_true = predictions["true_label"].astype(str)
        baseline_pred = predictions["predicted_label"].astype(str)
        rare_errors = y_true.eq(self.rare_class) & baseline_pred.ne(self.rare_class)
        non_rare = y_true.ne(self.rare_class)
        
        margins = pd.to_numeric(scored_candidates["marker_margin"], errors="coerce").dropna()
        if margins.empty:
            return float("inf")
            
        # 搜集各种可能的分界分位数作为候选阈值
        quantiles = margins.quantile([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).tolist()
        thresholds = sorted({float(x) for x in quantiles + [-1.0, -0.5, 0.0, 0.5, 1.0]})
        
        best_threshold = float("inf")
        best_f1 = -1.0
        
        # 遍历阈值寻找最适边界
        for th in thresholds:
            verified = pd.to_numeric(scored_candidates["marker_margin"], errors="coerce").ge(th).fillna(False)
            verified_ids = set(scored_candidates.loc[verified, "cell_id"].astype(str)) if "cell_id" in scored_candidates.columns else set()
            
            relabeled = baseline_pred.copy()
            if "cell_id" in predictions.columns:
                relabeled.loc[predictions["cell_id"].astype(str).isin(verified_ids)] = self.rare_class
                
            n_verified = int(verified.sum())
            false_rescues = int(non_rare.loc[predictions["cell_id"].astype(str).isin(verified_ids)].sum()) if "cell_id" in predictions.columns and n_verified else 0
            false_rescue_rate = false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0
            
            # 满足最大假拯救率约束条件
            if false_rescue_rate <= self.max_false_rescue_rate:
                metrics, _ = classification_tables(y_true, relabeled, rare_class=self.rare_class)
                val_f1 = metrics["rare_f1"]
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_threshold = th
                    
        # 容错：如果全都不满足，则采取默认的最严格阈值（正无穷，即不拯救）
        return best_threshold

# ==========================================
# 策略三：基于概率混合融合的自适应软校正
# ==========================================
class FusionRescuer:
    """ 自适应软概率融合，基于验证集自动进行 grid search 优选超参数并在测试集上应用 """
    def __init__(self, rare_class: str, max_false_rescue_rate: float = 0.005):
        self.rare_class = rare_class
        self.max_false_rescue_rate = max_false_rescue_rate
        self.best_params = {}

    def get_prototype_probabilities(
        self,
        query_latent: np.ndarray,
        ref_latent: np.ndarray,
        ref_labels: pd.Series,
        ref_is_labeled: np.ndarray,
        temperature: float = 1.0,
    ) -> pd.DataFrame:
        """ 根据测试集与已知型原型的距离，利用 Softmax 转化为概率分布 """
        classes = sorted(ref_labels[ref_is_labeled].unique())
        proto_vecs = np.vstack([
            ref_latent[ref_is_labeled & ref_labels.eq(cls).to_numpy()].mean(axis=0)
            for cls in classes
        ])
        
        distances = np.sqrt(((query_latent[:, None, :] - proto_vecs[None, :, :]) ** 2).sum(axis=2))
        logits = -distances / max(temperature, 1e-8)
        logits -= logits.max(axis=1, keepdims=True)  # 防溢出
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return pd.DataFrame(probs, columns=classes)

    def gated_fuse(
        self,
        predictions: pd.DataFrame,
        p_scanvi: pd.DataFrame,
        p_proto: pd.DataFrame,
        candidate_mask: np.ndarray,
        alpha: float,
        rare_prob_threshold: float,
    ) -> pd.Series:
        """ 对指定候选人细胞，使用融合公式融合概率并进行 Relabel """
        result = predictions["predicted_label"].astype(str).copy()
        if not candidate_mask.any() or self.rare_class not in p_scanvi.columns:
            return result

        p_s_rare = p_scanvi[self.rare_class].to_numpy(dtype=float)[candidate_mask]
        p_p_rare = p_proto[self.rare_class].to_numpy(dtype=float)[candidate_mask]

        # 概率线性插值
        fused_rare = (1.0 - alpha) * p_p_rare + alpha * p_s_rare
        cand_indices = np.where(candidate_mask)[0]
        
        for idx, p_rare in zip(cand_indices, fused_rare):
            if p_rare >= rare_prob_threshold:
                result.iloc[idx] = self.rare_class
        return result

    def select_best_params_on_val(
        self,
        val_pred: pd.DataFrame,
        val_latent: np.ndarray,
        ref_latent: np.ndarray,
        ref_labels: pd.Series,
        ref_is_labeled: np.ndarray,
        val_mask: np.ndarray,
        baseline_accuracy: float,
    ):
        """ 对温度 T, 结合权重 alpha, 与最终激活阈值进行 Validation 网格优化 """
        prob_cols = [c for c in val_pred.columns if c.startswith("prob_")]
        scanvi_val = val_pred[prob_cols].rename(columns=lambda c: c.removeprefix("prob_"))
        
        # 寻找参数网格
        grid = [
            (temp, alpha, th)
            for temp in [0.5, 1.0, 2.0]
            for alpha in [0.0, 0.2, 0.4]
            for th in [0.3, 0.5, 0.7]
        ]
        
        best_f1 = -1.0
        best_combo = None  # 无合法组合时 → abstain，不使用硬编码默认参数

        for temp, alpha, th in grid:
            p_proto = self.get_prototype_probabilities(val_latent, ref_latent, ref_labels, ref_is_labeled, temperature=temp)
            fused_pred = self.gated_fuse(val_pred, scanvi_val, p_proto, val_mask, alpha, th)
            
            # 计算精度与误拯救率
            overall, _ = classification_tables(val_pred["true_label"], fused_pred, rare_class=self.rare_class)
            changed = val_pred["predicted_label"].astype(str).ne(fused_pred.astype(str))
            false_rescues = int((changed & val_pred["true_label"].astype(str).ne(self.rare_class) & fused_pred.eq(self.rare_class)).sum())
            non_rare_cnt = int(val_pred["true_label"].astype(str).ne(self.rare_class).sum())
            false_rescue_rate = false_rescues / non_rare_cnt if non_rare_cnt else 0.0
            
            if overall["overall_accuracy"] >= (baseline_accuracy - 0.005) and false_rescue_rate <= self.max_false_rescue_rate:
                if overall["rare_f1"] > best_f1:
                    best_f1 = overall["rare_f1"]
                    best_combo = (temp, alpha, th)
                    
        if best_combo is None:
            self.best_params = None  # 无满足 accuracy+FFR 约束的组合 → 外层 abstain
        else:
            self.best_params = {
                "temperature": best_combo[0],
                "alpha": best_combo[1],
                "rare_prob_threshold": best_combo[2],
            }

# ==========================================
# 策略四：Conformal 重排序拯救（综合方案，跨数据集泛化、无 per-dataset 阈值）
# ==========================================

# 单一来源的 conformal 默认参数：run_pipeline.py 与 tools/comparison/run_scrarerefine_comparison.py
# 均从此处导入，避免两处各自硬编码同一语义的常量而产生数值漂移。
DEFAULT_CONFORMAL_ALPHA = 0.01   # 发表级 FFR 上界，跨数据集固定，非调参
CONFORMAL_LOW_SEP = 1.3          # conformal 策略弃权下限（高于 PrototypeRescuer.LOW_SEP=1.1，因
                                  # conformal 用的 isotropic_rank1 候选比 gate 宽松，1.1-1.3 区间候选精度可能很低）


class ConformalRescuer:
    """ 综合稀有细胞拯救：各向同性 rank=1 定候选 + 各向异性隶属度评分 + conformal 阈值控 FFR。

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
        self.alpha = alpha   # 目标 FFR 上界（发表标准 <=0.01），跨数据集固定，非调参
        self.tau = None

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """ 有限样本保守上分位：第 ceil((1-alpha)(n+1)) 个顺序统计量（n 不足以保证时返回 +inf=不拯救）。 """
        s = np.sort(np.asarray(scores, dtype=float))
        n = len(s)
        if n == 0:
            return float("inf")
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        return float("inf") if k > n else float(s[k - 1])

    def calibrate(self, val_scores: np.ndarray, val_true: pd.Series) -> float:
        """ 在 validation 的非稀有细胞 score 上校准 conformal 阈值 tau（不接触 test）。 """
        val_true = pd.Series(val_true).astype(str).to_numpy()
        nonrare_scores = np.asarray(val_scores)[val_true != self.rare_class]
        self.tau = self._conformal_quantile(nonrare_scores, self.alpha)
        return self.tau

    def relabel(self, predicted_labels: pd.Series, candidate_mask: np.ndarray, test_scores: np.ndarray) -> pd.Series:
        """ 对候选且 score>=tau 的细胞重标注为稀有类。 """
        result = predicted_labels.astype(str).copy()
        if self.tau is None or not np.isfinite(self.tau):
            return result
        fire = candidate_mask & (np.asarray(test_scores) >= self.tau)
        result.iloc[np.where(fire)[0]] = self.rare_class
        return result

# ==========================================
# 8. 顶层端到端 Post-hoc 拯救流水线主入口
# ==========================================
def run_post_hoc_rescue(
    adata,
    predictions_dict: dict[str, pd.DataFrame],
    latents_dict: dict[str, pd.DataFrame],
    selected_genes: list[str],
    *,
    rare_class: str,
    strategy: str = "gate_marker",
    max_false_rescue_rate: float = 0.001,
    conformal_alpha: float = DEFAULT_CONFORMAL_ALPHA,
) -> tuple[pd.Series, dict]:
    """ 端到端执行 Post-hoc 细胞身份精细校正与重标注。

    Args:
        adata: 全局 AnnData 表达矩阵
        predictions_dict: model 模块产出的预测结果 DataFrame 字典 ('train', 'validation', 'test')
        latents_dict: model 模块产出的潜表征 DataFrame 字典 ('train', 'validation', 'test')
        selected_genes: 训练用的 HVG 基因列表
        rare_class: 稀有细胞的真实类别名
        strategy: 拯救策略，可选值为 "gate_only" | "gate_marker" | "fusion" | "conformal"
        max_false_rescue_rate: gate/marker/fusion 路径的 FFR 约束（默认 0.001）
        conformal_alpha: conformal 路径的 FFR 校准目标（默认 0.01，发表级 FFR 上界）；
                         与 max_false_rescue_rate 语义独立，不互相覆盖
        
    Returns:
        final_test_pred: 校正后最终的测试集细胞类别预测 Series (index为cell_id, values为预测值)
        metrics: 包含校正效果（如 F1 提升、实际拯救细胞数等）的统计字典
    """
    print(f"\n====== [scRareRefine 拯救中心] 激活后处理校正算法 (策略: {strategy}) ======")
    
    train_pred = predictions_dict["train"]
    val_pred = predictions_dict["validation"]
    test_pred = predictions_dict["test"]
    
    train_latent = latents_dict["train"]
    val_latent = latents_dict["validation"]
    test_latent = latents_dict["test"]

    # 1. 初始化原型计算与距离度量 (fit训练集，计算验证/测试集的原型距离)
    def _latent_matrix(df):
        return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()

    ref_lat = _latent_matrix(train_latent)
    ref_labels = train_pred["true_label"]
    ref_is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    
    proto_rescuer = PrototypeRescuer(rare_class)
    proto_rescuer.fit(ref_lat, ref_labels, ref_is_labeled)

    # ===== 策略四：Conformal 重排序（综合方案，默认） =====
    if strategy == "conformal":
        val_lat = _latent_matrix(val_latent)
        test_lat = _latent_matrix(test_latent)
        # 候选：各向同性 rank=1；评分：各向异性隶属度
        test_cand = proto_rescuer.isotropic_rank1(test_lat, test_pred["predicted_label"])
        val_score = proto_rescuer.rare_membership_score(val_lat)
        test_score = proto_rescuer.rare_membership_score(test_lat)

        conf = ConformalRescuer(rare_class, alpha=conformal_alpha)
        # 弃权安全网：separability 低于"rescue 有效"阈值时不拯救（模块级 CONFORMAL_LOW_SEP=1.3，
        # 高于全局 PrototypeRescuer.LOW_SEP=1.1，原因见该常量旁注释）。
        if proto_rescuer.separability_ratio < CONFORMAL_LOW_SEP:
            print(f"-> [Conformal] separability={proto_rescuer.separability_ratio:.3f} < {CONFORMAL_LOW_SEP}（rescue 有效下限），弃权不拯救。")
            final_test_pred = test_pred["predicted_label"].astype(str).copy()
        else:
            tau = conf.calibrate(val_score, val_pred["true_label"])
            final_test_pred = conf.relabel(test_pred["predicted_label"], test_cand, test_score)
            print(f"-> [Conformal] separability={proto_rescuer.separability_ratio:.3f} | alpha={conf.alpha} | "
                  f"tau={tau:.4f} | rank1候选={int(test_cand.sum())} | 实际拯救={int(final_test_pred.ne(test_pred['predicted_label'].astype(str)).sum())}")

        y_true_test = test_pred["true_label"].astype(str)
        base_pred_test = test_pred["predicted_label"].astype(str)
        n_rescued = int((final_test_pred.ne(base_pred_test) & final_test_pred.eq(rare_class)).sum())
        n_false_rescues = int((final_test_pred.ne(base_pred_test) & final_test_pred.eq(rare_class) & y_true_test.ne(rare_class)).sum())
        bl_metrics, _ = classification_tables(y_true_test, base_pred_test, rare_class=rare_class)
        final_metrics, _ = classification_tables(y_true_test, final_test_pred, rare_class=rare_class)
        summary = {
            "n_rescued": n_rescued, "n_false_rescues": n_false_rescues,
            "baseline_f1": bl_metrics["rare_f1"], "rescued_f1": final_metrics["rare_f1"],
            "f1_gain": final_metrics["rare_f1"] - bl_metrics["rare_f1"],
            "overall_accuracy": final_metrics["overall_accuracy"],
        }
        print(f"   [拯救完成] 拯救={n_rescued} | 误拯救={n_false_rescues} | "
              f"F1: {summary['baseline_f1']:.4f} -> {summary['rescued_f1']:.4f} (提升 {summary['f1_gain']:.4f})")
        print(f"====== [scRareRefine 拯救中心] 后处理精细校正计算运行完毕 ======\n")
        return final_test_pred, summary

    # Inductive：在 validation 上用 FFR 约束选候选门控阈值，再同时应用到 val 与 test（不接触 test 标签）
    gate_params = proto_rescuer.select_gate_params_on_val(
        _latent_matrix(val_latent), val_pred["predicted_label"], val_pred["true_label"],
        val_pred["margin"].to_numpy(), max_false_rescue_rate=max_false_rescue_rate,
    )
    print(f"-> [候选门控调优] separability={proto_rescuer.separability_ratio:.3f} | val 选定门控: {gate_params}")

    val_scores = proto_rescuer.predict_scores(_latent_matrix(val_latent), val_pred["predicted_label"], val_pred["margin"].to_numpy(), **gate_params)
    test_scores = proto_rescuer.predict_scores(_latent_matrix(test_latent), test_pred["predicted_label"], test_pred["margin"].to_numpy(), **gate_params)

    # 提取候选细胞掩膜（v2: prototype_rescue_candidate 已经是 rank==1，无需二次过滤）
    def _rank1_mask(pred_df, score_df):
        return score_df["prototype_rescue_candidate"].to_numpy(dtype=bool)

    val_mask = _rank1_mask(val_pred, val_scores)
    test_mask = _rank1_mask(test_pred, test_scores)
    
    print(f"-> [原型距离筛选] 验证集 rank-1 候选细胞数: {val_mask.sum()} | 测试集 rank-1 候选细胞数: {test_mask.sum()}")

    # 2. 执行不同的拯救校正策略
    final_test_pred = test_pred["predicted_label"].astype(str).copy()
    val_baseline_accuracy, _ = classification_tables(val_pred["true_label"], val_pred["predicted_label"], rare_class=rare_class)
    
    if strategy == "gate_only":
        # 仅根据原型距离 rank=1 进行重标定，不进行二次验证
        print("-> [后处理校正] 策略: 仅使用原型距离 rank-1 候选直接覆盖标注。")
        test_relabeled = test_pred["predicted_label"].astype(str).copy()
        test_relabeled.loc[test_mask] = rare_class
        final_test_pred = test_relabeled

    elif strategy == "gate_marker":
        # 联合差异基因 Marker 验证 (在 validation 选择最佳阈值，应用至 test)
        print("-> [后处理校正] 策略: 联合特异 Marker 基因表达量 Margin 阈值校验。")
        marker_rescuer = MarkerRescuer(rare_class, max_false_rescue_rate=max_false_rescue_rate)
        
        # 提取训练集与候选集表达量子集
        train_cell_ids = train_pred["cell_id"].astype(str).tolist()
        train_expr = _load_expression_subset(adata, train_cell_ids, selected_genes)
        marker_rescuer.compute_marker_signatures(train_expr, selected_genes, ref_labels, ref_is_labeled)
        
        # 验证集阈值优选
        val_candidates = val_pred.loc[val_mask].copy().reset_index(drop=True)
        if not val_candidates.empty:
            val_expr = _load_expression_subset(adata, val_candidates["cell_id"].astype(str).tolist(), selected_genes)
            val_scored = pd.concat([val_candidates, marker_rescuer.score_candidates(val_expr, val_candidates, selected_genes)], axis=1)
            selected_th = marker_rescuer.select_threshold_on_val(val_pred, val_scored)
        else:
            selected_th = float("inf")
            
        print(f"   [验证调优] 差异 Marker 校正决策阈值: {selected_th:.4f}")
        
        # 在测试集应用校验阈值
        test_candidates = test_pred.loc[test_mask].copy().reset_index(drop=True)
        if not test_candidates.empty and selected_th != float("inf"):
            test_expr = _load_expression_subset(adata, test_candidates["cell_id"].astype(str).tolist(), selected_genes)
            test_scored = pd.concat([test_candidates, marker_rescuer.score_candidates(test_expr, test_candidates, selected_genes)], axis=1)
            
            # relabel 符合条件的细胞
            verified = test_scored["marker_margin"].ge(selected_th).fillna(False)
            verified_ids = set(test_scored.loc[verified, "cell_id"].astype(str))
            final_test_pred.loc[test_pred["cell_id"].astype(str).isin(verified_ids)] = rare_class

    elif strategy == "fusion":
        # 自适应概率融合拯救
        print("-> [后处理校正] 策略: 自适应概率插值融合。")
        fusion_rescuer = FusionRescuer(rare_class, max_false_rescue_rate=max_false_rescue_rate)
        
        # 搜寻验证集最佳参数
        fusion_rescuer.select_best_params_on_val(
            val_pred, _latent_matrix(val_latent), ref_lat, ref_labels, ref_is_labeled, val_mask, val_baseline_accuracy["overall_accuracy"]
        )
        bp = fusion_rescuer.best_params
        if bp is None:
            print("   [验证调优] 融合: 无合法参数组合满足 accuracy+FFR 约束，弃权不拯救。")
        else:
            print(f"   [验证调优] 融合最佳超参: T={bp['temperature']}, alpha={bp['alpha']}, threshold={bp['rare_prob_threshold']}")
            # 应用至测试集
            prob_cols = [c for c in test_pred.columns if c.startswith("prob_")]
            scanvi_test = test_pred[prob_cols].rename(columns=lambda c: c.removeprefix("prob_"))
            p_proto_test = fusion_rescuer.get_prototype_probabilities(
                _latent_matrix(test_latent), ref_lat, ref_labels, ref_is_labeled, temperature=bp["temperature"]
            )
            final_test_pred = fusion_rescuer.gated_fuse(
                test_pred, scanvi_test, p_proto_test, test_mask, alpha=bp["alpha"], rare_prob_threshold=bp["rare_prob_threshold"]
            )

    # 3. 统计拯救结果与相比于 baseline 的收益指标
    y_true_test = test_pred["true_label"].astype(str)
    base_pred_test = test_pred["predicted_label"].astype(str)
    
    n_rescued = int((final_test_pred.ne(base_pred_test) & final_test_pred.eq(rare_class)).sum())
    n_false_rescues = int((final_test_pred.ne(base_pred_test) & final_test_pred.eq(rare_class) & y_true_test.ne(rare_class)).sum())
    
    bl_metrics, _ = classification_tables(y_true_test, base_pred_test, rare_class=rare_class)
    final_metrics, _ = classification_tables(y_true_test, final_test_pred, rare_class=rare_class)
    
    summary = {
        "n_rescued": n_rescued,
        "n_false_rescues": n_false_rescues,
        "baseline_f1": bl_metrics["rare_f1"],
        "rescued_f1": final_metrics["rare_f1"],
        "f1_gain": final_metrics["rare_f1"] - bl_metrics["rare_f1"],
        "overall_accuracy": final_metrics["overall_accuracy"],
    }
    
    print(f"   [拯救完成] 实际拯救细胞数: {n_rescued} | 误拯救数: {n_false_rescues}")
    print(f"   [效益对比] F1-Score: Baseline={summary['baseline_f1']:.4f} -> Rescued={summary['rescued_f1']:.4f} (提升: {summary['f1_gain']:.4f})")
    print(f"====== [scRareRefine 拯救中心] 后处理精细校正计算运行完毕 ======\n")
    
    return final_test_pred, summary
