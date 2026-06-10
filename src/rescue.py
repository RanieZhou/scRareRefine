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
        """ 计算训练集中各已知细胞类型的潜在表示中心作为原型中心向量 """
        self.classes = sorted(train_labels[is_labeled].unique())
        if self.rare_class not in self.classes:
            raise ValueError(f"训练集中未发现稀有类别标签: {self.rare_class}")
            
        self.prototypes = {
            cls: train_latent[is_labeled & train_labels.eq(cls).to_numpy()].mean(axis=0)
            for cls in self.classes
        }

    def predict_scores(
        self,
        query_latent: np.ndarray,
        predicted_labels: pd.Series,
        margin: np.ndarray,
        margin_quantile: float = 0.25,
    ) -> pd.DataFrame:
        """ 计算测试细胞到各原型的距离，评估其距离稀有类的排名 (rank)，并标记候选人 """
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

        # 筛选不确定性较高的细胞：模型初步预测不是稀有类，但原型距离上稀有类位居前两位，且分类 margin 低于阈值
        threshold = float(np.quantile(margin, margin_quantile))
        candidates = (predicted_labels.to_numpy() != self.rare_class) & (ranks <= 2) & (margin <= threshold)

        return pd.DataFrame({
            f"distance_to_{self.rare_class}": rare_dist,
            "distance_to_pred": pred_dist,
            f"prototype_rank_{self.rare_class}": ranks,
            f"d_pred_minus_d_{self.rare_class}": pred_dist - rare_dist,
            "d_nearest_majority": d_nearest_majority,
            f"dist_ratio_{self.rare_class}": dist_ratio,
            "prototype_rescue_candidate": candidates,
        }, index=predicted_labels.index)

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
        best_combo = (1.0, 0.2, 0.5)
        
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
                    
        self.best_params = {
            "temperature": best_combo[0],
            "alpha": best_combo[1],
            "rare_prob_threshold": best_combo[2],
        }

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
) -> tuple[pd.Series, dict]:
    """ 端到端执行 Post-hoc 细胞身份精细校正与重标注。
    
    Args:
        adata: 全局 AnnData 表达矩阵
        predictions_dict: model 模块产出的预测结果 DataFrame 字典 ('train', 'validation', 'test')
        latents_dict: model 模块产出的潜表征 DataFrame 字典 ('train', 'validation', 'test')
        selected_genes: 训练用的 HVG 基因列表
        rare_class: 稀有细胞的真实类别名
        strategy: 拯救策略，可选值为 "gate_only" | "gate_marker" | "fusion"
        max_false_rescue_rate: 允许对非稀有类细胞进行重标定的最大容忍错判率
        
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
    
    val_scores = proto_rescuer.predict_scores(_latent_matrix(val_latent), val_pred["predicted_label"], val_pred["margin"].to_numpy())
    test_scores = proto_rescuer.predict_scores(_latent_matrix(test_latent), test_pred["predicted_label"], test_pred["margin"].to_numpy())

    # 提取 rank=1 候选细胞掩膜 (即初步预测不是稀有类但原型极其相似的细胞)
    def _rank1_mask(pred_df, score_df):
        return (score_df["prototype_rescue_candidate"] & score_df[f"prototype_rank_{rare_class}"].eq(1)).to_numpy(dtype=bool)

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
