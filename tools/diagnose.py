"""Step 1 失败诊断：加载缓存 embedding，量化候选筛选 / 可分性 / 阈值泛化 / 融合分解 / 误拯救来源。

用法:
    python tools/diagnose.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05
"""
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import run_preprocessing
from src.utils import load_config, load_adata, make_run_dir, parse_rare_train_size, classification_tables
from src.rescue import PrototypeRescuer, MarkerRescuer, _load_expression_subset, run_post_hoc_rescue


def _latent(df):
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def load_cache(run_dir):
    emb = run_dir / "embeddings"
    splits = ["train", "validation", "test"]
    preds = {s: pd.read_csv(emb / f"{s}_predictions.csv") for s in splits}
    lats = {s: pd.read_csv(emb / f"{s}_latent.csv") for s in splits}
    genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()
    return preds, lats, genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rare_train_size", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    exp = config.get("experiment", {})
    rare_class = exp.get("rare_class")
    label_column = config["dataset"].get("label_key", "label")
    batch_key = config["dataset"].get("batch_key", "batch")
    split_mode = exp.get("split_mode", "batch_heldout")
    size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, split_mode, args.seed, rare_class, size)

    preds, lats, genes = load_cache(run_dir)
    train_pred, val_pred, test_pred = preds["train"], preds["validation"], preds["test"]

    ref_lat = _latent(lats["train"])
    ref_labels = train_pred["true_label"]
    ref_is_lab = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()

    pr = PrototypeRescuer(rare_class)
    pr.fit(ref_lat, ref_labels, ref_is_lab)
    val_scores = pr.predict_scores(_latent(lats["validation"]), val_pred["predicted_label"], val_pred["margin"].to_numpy())
    test_scores = pr.predict_scores(_latent(lats["test"]), test_pred["predicted_label"], test_pred["margin"].to_numpy())

    def rank1_mask(score_df):
        return (score_df["prototype_rescue_candidate"] & score_df[f"prototype_rank_{rare_class}"].eq(1)).to_numpy(bool)

    test_mask = rank1_mask(test_scores)
    val_mask = rank1_mask(val_scores)

    y_true = test_pred["true_label"].astype(str).to_numpy()
    base_pred = test_pred["predicted_label"].astype(str).to_numpy()

    R = {}
    R["config"] = args.config
    R["seed"] = args.seed
    R["rare_train_size"] = str(size)
    R["rare_class"] = rare_class
    R["n_test"] = int(len(y_true))
    R["n_test_rare"] = int((y_true == rare_class).sum())

    # baseline
    bl, _ = classification_tables(y_true, base_pred, rare_class=rare_class)
    R["baseline_rare_f1"] = round(bl["rare_f1"], 4)
    R["baseline_rare_recall"] = round(bl["rare_recall"], 4)
    R["baseline_rare_precision"] = round(bl["rare_precision"], 4)

    # ---- A. 候选筛选质量 (test) ----
    missed = (y_true == rare_class) & (base_pred != rare_class)  # baseline 误判的真稀有
    cand = test_mask
    cand_and_rare = cand & (y_true == rare_class)
    R["A_n_missed_rare"] = int(missed.sum())
    R["A_n_candidates"] = int(cand.sum())
    R["A_candidate_recall"] = round(float((cand & missed).sum() / max(missed.sum(), 1)), 4)
    R["A_candidate_precision"] = round(float(cand_and_rare.sum() / max(cand.sum(), 1)), 4)
    # rank<=2 召回上限（未叠加 margin 过滤前）
    rank_le2 = (test_scores[f"prototype_rank_{rare_class}"].to_numpy() <= 2) & (base_pred != rare_class)
    R["A_rank2_recall_ceiling"] = round(float((rank_le2 & missed).sum() / max(missed.sum(), 1)), 4)
    rank1 = (test_scores[f"prototype_rank_{rare_class}"].to_numpy() == 1) & (base_pred != rare_class)
    R["A_rank1_recall_ceiling"] = round(float((rank1 & missed).sum() / max(missed.sum(), 1)), 4)

    # ---- B. 原型空间可分性 ----
    classes = pr.classes
    proto = np.vstack([pr.prototypes[c] for c in classes])
    rare_i = classes.index(rare_class)
    # intra-rare radius: train labeled rare 到 rare proto 的均值距离
    train_rare = ref_is_lab & ref_labels.eq(rare_class).to_numpy()
    if train_rare.sum() > 0:
        d_intra = np.sqrt(((ref_lat[train_rare] - proto[rare_i]) ** 2).sum(1)).mean()
    else:
        d_intra = np.nan
    # dist rare proto -> nearest majority proto
    maj_i = [i for i in range(len(classes)) if i != rare_i]
    d_proto_maj = np.sqrt(((proto[rare_i] - proto[maj_i]) ** 2).sum(1)).min()
    R["B_intra_rare_radius"] = round(float(d_intra), 4)
    R["B_dist_rareproto_nearest_majproto"] = round(float(d_proto_maj), 4)
    R["B_separability"] = round(float(d_proto_maj / d_intra), 4) if d_intra and d_intra > 0 else None
    # test 真稀有细胞：到 rare proto vs 到最近 majority proto
    test_lat = _latent(lats["test"])
    test_rare_lat = test_lat[y_true == rare_class]
    if len(test_rare_lat):
        d_to_rare = np.sqrt(((test_rare_lat - proto[rare_i]) ** 2).sum(1))
        d_to_maj = np.sqrt(((test_rare_lat[:, None, :] - proto[maj_i][None]) ** 2).sum(2)).min(1)
        R["B_testrare_closer_to_rare_frac"] = round(float((d_to_rare < d_to_maj).mean()), 4)

    # ---- D. 融合分解：三种策略 test F1 ----
    adata_raw = load_adata(config)
    adata, tr_idx, v_idx, te_idx = run_preprocessing(
        adata_raw, label_column=label_column, batch_key=batch_key,
        split_mode=split_mode, seed=args.seed, rare_class=rare_class)
    strat_f1 = {}
    for strat in ["gate_only", "gate_marker", "fusion", "conformal"]:
        final_pred, summ = run_post_hoc_rescue(
            adata, preds, lats, genes, rare_class=rare_class, strategy=strat,
            max_false_rescue_rate=0.001)
        m, _ = classification_tables(y_true, final_pred.astype(str).to_numpy(), rare_class=rare_class)
        strat_f1[strat] = {
            "rare_f1": round(m["rare_f1"], 4), "rare_recall": round(m["rare_recall"], 4),
            "rare_precision": round(m["rare_precision"], 4),
            "n_rescued": summ["n_rescued"], "n_false_rescues": summ["n_false_rescues"]}
    R["D_strategies"] = strat_f1

    # ---- C. 阈值泛化 (gate_marker)：val 选阈值 + test 实际 FFR ----
    mr = MarkerRescuer(rare_class, max_false_rescue_rate=0.001)
    train_ids = train_pred["cell_id"].astype(str).tolist()
    train_expr = _load_expression_subset(adata, train_ids, genes)
    mr.compute_marker_signatures(train_expr, genes, ref_labels, ref_is_lab)
    val_cand = val_pred.loc[val_mask].copy().reset_index(drop=True)
    if not val_cand.empty:
        val_expr = _load_expression_subset(adata, val_cand["cell_id"].astype(str).tolist(), genes)
        val_scored = pd.concat([val_cand, mr.score_candidates(val_expr, val_cand, genes)], axis=1)
        sel_th = mr.select_threshold_on_val(val_pred, val_scored)
    else:
        sel_th = float("inf")
    R["C_selected_marker_threshold"] = None if sel_th == float("inf") else round(float(sel_th), 4)
    R["C_n_val_candidates"] = int(val_mask.sum())
    # test 实际 FFR at sel_th
    test_cand = test_pred.loc[test_mask].copy().reset_index(drop=True)
    if not test_cand.empty and sel_th != float("inf"):
        test_expr = _load_expression_subset(adata, test_cand["cell_id"].astype(str).tolist(), genes)
        test_scored = pd.concat([test_cand, mr.score_candidates(test_expr, test_cand, genes)], axis=1)
        verified = test_scored["marker_margin"].ge(sel_th).fillna(False)
        vids = set(test_scored.loc[verified, "cell_id"].astype(str))
        in_test = test_pred["cell_id"].astype(str).isin(vids).to_numpy()
        n_ver = int(in_test.sum())
        n_false = int((in_test & (y_true != rare_class)).sum())
        nonrare = int((y_true != rare_class).sum())
        R["C_test_n_verified"] = n_ver
        R["C_test_FFR_at_selected_th"] = round(n_false / max(nonrare, 1), 6)

    # ---- E. 误拯救来源 (gate_only：所有 rank1 候选直接 relabel) ----
    false_cand = cand & (y_true != rare_class)
    if false_cand.sum() > 0:
        src = pd.Series(y_true[false_cand]).value_counts().to_dict()
        R["E_false_candidate_sources"] = src
        R["E_n_false_candidates"] = int(false_cand.sum())
    else:
        R["E_false_candidate_sources"] = {}
        R["E_n_false_candidates"] = 0

    print(json.dumps(R, ensure_ascii=False, indent=2))
    out = run_dir / "diagnosis.json"
    out.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
