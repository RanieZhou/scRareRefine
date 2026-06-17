"""批量评估：对每个 (config, seed, rare_train_size) 组合加载缓存 embedding，
输出 baseline + scRareRefine (gate_only / gate_marker / fusion) 对比表。

用法:
    python tools/analysis/evaluate_all.py
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocess import run_preprocessing
from src.utils import load_config, load_adata, make_run_dir, parse_rare_train_size, classification_tables
from src.rescue import run_post_hoc_rescue

RUNS = [
    ("configs/immune_dc.yaml",       42, "0.05"),
    ("configs/immune_dc.yaml",       43, "0.05"),
    ("configs/immune_dc.yaml",       44, "0.05"),
    ("configs/pancreas_baron.yaml",  42, "0.10"),
    ("configs/pancreas_baron.yaml",  43, "0.10"),
    ("configs/pancreas_baron.yaml",  44, "0.10"),
    ("configs/tabula_lung_endo.yaml", 42, "0.10"),
    ("configs/tabula_lung_endo.yaml", 43, "0.10"),
    ("configs/tabula_lung_endo.yaml", 44, "0.10"),
]

rows = []
for cfg_path, seed, rts_str in RUNS:
    config = load_config(cfg_path)
    exp = config.get("experiment", {})
    rare_class = exp.get("rare_class")
    label_column = config["dataset"].get("label_key", "label")
    batch_key = config["dataset"].get("batch_key", "batch")
    split_mode = exp.get("split_mode", "batch_heldout")
    size = parse_rare_train_size(rts_str)
    run_dir = make_run_dir(config, split_mode, seed, rare_class, size)
    emb_dir = run_dir / "embeddings"

    if not (emb_dir / "test_latent.csv").exists():
        print(f"[SKIP] {run_dir} 缓存不存在")
        continue

    splits = ["train", "validation", "test"]
    preds = {s: pd.read_csv(emb_dir / f"{s}_predictions.csv") for s in splits}
    lats  = {s: pd.read_csv(emb_dir / f"{s}_latent.csv")      for s in splits}
    genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()

    y_true = preds["test"]["true_label"].astype(str).to_numpy()
    base   = preds["test"]["predicted_label"].astype(str).to_numpy()
    bl, _  = classification_tables(y_true, base, rare_class=rare_class)
    dataset = config["dataset"]["name"]

    row_base = {
        "dataset": dataset, "seed": seed, "rare_train_size": rts_str,
        "rare_class": rare_class, "method": "baseline",
        **{k: round(v, 4) for k, v in bl.items()},
        "n_rescued": 0, "n_false_rescues": 0, "ffr": 0.0,
    }
    rows.append(row_base)
    print(f"[{dataset} seed={seed}] baseline rare_f1={bl['rare_f1']:.4f}")

    adata_raw = load_adata(config)
    adata, tr_idx, v_idx, te_idx = run_preprocessing(
        adata_raw, label_column=label_column, batch_key=batch_key,
        split_mode=split_mode, seed=seed, rare_class=rare_class)

    for strat in ["gate_only", "fusion", "conformal"]:
        # conformal 分支只读 conformal_alpha，不读 max_false_rescue_rate（语义独立，
        # 详见 src/rescue.py run_post_hoc_rescue docstring）；两个都传 0.001 才能让
        # 三种策略在同一 FFR 预算下做公平对比，否则 conformal 会静默退回默认 alpha=0.01。
        final_pred, summ = run_post_hoc_rescue(
            adata, preds, lats, genes, rare_class=rare_class,
            strategy=strat, max_false_rescue_rate=0.001, conformal_alpha=0.001)
        m, _ = classification_tables(y_true, final_pred.astype(str).to_numpy(), rare_class=rare_class)
        n_nonrare = int((y_true != rare_class).sum())
        row = {
            "dataset": dataset, "seed": seed, "rare_train_size": rts_str,
            "rare_class": rare_class, "method": f"scRareRefine_{strat}",
            **{k: round(v, 4) for k, v in m.items()},
            "n_rescued": summ["n_rescued"], "n_false_rescues": summ["n_false_rescues"],
            "ffr": round(summ["n_false_rescues"] / max(n_nonrare, 1), 6),
        }
        rows.append(row)
        print(f"  [{strat}] rare_f1={m['rare_f1']:.4f}  recall={m['rare_recall']:.4f}  "
              f"precision={m['rare_precision']:.4f}  rescued={summ['n_rescued']}  false={summ['n_false_rescues']}  ffr={row['ffr']:.5f}")

out_df = pd.DataFrame(rows)
out_path = Path("results/eval_summary.csv")
out_path.parent.mkdir(exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"\n[saved] {out_path}")
print(out_df.to_string(index=False))
