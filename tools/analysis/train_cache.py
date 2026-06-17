"""训练 scANVI 并把 train/val/test 的 predictions+latent 缓存到 outputs/.../embeddings/。

缓存一次后，rescue 迭代与诊断均可离线进行，无需重训。

用法:
    python tools/analysis/train_cache.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocess import run_preprocessing
from src.model import run_model_training
from src.utils import (
    load_config, load_adata, make_run_dir, parse_rare_train_size, write_table,
    build_manifest,
)


def _build_manifest(config, args, label_column, batch_key, split_mode,
                    rare_class, parsed_size, pred_dict, n_train, n_val, n_test) -> dict:
    """构建 provenance manifest（参数 + split 哈希 + 代码版本）。委托给 src.utils.build_manifest
    （与 run_pipeline.py / run_scrarerefine_comparison.py 共用同一实现，避免漂移）。"""
    return build_manifest(
        config, args.config,
        label_column=label_column, batch_key=batch_key, split_mode=split_mode,
        seed=args.seed, rare_class=rare_class, rare_train_size=parsed_size,
        predictions_dict=pred_dict, n_train=n_train, n_val=n_val, n_test=n_test,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rare_train_size", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    exp = config.get("experiment", {})
    rare_class = exp.get("rare_class")
    label_column = config["dataset"].get("label_key", "label")
    batch_key = config["dataset"].get("batch_key", "batch")
    split_mode = exp.get("split_mode", "batch_heldout")
    parsed_size = parse_rare_train_size(args.rare_train_size)

    run_dir = make_run_dir(config, split_mode, args.seed, rare_class, parsed_size)
    emb_dir = run_dir / "embeddings"
    hvg_file = run_dir / "selected_hvg_genes.csv"
    splits = ["train", "validation", "test"]
    required = [emb_dir / f"{s}_{t}.csv" for s in splits for t in ("predictions", "latent")] + [hvg_file]
    if not args.force and all(p.exists() for p in required):
        mf = run_dir / "manifest.json"
        if not mf.exists():   # 旧缓存补写 manifest（从已有 predictions csv 读 split）
            import pandas as pd
            pd_dict = {s: pd.read_csv(emb_dir / f"{s}_predictions.csv") for s in splits}
            manifest = _build_manifest(
                config, args, label_column, batch_key, split_mode, rare_class, parsed_size,
                pd_dict, len(pd_dict["train"]), len(pd_dict["validation"]), len(pd_dict["test"]))
            mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[skip+manifest] 缓存已存在，补写 provenance: split_hash={manifest['split_hash']}")
        else:
            print(f"[skip] 缓存已存在: {emb_dir}")
        return

    adata_raw = load_adata(config)
    adata, train_idx, val_idx, test_idx = run_preprocessing(
        adata_raw, label_column=label_column, batch_key=batch_key,
        split_mode=split_mode, seed=args.seed, rare_class=rare_class,
    )
    _, predictions_dict, latents_dict, selected_genes = run_model_training(
        adata, train_idx, val_idx, test_idx,
        label_column=label_column, batch_key=batch_key, rare_class=rare_class,
        rare_train_size=parsed_size, config=config, seed=args.seed,
    )

    emb_dir.mkdir(parents=True, exist_ok=True)
    for s in splits:
        write_table(predictions_dict[s], emb_dir / f"{s}_predictions.csv")
        write_table(latents_dict[s], emb_dir / f"{s}_latent.csv")
    import pandas as pd
    write_table(pd.DataFrame({"gene": selected_genes}), hvg_file)

    # provenance manifest：记录参数 + split 哈希 + 代码版本，供消费端校验
    manifest = _build_manifest(
        config, args, label_column, batch_key, split_mode, rare_class, parsed_size,
        predictions_dict, len(train_idx), len(val_idx), len(test_idx))
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 已缓存 embeddings 至: {emb_dir}")
    print(f"[manifest] split_hash={manifest['split_hash']}  git_sha={manifest['git_sha']}")


if __name__ == "__main__":
    main()
