"""多 seed 嵌入生成驱动（G01 第十三轮）：幂等批量生成 scANVI 嵌入缓存。

对 SEEDS × 6 数据集 × 4 rts，逐个 subprocess 调 tools/analysis/train_cache.py。
train_cache.py 自身幂等（缓存+manifest 存在则跳过），故可随时中断/续跑。

不改任何已有 seed=42 结果；只新增 outputs/<ds>/<run_id> 目录。

用法：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/gen_multiseed_cache.py --seeds 43 44
  # 单测一个：--seeds 43 --only immune_dc --rts 0.01
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.utils import load_config, make_run_dir, parse_rare_train_size  # noqa: E402

CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_small_intestine.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/pancreas_integrated.yaml",
]
RTS = ["0.01", "0.05", "0.10", "all"]
TRAIN_CACHE = str(ROOT / "tools" / "analysis" / "train_cache.py")


def _cached(cfg_path: str, seed: int, rts: str) -> bool:
    cfg = load_config(cfg_path)
    exp = cfg.get("experiment", {})
    rd = make_run_dir(cfg, exp.get("split_mode", "batch_heldout"), seed,
                      exp.get("rare_class"), parse_rare_train_size(rts))
    return (rd / "embeddings" / "test_latent.csv").exists() and (rd / "manifest.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    ap.add_argument("--only", default=None, help="只跑 config 名含此子串的数据集")
    ap.add_argument("--rts", default=None, help="只跑此 rts（如 0.01）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    configs = [c for c in CONFIGS if (args.only is None or args.only in c)]
    rts_list = [args.rts] if args.rts else RTS

    jobs = [(c, s, r) for s in args.seeds for c in configs for r in rts_list]
    print(f"[plan] {len(jobs)} 个 (config,seed,rts) 任务  seeds={args.seeds}")

    done, skipped, failed = 0, 0, 0
    t_all = time.time()
    for i, (cfg, seed, rts) in enumerate(jobs, 1):
        ds = load_config(cfg)["dataset"]["name"]
        if not args.force and _cached(cfg, seed, rts):
            print(f"[{i}/{len(jobs)}] [skip] {ds} seed={seed} rts={rts}（缓存存在）")
            skipped += 1
            continue
        print(f"[{i}/{len(jobs)}] [run ] {ds} seed={seed} rts={rts} ...", flush=True)
        t0 = time.time()
        cmd = [sys.executable, TRAIN_CACHE, "--config", cfg, "--seed", str(seed), "--rare_train_size", rts]
        if args.force:
            cmd.append("--force")
        r = subprocess.run(cmd)
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"      done in {dt/60:.1f} min")
            done += 1
        else:
            print(f"      FAILED (rc={r.returncode}) after {dt/60:.1f} min")
            failed += 1

    print(f"\n[summary] done={done} skipped={skipped} failed={failed}  "
          f"总耗时 {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
