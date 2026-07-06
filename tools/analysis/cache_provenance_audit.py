"""Audit cached embedding provenance for publication results.

The audit is cache-only. It reads ``outputs/*/*/manifest.json`` plus cached
``train/validation/test_predictions.csv`` cell IDs, then checks whether the
manifest split hash matches the actual cached split.

Outputs:
  results/provenance/cache_audit.csv
  results/provenance/cache_audit.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import compute_cached_split_hash, git_sha  # noqa: E402

OUT_DIR = ROOT / "results" / "provenance"


def _required_files(run_dir: Path) -> list[Path]:
    emb = run_dir / "embeddings"
    return [
        emb / f"{split}_{kind}.csv"
        for split in ["train", "validation", "test"]
        for kind in ["predictions", "latent"]
    ]


def audit() -> pd.DataFrame:
    rows = []
    current_git = git_sha()
    for run_dir in sorted((ROOT / "outputs").glob("*/*")):
        emb = run_dir / "embeddings"
        if not emb.exists():
            continue

        manifest_path = run_dir / "manifest.json"
        required = _required_files(run_dir)
        row: dict[str, object] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "run_id": run_dir.name,
            "required_files_ok": all(p.exists() for p in required),
            "manifest_exists": manifest_path.exists(),
            "current_git_sha": current_git,
        }

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in [
                "dataset",
                "dataset_path",
                "split_mode",
                "seed",
                "rare_class",
                "rare_train_size",
                "split_hash",
                "git_sha",
                "n_train",
                "n_val",
                "n_test",
            ]:
                row[f"manifest_{key}"] = manifest.get(key)
        else:
            manifest = {}

        cached_hash = compute_cached_split_hash(run_dir)
        row["cached_split_hash"] = cached_hash or ""
        row["split_hash_ok"] = bool(cached_hash and manifest.get("split_hash") == cached_hash)

        manifest_git = str(manifest.get("git_sha", ""))
        if not manifest_git or manifest_git == "unknown":
            git_status = "legacy_unknown"
        elif current_git == "unknown":
            git_status = "current_unknown"
        elif manifest_git == current_git:
            git_status = "current"
        else:
            git_status = "different_commit"
        row["git_status"] = git_status

        rows.append(row)

    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "cache_audit.csv"
    md_path = OUT_DIR / "cache_audit.md"
    df.to_csv(csv_path, index=False)

    total = len(df)
    split_ok = int(df["split_hash_ok"].sum()) if total else 0
    required_ok = int(df["required_files_ok"].sum()) if total else 0
    manifest_ok = int(df["manifest_exists"].sum()) if total else 0
    git_counts = df["git_status"].value_counts().to_dict() if total else {}

    lines = [
        "# Cache Provenance Audit",
        "",
        f"- Cached run directories: {total}",
        f"- Required embedding files complete: {required_ok}/{total}",
        f"- Manifest present: {manifest_ok}/{total}",
        f"- Split hash matches cached cell IDs: {split_ok}/{total}",
        f"- Current git SHA: {git_sha()}",
        f"- Git status counts: {git_counts}",
        "",
        "Interpretation:",
        "- `split_hash_ok=False` is a hard provenance problem: cached cell IDs do not match the manifest.",
        "- `git_status=legacy_unknown` means the cache predates git-sha recording; use `--force` for strict reruns.",
        "- `git_status=different_commit` means embeddings were generated on another commit; inspect before publication reruns.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    df = audit()
    write_report(df)


if __name__ == "__main__":
    main()
