from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scrare_refine.config import load_config, output_dir
from scrare_refine.output_layout import ARTIFACT_FILES, classify_root_file


def plan_output_moves(root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(root)
    moves: list[tuple[Path, Path]] = []

    for path in root.iterdir() if root.exists() else []:
        if not path.is_file():
            continue
        kind = classify_root_file(path.name)
        if kind == "misc":
            continue
        moves.append((path, root / kind / path.name))

    runs_dir = root / "runs"
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for filename in ARTIFACT_FILES:
                src = run_dir / filename
                if src.exists():
                    moves.append((src, run_dir / "artifacts" / filename))
    return moves


def apply_moves(moves: list[tuple[Path, Path]], *, overwrite: bool = False) -> list[dict[str, str]]:
    manifest = []
    for src, dst in moves:
        if src.resolve() == dst.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not overwrite:
                manifest.append({"source": str(src), "destination": str(dst), "status": "skipped_exists"})
                continue
            dst.unlink()
        src.replace(dst)
        manifest.append({"source": str(src), "destination": str(dst), "status": "moved"})
    return manifest


def write_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "tables" / "output_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "destination", "status"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize scRareRefine output files into stage folders.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = output_dir(config)
    moves = plan_output_moves(root)
    if args.dry_run:
        for src, dst in moves:
            print(f"{src} -> {dst}")
        return
    manifest = apply_moves(moves, overwrite=args.overwrite)
    manifest_path = write_manifest(root, manifest)
    print(f"Moved/skipped {len(manifest)} files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
