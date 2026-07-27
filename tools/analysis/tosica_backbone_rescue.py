"""Evaluate the unchanged scRareRefine rescue on native TOSICA embeddings.

This is a single-seed backbone-portability experiment. TOSICA is trained only
on cells carrying the frozen ``is_labeled_for_scanvi`` permission. Its native
48-dimensional CLS representation and its own predictions are extracted for
the labeled training reference, validation cells, and test cells. The existing
``PrototypeRescuer`` and ``conformal_rescue`` implementation is then applied
without changing alpha, separability, rank, or minimum-validation-support
constants.

Scientific boundary
-------------------
No scANVI latent vector, probability, or predicted label enters this analysis.
The existing cache supplies only cell IDs, the train/validation/test split,
label visibility, true labels, and train-only variance-selected HVG identities.
Validation labels select rescue rank and calibrate tau. Test labels are read
only after TOSICA and refined test predictions have been frozen.

Run from the TOSICA environment, for example::

    D:/setup/anaconda/envs/sandbox310/python.exe \
        tools/analysis/tosica_backbone_rescue.py

The default grid is eight datasets x seed 42 x four rare-label budgets = 32
runs. CLI overrides are intentionally supported so seeds 43/44 can be added
without editing this file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rescue import (  # noqa: E402
    CONFORMAL_LOW_SEP,
    CONFORMAL_RANK_GRID,
    DEFAULT_CONFORMAL_ALPHA,
    MIN_VAL_MISSED,
    PrototypeRescuer,
    conformal_rescue,
)
from src.utils import (  # noqa: E402
    check_manifest,
    classification_tables,
    load_config,
    make_run_dir,
    parse_rare_train_size,
)
from tools.comparison._conda_python import conda_python  # noqa: E402


CONFIGS = (
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/pancreas_integrated.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/tabula_small_intestine.yaml",
    "configs/mouse_lung_tms_10x.yaml",
    "configs/mouse_pancreas_tms_10x.yaml",
)
SEEDS = (42,)
RARE_TRAIN_SIZES = ("0.01", "0.05", "0.10", "all")
GMT_MAP = {
    "immune_dc": "human_gobp",
    "pancreas_baron": "human_gobp",
    "pancreas_integrated": "human_gobp",
    "tabula_lung_endo": "human_gobp",
    "tabula_sapiens_stomach": "human_gobp",
    "tabula_small_intestine": "human_gobp",
    "mouse_lung_tms_10x": "mouse_gobp",
    "mouse_pancreas_tms_10x": "mouse_gobp",
}

VERSION = "v1"
OUT = ROOT / "results" / "tosica_backbone_rescue" / VERSION
LOG_DIR = ROOT / "logs" / "tosica_backbone_rescue"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "tosica_backbone_rescue" / VERSION
PER_RUN_DIRNAME = "tosica_backbone_rescue_v1"
TOSICA_EPOCHS = 10
TOSICA_MAX_GS = 300
TOSICA_EMBED_DIM = 48
# The published implementation fixes its internal train/validation split and
# network RNG to 1. Project seed 42 controls the frozen biological split and
# labeled-rare identities; this implementation detail is recorded explicitly.
TOSICA_INTERNAL_SEED = 1
# Official TOSICA prediction default. Unknown remains a genuine TOSICA output;
# it is never replaced by a scANVI label in this portability experiment.
TOSICA_PREDICTION_CUTOFF = 0.10


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_values(values: list[str]) -> str:
    payload = json.dumps(sorted(map(str, values)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_key(dataset: str, seed: int, rare_train_size: str) -> str:
    safe_rts = str(rare_train_size).replace(".", "p")
    return f"{dataset}_seed{seed}_rare{safe_rts}"


def build_grid(
    configs: list[str] | tuple[str, ...] = CONFIGS,
    seeds: list[int] | tuple[int, ...] = SEEDS,
    rare_train_sizes: list[str] | tuple[str, ...] = RARE_TRAIN_SIZES,
) -> list[tuple[str, int, str]]:
    """Return the closed config x seed x budget experiment grid."""
    grid = [
        (str(config), int(seed), str(rts))
        for config in configs
        for seed in seeds
        for rts in rare_train_sizes
    ]
    if len(grid) != len(set(grid)):
        raise ValueError("Duplicate TOSICA backbone experiment keys")
    return grid


def _normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique())
        raise ValueError(f"Invalid is_labeled_for_scanvi values: {bad}")
    return normalized.astype(bool)


def _load_split(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / "embeddings" / f"{split}_predictions.csv"
    frame = pd.read_csv(path, dtype={"cell_id": str})
    required = {"cell_id", "true_label"}
    if split == "train":
        required.add("is_labeled_for_scanvi")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["cell_id"].duplicated().any():
        raise ValueError(f"Duplicate cell_id in {path}")
    return frame.reset_index(drop=True)


def _validate_splits(frames: dict[str, pd.DataFrame]) -> None:
    id_sets = {name: set(frame["cell_id"].astype(str)) for name, frame in frames.items()}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = id_sets[left].intersection(id_sets[right])
        if overlap:
            raise ValueError(f"{len(overlap)} cell IDs overlap between {left} and {right}")


def _expression_from_adata(
    adata,
    config: dict[str, Any],
    ids_by_split: dict[str, list[str]],
    genes: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Extract ordered source expression without modifying the source AnnData."""
    obs_names = pd.Index(adata.obs_names.astype(str))
    if obs_names.duplicated().any():
        raise ValueError("Source AnnData contains duplicate obs_names")
    id_to_row = pd.Series(np.arange(adata.n_obs), index=obs_names)

    use_raw = bool(config["dataset"].get("use_raw", False))
    use_layer = config["dataset"].get("use_layer")
    source_genes = pd.Index(
        adata.raw.var_names.astype(str) if use_raw and adata.raw is not None else adata.var_names.astype(str)
    )
    available = [gene for gene in genes if gene in source_genes]
    if not available:
        raise ValueError("No cached HVG is present in the source expression matrix")

    arrays: dict[str, np.ndarray] = {}
    for split, cell_ids in ids_by_split.items():
        missing = [cell_id for cell_id in cell_ids if cell_id not in id_to_row.index]
        if missing:
            raise ValueError(f"{len(missing)} {split} cell IDs are absent from source AnnData")
        rows = id_to_row.loc[cell_ids].to_numpy(dtype=int)
        if use_raw and adata.raw is not None:
            matrix = adata.raw[rows, available].X
        else:
            subset = adata[rows, available]
            matrix = subset.layers[use_layer] if use_layer and use_layer in subset.layers else subset.X
        if sp.issparse(matrix):
            matrix = matrix.toarray()
        array = np.asarray(matrix, dtype=np.float32)
        if array.shape != (len(cell_ids), len(available)) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {split} expression matrix: {array.shape}")
        arrays[split] = array
    return arrays, available


def _write_extract_spec(
    path: Path,
    config_path: str,
    ids_by_split: dict[str, list[str]],
    genes: list[str],
) -> None:
    payload = {
        "config_path": config_path,
        "ids_by_split": ids_by_split,
        "genes": genes,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _extract_to_npz(spec_path: Path, output_path: Path) -> None:
    """Extraction-only entry point used in the scanvi311 compatibility process."""
    import anndata as ad

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    config = load_config(ROOT / spec["config_path"])
    adata = ad.read_h5ad(ROOT / config["dataset"]["path"])
    arrays, genes = _expression_from_adata(
        adata,
        config,
        {key: list(map(str, value)) for key, value in spec["ids_by_split"].items()},
        list(map(str, spec["genes"])),
    )
    np.savez_compressed(
        output_path,
        genes=np.asarray(genes, dtype=str),
        **{f"X_{key}": value for key, value in arrays.items()},
    )


def _extract_expression(
    config_path: str,
    config: dict[str, Any],
    ids_by_split: dict[str, list[str]],
    genes: list[str],
    work_dir: Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Extract in sandbox310, falling back to scanvi311 for incompatible h5ad files."""
    import anndata as ad

    try:
        adata = ad.read_h5ad(ROOT / config["dataset"]["path"])
        return _expression_from_adata(adata, config, ids_by_split, genes)
    except Exception as direct_error:
        work_dir.mkdir(parents=True, exist_ok=True)
        spec_path = work_dir / "extract_spec.json"
        output_path = work_dir / "expression.npz"
        _write_extract_spec(spec_path, config_path, ids_by_split, genes)
        cmd = [
            conda_python("scanvi311"),
            str(Path(__file__).resolve()),
            "--extract-spec",
            str(spec_path),
            "--extract-output",
            str(output_path),
        ]
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Expression extraction failed in both environments. "
                f"Direct error: {direct_error}; subprocess stderr: {completed.stderr}"
            )
        payload = np.load(output_path, allow_pickle=False)
        available = payload["genes"].astype(str).tolist()
        arrays = {split: payload[f"X_{split}"].astype(np.float32) for split in ids_by_split}
        return arrays, available


def _log1p_cp10k(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    totals = array.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(array / totals * 10000.0).astype(np.float32)


def _make_adata(matrix: np.ndarray, cell_ids: list[str], genes: list[str], labels=None):
    import anndata as ad

    obs = pd.DataFrame(index=pd.Index(cell_ids, dtype=str))
    if labels is not None:
        obs["celltype"] = pd.Series(labels, index=obs.index, dtype=str)
    return ad.AnnData(
        X=_log1p_cp10k(matrix),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, dtype=str)),
    )


def _annotation_arrays(result, expected_ids: list[str]) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Validate and align native TOSICA predictions, confidence, and latent."""
    result_ids = result.obs_names.astype(str).tolist()
    if result_ids != list(map(str, expected_ids)):
        raise ValueError("TOSICA result cell order differs from the requested order")
    predictions = result.obs["Prediction"].astype(str).reset_index(drop=True)
    confidence = pd.to_numeric(result.obs["Probability"], errors="coerce").to_numpy(dtype=float)
    latent = result.X.toarray() if sp.issparse(result.X) else np.asarray(result.X)
    latent = np.asarray(latent, dtype=np.float32)
    if latent.shape != (len(expected_ids), TOSICA_EMBED_DIM):
        raise ValueError(f"Expected native TOSICA latent shape (*, 48), received {latent.shape}")
    if not np.isfinite(latent).all() or not np.isfinite(confidence).all():
        raise ValueError("Non-finite TOSICA latent or confidence")
    return predictions, confidence, latent


def _evaluate_pair(
    y_true: pd.Series | np.ndarray,
    baseline: pd.Series | np.ndarray,
    refined: pd.Series | np.ndarray,
    rare_class: str,
) -> dict[str, float | int | bool]:
    y = np.asarray(y_true).astype(str)
    base = np.asarray(baseline).astype(str)
    final = np.asarray(refined).astype(str)
    if not (len(y) == len(base) == len(final)):
        raise ValueError("Prediction and truth lengths differ")
    baseline_metrics, _ = classification_tables(y, base, rare_class=rare_class)
    refined_metrics, _ = classification_tables(y, final, rare_class=rare_class)
    changed_to_rare = (base != final) & (final == rare_class)
    true_rescues = int((changed_to_rare & (y == rare_class)).sum())
    false_rescues = int((changed_to_rare & (y != rare_class)).sum())
    all_rescues = int(changed_to_rare.sum())
    n_nonrare = int((y != rare_class).sum())
    incremental_fpr = false_rescues / n_nonrare if n_nonrare else float("nan")
    rescue_precision = true_rescues / all_rescues if all_rescues else float("nan")
    return {
        "baseline_rare_f1": baseline_metrics["rare_f1"],
        "baseline_rare_recall": baseline_metrics["rare_recall"],
        "baseline_rare_precision": baseline_metrics["rare_precision"],
        "refined_rare_f1": refined_metrics["rare_f1"],
        "refined_rare_recall": refined_metrics["rare_recall"],
        "refined_rare_precision": refined_metrics["rare_precision"],
        "delta_rare_f1": refined_metrics["rare_f1"] - baseline_metrics["rare_f1"],
        "delta_rare_recall": refined_metrics["rare_recall"] - baseline_metrics["rare_recall"],
        "true_rescues": true_rescues,
        "false_rescues": false_rescues,
        "all_rescues": all_rescues,
        "rescue_precision": rescue_precision,
        "rescue_fdp": 1.0 - rescue_precision if all_rescues else float("nan"),
        "incremental_fpr": incremental_fpr,
        "rescue_ffr": incremental_fpr,
        "alpha_violation": bool(np.isfinite(incremental_fpr) and incremental_fpr > DEFAULT_CONFORMAL_ALPHA),
    }


def _save_split_artifacts(
    output_dir: Path,
    split: str,
    cell_ids: list[str],
    predictions: pd.Series,
    confidence: np.ndarray,
    latent: np.ndarray,
    true_labels: pd.Series,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "cell_id": list(map(str, cell_ids)),
            "true_label": pd.Series(true_labels).astype(str).to_numpy(),
            "tosica_predicted_label": pd.Series(predictions).astype(str).to_numpy(),
            "tosica_confidence": np.asarray(confidence, dtype=float),
        }
    )
    frame.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    np.savez_compressed(
        output_dir / f"{split}_latent.npz",
        cell_id=np.asarray(cell_ids, dtype=str),
        latent=np.asarray(latent, dtype=np.float32),
    )


def _artifacts_complete(run_dir: Path) -> bool:
    output_dir = run_dir / PER_RUN_DIRNAME
    required = [
        *(output_dir / f"{split}_predictions.csv" for split in ("train_labeled", "validation", "test")),
        *(output_dir / f"{split}_latent.npz" for split in ("train_labeled", "validation", "test")),
        output_dir / "refined_test_predictions.csv",
        output_dir / "decision.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _run_one(
    config_path: str,
    seed: int,
    rare_train_size: str,
    *,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import TOSICA

    started = time.perf_counter()
    config = load_config(ROOT / config_path)
    dataset = str(config["dataset"]["name"])
    experiment = config["experiment"]
    rare_class = str(experiment["rare_class"])
    split_mode = str(experiment.get("split_mode", "batch_heldout"))
    parsed_rts = parse_rare_train_size(rare_train_size)
    run_dir = ROOT / make_run_dir(config, split_mode, seed, rare_class, parsed_rts)
    key = _safe_key(dataset, seed, rare_train_size)
    checkpoint_dir = CHECKPOINT_ROOT / key
    per_run_output = run_dir / PER_RUN_DIRNAME
    work_dir = ROOT / "tmp" / "tosica_backbone_rescue" / key
    base = {
        "dataset": dataset,
        "seed": seed,
        "rare_train_size": str(rare_train_size),
        "rare_class": rare_class,
        "split_mode": split_mode,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "backbone": "TOSICA",
        "representation": "native_48d_cls_latent",
        "alpha": DEFAULT_CONFORMAL_ALPHA,
        "tosica_cutoff": TOSICA_PREDICTION_CUTOFF,
    }
    provenance: dict[str, Any] = {"key": key, "inputs": {}, "outputs": {}}
    try:
        if not (run_dir / "manifest.json").exists():
            raise ValueError("Required cache manifest.json is missing")
        if not check_manifest(
            run_dir,
            config,
            seed=seed,
            rare_class=rare_class,
            rare_train_size=parsed_rts,
            label_column=config["dataset"]["label_key"],
            batch_key=config["dataset"]["batch_key"],
            split_mode=split_mode,
            validate_split_hash=True,
        ):
            raise ValueError("Cache manifest validation failed")

        frames = {split: _load_split(run_dir, split) for split in ("train", "validation", "test")}
        _validate_splits(frames)
        labeled_mask = _normalize_bool(frames["train"]["is_labeled_for_scanvi"])
        labeled_train = frames["train"].loc[labeled_mask].reset_index(drop=True)
        if int(labeled_train["true_label"].astype(str).eq(rare_class).sum()) < 1:
            raise ValueError("No labeled rare training cell")

        hvg_path = run_dir / "selected_hvg_genes.csv"
        genes = pd.read_csv(hvg_path)["gene"].astype(str).tolist()
        ids_by_split = {
            "train_labeled": labeled_train["cell_id"].astype(str).tolist(),
            "validation": frames["validation"]["cell_id"].astype(str).tolist(),
            "test": frames["test"]["cell_id"].astype(str).tolist(),
        }
        arrays, available_genes = _extract_expression(
            config_path, config, ids_by_split, genes, work_dir
        )

        if force:
            for target in (checkpoint_dir, per_run_output):
                if target.exists():
                    shutil.rmtree(target)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        project = str(checkpoint_dir.relative_to(ROOT)).replace("\\", "/")
        weights = sorted(checkpoint_dir.glob("model-*.pth"))
        if not weights:
            reference = _make_adata(
                arrays["train_labeled"],
                ids_by_split["train_labeled"],
                available_genes,
                labeled_train["true_label"].astype(str).tolist(),
            )
            TOSICA.train(
                reference,
                gmt_path=GMT_MAP[dataset],
                project=project,
                label_name="celltype",
                epochs=TOSICA_EPOCHS,
                max_gs=TOSICA_MAX_GS,
                batch_size=8,
            )
            weights = sorted(checkpoint_dir.glob("model-*.pth"))
            if not weights:
                raise RuntimeError("TOSICA training produced no model checkpoint")
            for intermediate in weights[:-1]:
                intermediate.unlink(missing_ok=True)
            weights = weights[-1:]
        model_weight = str(weights[-1])

        annotations: dict[str, tuple[pd.Series, np.ndarray, np.ndarray]] = {}
        split_labels = {
            "train_labeled": labeled_train["true_label"].astype(str),
            "validation": frames["validation"]["true_label"].astype(str),
            "test": frames["test"]["true_label"].astype(str),
        }
        for split in ("train_labeled", "validation", "test"):
            query = _make_adata(arrays[split], ids_by_split[split], available_genes)
            result = TOSICA.pre(
                query,
                model_weight_path=model_weight,
                project=project,
                laten=True,
                cutoff=TOSICA_PREDICTION_CUTOFF,
                embed_dim=TOSICA_EMBED_DIM,
            )
            annotations[split] = _annotation_arrays(result, ids_by_split[split])

        train_prediction, train_confidence, train_latent = annotations["train_labeled"]
        val_prediction, val_confidence, val_latent = annotations["validation"]
        test_prediction, test_confidence, test_latent = annotations["test"]

        proto = PrototypeRescuer(rare_class)
        proto.fit(
            train_latent,
            labeled_train["true_label"].astype(str),
            np.ones(len(labeled_train), dtype=bool),
        )
        # Test truth is deliberately not passed to, or read by, the rescue path.
        refined_test, decision = conformal_rescue(
            proto,
            test_prediction,
            val_prediction,
            frames["validation"]["true_label"].astype(str),
            val_latent,
            test_latent,
            alpha=DEFAULT_CONFORMAL_ALPHA,
        )
        frozen_test_prediction = test_prediction.astype(str).copy()
        frozen_refined_test = refined_test.astype(str).copy()
        test_true = frames["test"]["true_label"].astype(str).reset_index(drop=True)
        metrics = _evaluate_pair(test_true, frozen_test_prediction, frozen_refined_test, rare_class)

        _save_split_artifacts(
            per_run_output,
            "train_labeled",
            ids_by_split["train_labeled"],
            train_prediction,
            train_confidence,
            train_latent,
            split_labels["train_labeled"],
        )
        _save_split_artifacts(
            per_run_output,
            "validation",
            ids_by_split["validation"],
            val_prediction,
            val_confidence,
            val_latent,
            split_labels["validation"],
        )
        _save_split_artifacts(
            per_run_output,
            "test",
            ids_by_split["test"],
            test_prediction,
            test_confidence,
            test_latent,
            split_labels["test"],
        )
        pd.DataFrame(
            {
                "cell_id": ids_by_split["test"],
                "tosica_predicted_label": frozen_test_prediction,
                "refined_predicted_label": frozen_refined_test,
            }
        ).to_csv(per_run_output / "refined_test_predictions.csv", index=False)
        (per_run_output / "decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )

        row = {
            **base,
            "status": "success",
            "error_reason": "",
            "n_hvg": len(available_genes),
            "n_classes": int(labeled_train["true_label"].astype(str).nunique()),
            "n_labeled_train": len(labeled_train),
            "n_labeled_rare": int(labeled_train["true_label"].astype(str).eq(rare_class).sum()),
            "n_validation": len(frames["validation"]),
            "n_test": len(frames["test"]),
            "n_validation_unknown": int(val_prediction.eq("Unknown").sum()),
            "n_test_unknown": int(test_prediction.eq("Unknown").sum()),
            "separability": float(proto.separability_ratio),
            **metrics,
            "abstain": bool(decision.get("abstain", False)),
            "abstain_reason": str(decision.get("reason", "") or "rescue_applied"),
            "chosen_rank": int(decision.get("chosen_rank", 0)),
            "tau": float(decision.get("tau", np.nan)),
            "val_missed": int(decision.get("val_missed", 0)),
            "n_candidate": int(decision.get("n_candidate", 0)),
            "wall_time_seconds": time.perf_counter() - started,
            "labeled_train_id_sha256": _sha256_values(ids_by_split["train_labeled"]),
            "hvg_sha256": _sha256_values(available_genes),
        }
        input_paths = [
            run_dir / "manifest.json",
            hvg_path,
            *(run_dir / "embeddings" / f"{split}_predictions.csv" for split in frames),
        ]
        provenance["inputs"] = {
            str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths
        }
        output_paths = [
            *(per_run_output / f"{split}_predictions.csv" for split in ("train_labeled", "validation", "test")),
            *(per_run_output / f"{split}_latent.npz" for split in ("train_labeled", "validation", "test")),
            per_run_output / "refined_test_predictions.csv",
            per_run_output / "decision.json",
            *sorted(path for path in checkpoint_dir.rglob("*") if path.is_file()),
        ]
        provenance["outputs"] = {
            str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths
        }
        return row, provenance
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "error_reason": f"{type(exc).__name__}: {exc}",
            "wall_time_seconds": time.perf_counter() - started,
        }, provenance
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    successful = runs[runs["status"].eq("success")]
    for dataset, frame in successful.groupby("dataset", sort=False):
        delta = frame["delta_rare_f1"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "n_runs": len(frame),
                "baseline_rare_f1_mean": frame["baseline_rare_f1"].mean(),
                "refined_rare_f1_mean": frame["refined_rare_f1"].mean(),
                "delta_rare_f1_mean": delta.mean(),
                "baseline_rare_recall_mean": frame["baseline_rare_recall"].mean(),
                "refined_rare_recall_mean": frame["refined_rare_recall"].mean(),
                "delta_rare_recall_mean": frame["delta_rare_recall"].mean(),
                "incremental_fpr_max": frame["incremental_fpr"].max(),
                "n_alpha_violations": int(frame["alpha_violation"].sum()),
                "n_abstentions": int(frame["abstain"].sum()),
                "wins_ties_losses_f1": (
                    f"{int((delta > 1e-12).sum())}/"
                    f"{int(np.isclose(delta, 0.0, atol=1e-12).sum())}/"
                    f"{int((delta < -1e-12).sum())}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": "unknown", "dirty": None}


def _append_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "tosica_backbone_rescue_v1.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _write_final_outputs(
    runs: pd.DataFrame,
    requested_grid: list[tuple[str, int, str]],
    provenance: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_path = OUT / "run_level.csv"
    summary_path = OUT / "summary.csv"
    notes_path = OUT / "analysis_notes.md"
    manifest_path = OUT / "manifest.json"
    runs.to_csv(run_path, index=False)
    summary = summarize_runs(runs)
    summary.to_csv(summary_path, index=False)

    successes = runs[runs["status"].eq("success")]
    failures = runs[runs["status"].eq("failed")]
    wins = int((successes.get("delta_rare_f1", pd.Series(dtype=float)) > 1e-12).sum())
    ties = int(np.isclose(successes.get("delta_rare_f1", pd.Series(dtype=float)), 0.0, atol=1e-12).sum())
    losses = int((successes.get("delta_rare_f1", pd.Series(dtype=float)) < -1e-12).sum())
    notes_path.write_text(
        "# TOSICA-backbone rescue notes\n\n"
        f"- Closed ledger successful runs: {len(successes)}/{len(runs)}; failures: {len(failures)}.\n"
        f"- Runs requested in this invocation: {len(requested_grid)}.\n"
        "- TOSICA was trained only on labeled training cells and supplied its native 48-dimensional CLS latent.\n"
        "- No scANVI latent, probability, or predicted label entered training, prototypes, rank selection, tau calibration, or test rescue.\n"
        f"- Rare-F1 wins/ties/losses after fixed rescue: {wins}/{ties}/{losses}.\n"
        f"- Empirical alpha violations: {int(successes.get('alpha_violation', pd.Series(dtype=bool)).sum())}.\n"
        "- This is a seed-42, eight-dataset portability screen and does not establish universal backbone independence.\n",
        encoding="utf-8",
    )

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "src" / "rescue.py",
        ROOT / "src" / "utils.py",
        *(ROOT / config for config in args.configs),
    ]
    manifest = {
        "analysis": "tosica_backbone_rescue",
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("TOSICA", "anndata", "numpy", "pandas", "torch")
            },
        },
        "git": _git_state(),
        "parameters": {
            "configs": args.configs,
            "seeds": args.seeds,
            "rare_train_sizes": args.rts,
            "requested_runs": len(requested_grid),
            "ledger_runs": len(runs),
            "tosica_epochs": TOSICA_EPOCHS,
            "tosica_max_gene_sets": TOSICA_MAX_GS,
            "tosica_embedding_dim": TOSICA_EMBED_DIM,
            "tosica_internal_seed": TOSICA_INTERNAL_SEED,
            "tosica_prediction_cutoff": TOSICA_PREDICTION_CUTOFF,
            "alpha": DEFAULT_CONFORMAL_ALPHA,
            "low_sep": CONFORMAL_LOW_SEP,
            "rank_grid": list(CONFORMAL_RANK_GRID),
            "min_val_missed": MIN_VAL_MISSED,
        },
        "test_label_usage": "final paired metrics only; never TOSICA training, prototype fitting, gates, rank, or tau",
        "status_counts": runs["status"].value_counts().to_dict(),
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
        "run_provenance": provenance,
        "outputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (run_path, summary_path, notes_path)
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply unchanged scRareRefine rescue to native TOSICA embeddings."
    )
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--rts", nargs="*", default=list(RARE_TRAIN_SIZES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the closed grid without training")
    parser.add_argument("--extract-spec", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--extract-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.extract_spec is not None:
        if args.extract_output is None:
            raise SystemExit("--extract-output is required with --extract-spec")
        _extract_to_npz(args.extract_spec, args.extract_output)
        return

    grid = build_grid(args.configs, args.seeds, args.rts)
    expected_default = len(CONFIGS) * len(SEEDS) * len(RARE_TRAIN_SIZES)
    if args.configs == list(CONFIGS) and args.seeds == list(SEEDS) and args.rts == list(RARE_TRAIN_SIZES):
        if len(grid) != expected_default or len(grid) != 32:
            raise AssertionError("Default TOSICA portability grid must contain exactly 32 runs")
    if args.dry_run:
        for index, (config_path, seed, rts) in enumerate(grid, start=1):
            dataset = load_config(ROOT / config_path)["dataset"]["name"]
            print(f"{index:02d}\t{dataset}\tseed={seed}\trts={rts}")
        print(f"selected_runs={len(grid)}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    run_path = OUT / "run_level.csv"
    manifest_path = OUT / "manifest.json"
    prior = pd.read_csv(run_path, dtype={"rare_train_size": str}) if run_path.exists() else pd.DataFrame()
    prior_provenance: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            prior_provenance = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("run_provenance", [])
        except (json.JSONDecodeError, OSError):
            prior_provenance = []
    if not prior.empty:
        selected_datasets = {
            str(load_config(ROOT / config)["dataset"]["name"]) for config in args.configs
        }
        selected_keys = {
            (dataset, int(seed), str(rts))
            for dataset in selected_datasets
            for seed in args.seeds
            for rts in args.rts
        }
        keep = ~prior.apply(
            lambda row: (str(row["dataset"]), int(row["seed"]), str(row["rare_train_size"]))
            in selected_keys,
            axis=1,
        )
        untouched = prior.loc[keep].to_dict("records")
        selected_prior = prior.loc[~keep]
    else:
        untouched = []
        selected_prior = pd.DataFrame()

    rows: list[dict[str, Any]] = list(untouched)
    provenance_by_key = {
        str(item.get("key")): item
        for item in prior_provenance
        if item.get("key") is not None
    }
    for config_path, seed, rts in grid:
        config = load_config(ROOT / config_path)
        dataset = str(config["dataset"]["name"])
        rare = str(config["experiment"]["rare_class"])
        run_dir = ROOT / make_run_dir(
            config,
            str(config["experiment"].get("split_mode", "batch_heldout")),
            seed,
            rare,
            parse_rare_train_size(rts),
        )
        previous = selected_prior[
            selected_prior["dataset"].astype(str).eq(dataset)
            & selected_prior["seed"].astype(int).eq(seed)
            & selected_prior["rare_train_size"].astype(str).eq(str(rts))
        ] if not selected_prior.empty else pd.DataFrame()
        provenance_key = _safe_key(dataset, seed, str(rts))
        if (
            not args.force
            and len(previous) == 1
            and previous.iloc[0]["status"] == "success"
            and _artifacts_complete(run_dir)
            and provenance_key in provenance_by_key
        ):
            row = previous.iloc[0].to_dict()
            rows.append(row)
            _append_log(f"{dataset}\t{seed}\t{rts}\tresumed")
            print(f"{dataset}\t{seed}\t{rts}\tresumed", flush=True)
            continue

        row, run_provenance = _run_one(
            config_path, seed, str(rts), force=args.force
        )
        rows.append(row)
        provenance_by_key[str(run_provenance["key"])] = run_provenance
        _append_log(
            f"{dataset}\t{seed}\t{rts}\t{row['status']}\t{row.get('error_reason', '')}"
        )
        current = pd.DataFrame(rows)
        current.to_csv(run_path, index=False)
        print(
            f"{dataset}\t{seed}\t{rts}\t{row['status']}\t{row.get('error_reason', '')}",
            flush=True,
        )

    runs = pd.DataFrame(rows)
    selected_datasets = {
        str(load_config(ROOT / config)["dataset"]["name"]) for config in args.configs
    }
    selected = runs[
        runs["dataset"].astype(str).isin(selected_datasets)
        & runs["seed"].astype(int).isin(args.seeds)
        & runs["rare_train_size"].astype(str).isin(set(map(str, args.rts)))
    ]
    key_columns = ["dataset", "seed", "rare_train_size"]
    if len(selected) != len(grid) or selected.duplicated(key_columns).any():
        raise AssertionError("TOSICA portability ledger does not match the requested closed grid")

    _write_final_outputs(runs, grid, list(provenance_by_key.values()), args)
    failed = selected[~selected["status"].eq("success")]
    if not failed.empty:
        raise SystemExit(f"Fail-closed ledger written; {len(failed)} requested runs failed")
    print(summarize_runs(selected).to_string(index=False))


if __name__ == "__main__":
    main()
