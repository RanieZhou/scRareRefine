"""P4: CellTypist native-score backbone with unchanged scRareRefine rescue.

The backbone is a custom CellTypist classifier trained only on the labeled
training cells. Its multiclass decision-function matrix is the native
representation used by PrototypeRescuer. Validation selects rank and calibrates
tau; test labels are read only after final predictions have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.parent
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
    load_adata,
    load_config,
    make_run_dir,
    parse_rare_train_size,
)
from tools.analysis.label_budget import normalize_bool, sha256_file  # noqa: E402
from tools.analysis.supplementary_ablation import evaluate_predictions  # noqa: E402


CONFIGS = (
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/tabula_sapiens_stomach.yaml",
)
SEEDS = (42,)
RARE_TRAIN_SIZES = ("0.01", "0.05", "0.10")
OUT = ROOT / "results" / "p4_celltypist_backbone" / "v1"
LOG_DIR = ROOT / "logs" / "p4_celltypist_backbone"


def _sha256_values(values: list[str]) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        return None


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _patch_celltypist_for_sklearn() -> None:
    """Remove the deprecated ``multi_class`` kwarg under sklearn >=1.8."""
    import celltypist
    from sklearn.linear_model import LogisticRegression as OriginalLR

    module = sys.modules["celltypist.train"]

    def patched(indata, labels, C, solver, max_iter, n_jobs, **kwargs):
        kwargs.pop("multi_class", None)
        clf = OriginalLR(
            C=C,
            solver=solver or "lbfgs",
            max_iter=max_iter or 1000,
            n_jobs=n_jobs,
            **kwargs,
        )
        clf.fit(indata, labels)
        return clf

    module._LRClassifier = patched
    _ = celltypist


def _load_split(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / "embeddings" / f"{split}_predictions.csv"
    frame = pd.read_csv(path)
    required = {"cell_id", "true_label"}
    if split == "train":
        required.add("is_labeled_for_scanvi")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["cell_id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate cell_id in {path}")
    return frame


def _expression(
    adata: ad.AnnData, cell_ids: list[str], genes: list[str]
) -> ad.AnnData:
    """Return ordered log1p(CP10K) expression without touching source AnnData."""
    index = pd.Series(np.arange(adata.n_obs), index=adata.obs_names.astype(str))
    if index.index.duplicated().any():
        raise ValueError("AnnData obs_names are not unique")
    missing = [cell for cell in cell_ids if cell not in index.index]
    if missing:
        raise ValueError(f"{len(missing)} cell IDs absent from AnnData")
    available = [gene for gene in genes if gene in adata.var_names]
    if len(available) != len(genes):
        raise ValueError(f"{len(genes) - len(available)} cached HVGs absent from AnnData")
    rows = index.loc[cell_ids].to_numpy(dtype=int)
    sub = adata[rows, available].copy()
    x = sub.X
    if sp.issparse(x):
        x = x.tocsr().astype(np.float32)
        totals = np.asarray(x.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        x = sp.diags(10000.0 / totals) @ x
        x.data = np.log1p(x.data)
    else:
        x = np.asarray(x, dtype=np.float32)
        totals = x.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        x = np.log1p(x / totals * 10000.0)
    return ad.AnnData(X=x, obs=pd.DataFrame(index=cell_ids), var=pd.DataFrame(index=available))


def _annotation_frames(result, expected_classes: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    pred = result.predicted_labels["predicted_labels"].astype(str).reset_index(drop=True)
    scores = result.decision_matrix.copy()
    scores.columns = scores.columns.astype(str)
    if set(scores.columns) != set(expected_classes):
        raise ValueError("CellTypist decision columns do not match trained classes")
    scores = scores.loc[:, expected_classes].reset_index(drop=True).astype(float)
    if len(pred) != len(scores) or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("invalid CellTypist prediction/score matrix")
    return pred, scores


def _fit_and_annotate(
    train: ad.AnnData,
    train_labels: pd.Series,
    validation: ad.AnnData,
    test: ad.AnnData,
) -> tuple[pd.Series, np.ndarray, pd.Series, np.ndarray, pd.Series, np.ndarray, list[str]]:
    import celltypist

    model = celltypist.train(
        train,
        labels=train_labels.astype(str).to_numpy(),
        check_expression=False,
        max_iter=1000,
        n_jobs=-1,
    )
    classes = [str(value) for value in model.cell_types]
    outputs = []
    for query in (train, validation, test):
        result = celltypist.annotate(
            query, model=model, majority_voting=False
        )
        pred, scores = _annotation_frames(result, classes)
        outputs.extend((pred, scores.to_numpy(dtype=float)))
    return (*outputs, classes)


def _run_one(
    config_path: str,
    seed: int,
    rare_train_size: str,
    adata_cache: dict[str, ad.AnnData],
) -> tuple[dict, dict]:
    config = load_config(ROOT / config_path)
    dataset = str(config["dataset"]["name"])
    experiment = config["experiment"]
    rare = str(experiment["rare_class"])
    split_mode = str(experiment.get("split_mode", "batch_heldout"))
    parsed_rts = parse_rare_train_size(rare_train_size)
    run_dir = ROOT / make_run_dir(config, split_mode, seed, rare, parsed_rts)
    base = {
        "dataset": dataset,
        "seed": seed,
        "rare_train_size": rare_train_size,
        "rare_class": rare,
        "split_mode": split_mode,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "backbone": "CellTypist",
        "representation": "native_multiclass_decision_scores",
        "alpha": DEFAULT_CONFORMAL_ALPHA,
    }
    provenance: dict[str, object] = {"run_dir": base["run_dir"], "files": {}}
    started = time.perf_counter()
    try:
        if not check_manifest(
            run_dir,
            config,
            seed=seed,
            rare_class=rare,
            rare_train_size=parsed_rts,
            validate_split_hash=True,
        ):
            raise ValueError("cache manifest validation failed")
        frames = {split: _load_split(run_dir, split) for split in ("train", "validation", "test")}
        id_sets = {split: set(frame["cell_id"].astype(str)) for split, frame in frames.items()}
        if any(
            id_sets[a].intersection(id_sets[b])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError("cell_id overlap across splits")
        labeled = normalize_bool(frames["train"]["is_labeled_for_scanvi"]).to_numpy()
        labeled_train = frames["train"].loc[labeled].reset_index(drop=True)
        if int(labeled_train["true_label"].astype(str).eq(rare).sum()) < 1:
            raise ValueError("no labeled rare training cells")
        genes_path = run_dir / "selected_hvg_genes.csv"
        genes = pd.read_csv(genes_path)["gene"].astype(str).tolist()
        if dataset not in adata_cache:
            source = load_adata(config)
            source.obs_names = source.obs_names.astype(str)
            source.obs_names_make_unique()
            adata_cache[dataset] = source
        source = adata_cache[dataset]
        train_x = _expression(source, labeled_train["cell_id"].astype(str).tolist(), genes)
        val_x = _expression(source, frames["validation"]["cell_id"].astype(str).tolist(), genes)
        test_x = _expression(source, frames["test"]["cell_id"].astype(str).tolist(), genes)
        (
            _train_pred,
            train_scores,
            val_pred,
            val_scores,
            test_pred,
            test_scores,
            classes,
        ) = _fit_and_annotate(
            train_x,
            labeled_train["true_label"].astype(str),
            val_x,
            test_x,
        )
        if rare not in classes:
            raise ValueError("rare class absent from CellTypist classes")
        proto = PrototypeRescuer(rare)
        proto.fit(
            train_scores,
            labeled_train["true_label"].astype(str),
            np.ones(len(labeled_train), dtype=bool),
        )
        final, decision = conformal_rescue(
            proto,
            test_pred,
            val_pred,
            frames["validation"]["true_label"].astype(str),
            val_scores,
            test_scores,
            alpha=DEFAULT_CONFORMAL_ALPHA,
        )
        y_test = frames["test"]["true_label"].astype(str).reset_index(drop=True)
        baseline_metrics = evaluate_predictions(y_test, test_pred, test_pred, rare)
        refined_metrics = evaluate_predictions(y_test, test_pred, final, rare)
        row = {
            **base,
            "status": "success",
            "error_reason": "",
            "n_hvg": len(genes),
            "n_classes": len(classes),
            "n_labeled_train": len(labeled_train),
            "n_labeled_rare": int(labeled_train["true_label"].astype(str).eq(rare).sum()),
            "n_validation": len(val_pred),
            "n_test": len(test_pred),
            "separability": float(proto.separability_ratio),
            "baseline_rare_f1": baseline_metrics["rare_f1"],
            "baseline_rare_recall": baseline_metrics["rare_recall"],
            "baseline_rare_precision": baseline_metrics["rare_precision"],
            "refined_rare_f1": refined_metrics["rare_f1"],
            "refined_rare_recall": refined_metrics["rare_recall"],
            "refined_rare_precision": refined_metrics["rare_precision"],
            "delta_rare_f1": refined_metrics["rare_f1"] - baseline_metrics["rare_f1"],
            "delta_rare_recall": refined_metrics["rare_recall"] - baseline_metrics["rare_recall"],
            "true_rescues": refined_metrics["true_rescues"],
            "false_rescues": refined_metrics["false_rescues"],
            "all_rescues": refined_metrics["all_rescues"],
            "rescue_precision": refined_metrics["rescue_precision"],
            "incremental_fpr": refined_metrics["incremental_fpr"],
            "alpha_violation": refined_metrics["alpha_violation"],
            "abstain": bool(decision.get("abstain", False)),
            "abstain_reason": str(decision.get("reason", "") or "rescue_applied"),
            "chosen_rank": int(decision.get("chosen_rank", 0)),
            "tau": float(decision.get("tau", np.nan)),
            "val_missed": int(decision.get("val_missed", 0)),
            "n_candidate": int(decision.get("n_candidate", 0)),
            "wall_time_seconds": time.perf_counter() - started,
            "labeled_train_id_sha256": _sha256_values(labeled_train["cell_id"].astype(str).tolist()),
            "hvg_sha256": _sha256_values(genes),
        }
        for path in (
            run_dir / "manifest.json",
            genes_path,
            *(run_dir / "embeddings" / f"{split}_predictions.csv" for split in frames),
        ):
            provenance["files"][str(path.relative_to(ROOT))] = sha256_file(path)
        return row, provenance
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "error_reason": f"{type(exc).__name__}: {exc}",
            "wall_time_seconds": time.perf_counter() - started,
        }, provenance


def _summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in runs[runs["status"].eq("success")].groupby("dataset", sort=False):
        rows.append(
            {
                "dataset": dataset,
                "n_runs": len(frame),
                "baseline_rare_f1_mean": frame["baseline_rare_f1"].mean(),
                "refined_rare_f1_mean": frame["refined_rare_f1"].mean(),
                "delta_rare_f1_mean": frame["delta_rare_f1"].mean(),
                "baseline_rare_recall_mean": frame["baseline_rare_recall"].mean(),
                "refined_rare_recall_mean": frame["refined_rare_recall"].mean(),
                "delta_rare_recall_mean": frame["delta_rare_recall"].mean(),
                "incremental_fpr_max": frame["incremental_fpr"].max(),
                "n_alpha_violations": int(frame["alpha_violation"].sum()),
                "n_abstentions": int(frame["abstain"].sum()),
                "wins_ties_losses_f1": f"{int((frame.delta_rare_f1 > 1e-12).sum())}/{int(np.isclose(frame.delta_rare_f1, 0).sum())}/{int((frame.delta_rare_f1 < -1e-12).sum())}",
            }
        )
    return pd.DataFrame(rows)


def _figure(runs: pd.DataFrame, output_dir: Path) -> list[Path]:
    ok = runs[runs["status"].eq("success")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, metric, title in (
        (axes[0], "rare_f1", "Rare-cell F1"),
        (axes[1], "rare_recall", "Rare-cell recall"),
    ):
        for dataset, frame in ok.groupby("dataset", sort=False):
            frame = frame.sort_values("rare_train_size")
            x = np.arange(len(frame))
            ax.plot(x, frame[f"baseline_{metric}"], "o--", alpha=0.65, label=f"{dataset}: CellTypist")
            ax.plot(x, frame[f"refined_{metric}"], "o-", linewidth=2, label=f"{dataset}: + rescue")
        ax.set_xticks(range(len(RARE_TRAIN_SIZES)), RARE_TRAIN_SIZES)
        ax.set_xlabel("Rare training-label fraction")
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("P4: CellTypist native-score backbone with unchanged scRareRefine", x=0.02, ha="left")
    fig.tight_layout()
    png = output_dir / "figures" / "p4_celltypist_backbone.png"
    pdf = output_dir / "figures" / "p4_celltypist_backbone.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--rts", nargs="*", default=list(RARE_TRAIN_SIZES))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true", help="rerun completed ledger keys")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    for path in (output_dir, output_dir / "figures", LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    _patch_celltypist_for_sklearn()
    rows, provenance, logs = [], [], []
    cache: dict[str, ad.AnnData] = {}
    run_path = output_dir / "run_level.csv"
    if run_path.exists() and not args.force:
        prior = pd.read_csv(run_path, dtype={"rare_train_size": str})
        rows = prior[prior["status"].eq("success")].to_dict("records")
        logs.append(f"resume\t{len(rows)} completed runs loaded")
    completed = {
        (str(row["dataset"]), int(row["seed"]), str(row["rare_train_size"]))
        for row in rows
    }
    for config_path in args.configs:
        dataset = str(load_config(ROOT / config_path)["dataset"]["name"])
        for seed in args.seeds:
            for rare_train_size in args.rts:
                if (dataset, seed, str(rare_train_size)) in completed:
                    print(f"{dataset}\t{seed}\t{rare_train_size}\tresumed", flush=True)
                    continue
                row, prov = _run_one(config_path, seed, rare_train_size, cache)
                rows.append(row)
                provenance.append(prov)
                logs.append(
                    f"{row['dataset']}\t{seed}\t{rare_train_size}\t{row['status']}\t{row.get('error_reason', '')}"
                )
                pd.DataFrame(rows).to_csv(run_path, index=False)
                print(logs[-1], flush=True)
    runs = pd.DataFrame(rows)
    expected = len(args.configs) * len(args.seeds) * len(args.rts)
    keys = ["dataset", "seed", "rare_train_size"]
    if len(runs) != expected or runs.duplicated(keys).any():
        raise AssertionError("P4 ledger does not match requested grid")
    log_path = LOG_DIR / "p4_celltypist_backbone_v1.log"
    log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")
    if not runs["status"].eq("success").all():
        raise SystemExit(f"Fail-closed P4 ledger written; {(~runs.status.eq('success')).sum()} runs failed")
    summary = _summary(runs)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    figure_paths = _figure(runs, output_dir)
    notes_path = output_dir / "analysis_notes.md"
    notes_path.write_text(
        "# P4 CellTypist-backbone notes\n\n"
        f"- Completed {len(runs)}/{expected} prespecified runs.\n"
        "- CellTypist was trained only on labeled training cells; its native multiclass decision scores supplied the prototype space.\n"
        "- Rank and conformal tau used validation only; test labels were used only for frozen final metrics.\n"
        f"- F1 wins/ties/losses: {int((runs.delta_rare_f1 > 1e-12).sum())}/{int(np.isclose(runs.delta_rare_f1, 0).sum())}/{int((runs.delta_rare_f1 < -1e-12).sum())}.\n"
        f"- Empirical alpha violations: {int(runs.alpha_violation.sum())}; abstentions: {int(runs.abstain.sum())}.\n"
        "- This is a single-seed, three-dataset portability screen, not evidence of universal backbone independence.\n",
        encoding="utf-8",
    )
    sources = [
        ROOT / "tools" / "analysis" / "p4_celltypist_backbone.py",
        ROOT / "src" / "rescue.py",
        ROOT / "results" / "supplementary_program" / "v1" / "methodology.md",
        *(ROOT / path for path in args.configs),
    ]
    outputs = [run_path, summary_path, notes_path, log_path, *figure_paths]
    manifest = {
        "analysis": "p4_celltypist_backbone",
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: _package_version(name) for name in ("celltypist", "numpy", "pandas", "scikit-learn")},
        },
        "git": {"sha": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(), "dirty": _git_dirty()},
        "parameters": {
            "configs": args.configs,
            "seeds": args.seeds,
            "rare_train_sizes": args.rts,
            "alpha": DEFAULT_CONFORMAL_ALPHA,
            "low_sep": CONFORMAL_LOW_SEP,
            "rank_grid": list(CONFORMAL_RANK_GRID),
            "min_val_missed": MIN_VAL_MISSED,
            "representation": "CellTypist native multiclass decision-function vector",
        },
        "review_exception": "External self-review skipped by explicit user instruction on 2026-07-17.",
        "test_label_usage": "final metrics only; never training, prototype fitting, gates, rank, or tau",
        "expected_runs": expected,
        "status_counts": runs["status"].value_counts().to_dict(),
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        "inputs": {"runs": provenance},
        "outputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in outputs},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
