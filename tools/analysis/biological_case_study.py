"""Held-out biological plausibility analysis for the scRareRefine rescue.

This script is deliberately downstream of the frozen benchmark.  It replays the
validation-calibrated rescue rule from cached train/validation/test predictions,
uses test labels only to define reporting groups after prediction, and then
computes expression-based evidence that is independent of the scANVI latent
space.  No expression-derived quantity is fed back into model selection.

Run with the project scientific Python environment, for example::

    D:\\setup\\anaconda\\envs\\sandbox310\\python.exe \\
        tools\\analysis\\biological_case_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rescue import PrototypeRescuer, conformal_rescue  # noqa: E402


OUT = ROOT / "results" / "biological_case_study" / "v1"
ADAPTIVE_RUN_LEVEL = ROOT / "results" / "adaptive_separability_gate" / "v1" / "human_run_level.csv"
MARKER_PANEL_PATH = ROOT / "data" / "marker_panels" / "figure4_marker_panels.csv"

SEED = 42
BUDGET = "0.05"
PRIMARY_CASE = "immune_dc"
MIN_TEST_RARE = 20
MIN_BASELINE_MISSED = 5
MIN_TRUE_RESCUED = 5
N_HVG = 1500
N_PROFILE_GENES = 1500

DATA_SOURCES = {
    "immune_dc": {
        "path": ROOT / "data" / "raw" / "human_immune_health" / "human_immune_health_atlas_dc.h5ad",
        "use_raw": True,
        "display": "Immune DC",
    },
    "pancreas_baron": {
        "path": ROOT / "data" / "raw" / "pancreas_baron" / "pancreas_baron.h5ad",
        "use_raw": False,
        "display": "Baron pancreas",
    },
}


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values],
        dtype=object,
    )


def _latent(frame: pd.DataFrame) -> np.ndarray:
    columns = [column for column in frame.columns if column.startswith("latent_")]
    if not columns:
        raise ValueError("latent cache has no latent_* columns")
    return frame[columns].to_numpy(dtype=float)


def _budget_key(value: object) -> str:
    text = str(value).strip().lower()
    if text == "all":
        return text
    return f"{float(text):.2f}"


def _run_dir(dataset: str, rare: str) -> Path:
    safe = rare.replace("+", "pos").replace(" ", "_").replace("/", "_").lower()
    return ROOT / "outputs" / dataset / f"batch_heldout_seed{SEED}_{safe}_rare5pct"


def _load_predictions(dataset: str, rare: str) -> dict[str, pd.DataFrame]:
    run_dir = _run_dir(dataset, rare)
    frames = {
        split: pd.read_csv(run_dir / "embeddings" / f"{split}_predictions.csv")
        for split in ("train", "validation", "test")
    }
    latents = {
        split: pd.read_csv(run_dir / "embeddings" / f"{split}_latent.csv")
        for split in ("train", "validation", "test")
    }
    for split in frames:
        if len(frames[split]) != len(latents[split]):
            raise ValueError(f"prediction/latent length mismatch for {dataset}/{split}")
        if not frames[split]["cell_id"].astype(str).equals(latents[split]["cell_id"].astype(str)):
            raise ValueError(f"prediction/latent cell order mismatch for {dataset}/{split}")
    id_sets = {split: set(frame["cell_id"].astype(str)) for split, frame in frames.items()}
    if id_sets["train"] & id_sets["validation"] or id_sets["train"] & id_sets["test"] or id_sets["validation"] & id_sets["test"]:
        raise ValueError(f"split cell overlap for {dataset}")
    return {**frames, **{f"{split}_latent": latents[split] for split in latents}}


def _replay(dataset: str, rare: str) -> tuple[pd.DataFrame, dict, dict]:
    cached = _load_predictions(dataset, rare)
    train = cached["train"]
    validation = cached["validation"]
    test = cached["test"]
    proto = PrototypeRescuer(rare)
    proto.fit(
        _latent(cached["train_latent"]),
        train["true_label"].astype(str),
        train["is_labeled_for_scanvi"].astype(bool).to_numpy(),
    )
    baseline = test["predicted_label"].astype(str).reset_index(drop=True)
    final, rescue_summary = conformal_rescue(
        proto,
        baseline,
        validation["predicted_label"].astype(str),
        validation["true_label"].astype(str),
        _latent(cached["validation_latent"]),
        _latent(cached["test_latent"]),
    )
    y = test["true_label"].astype(str).reset_index(drop=True)
    final = final.astype(str).reset_index(drop=True)
    rare_mask = y.eq(rare).to_numpy()
    base_rare = baseline.eq(rare).to_numpy()
    final_rare = final.eq(rare).to_numpy()
    changed = final.ne(baseline).to_numpy()
    if np.any(changed & ~(~base_rare & final_rare)):
        raise AssertionError("rescue produced a label transition other than non-rare -> rare")
    groups = np.full(len(test), "non-target", dtype=object)
    masks = {
        "Baseline TP": rare_mask & base_rare,
        "Rescued TP": rare_mask & ~base_rare & final_rare,
        "Unrescued FN": rare_mask & ~final_rare,
    }
    for name, mask in masks.items():
        groups[mask] = name
    false_rescue = ~rare_mask & ~base_rare & final_rare
    groups[false_rescue] = "Rescue FP"
    comp_counts = baseline[rare_mask & ~base_rare].value_counts()
    if comp_counts.empty:
        raise ValueError(f"no baseline-missed rare cells to define competitor for {dataset}")
    competitor = str(comp_counts.index[0])
    true_rescued = int(masks["Rescued TP"].sum())
    false_rescued = int(false_rescue.sum())
    n_test_rare = int(rare_mask.sum())
    base_missed = int((rare_mask & ~base_rare).sum())
    current = pd.DataFrame(
        {
            "dataset": dataset,
            "cell_id": test["cell_id"].astype(str).to_numpy(),
            "true_label": y.to_numpy(),
            "baseline_label": baseline.to_numpy(),
            "refined_label": final.to_numpy(),
            "primary_group": groups,
            "competitor_type": competitor,
            "is_competitor": y.eq(competitor).to_numpy(),
            "is_baseline_tp": masks["Baseline TP"],
            "is_rescued_tp": masks["Rescued TP"],
            "is_unrescued_fn": masks["Unrescued FN"],
            "is_rescue_fp": false_rescue,
        }
    )
    metrics = {
        "dataset": dataset,
        "rare_class": rare,
        "competitor_type": competitor,
        "separability": float(proto.separability_ratio),
        "n_test": int(len(test)),
        "n_test_rare": n_test_rare,
        "baseline_missed": base_missed,
        "true_rescued": true_rescued,
        "false_rescued": false_rescued,
        "n_rescued": int(changed.sum()),
        "chosen_rank": int(rescue_summary.get("chosen_rank", 0)),
        "tau": float(rescue_summary.get("tau", np.nan)),
        "abstain": bool(rescue_summary.get("abstain", False)),
        "rescue_reason": str(rescue_summary.get("reason", "")),
        "competitor_counts": json.dumps({str(k): int(v) for k, v in comp_counts.items()}),
    }
    return current, metrics, cached


def _candidate_table(marker_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, dict, dict]]]:
    ledger = pd.read_csv(ADAPTIVE_RUN_LEVEL)
    ledger["budget_key"] = ledger["rare_train_size"].map(_budget_key)
    eligible = ledger[
        (ledger["seed"].astype(int) == SEED)
        & (ledger["budget_key"] == BUDGET)
        & (ledger["variant"] == "adaptive_sep_gate")
        & (ledger["status"] == "success")
        & (ledger["gate_mode"] == "fixed_pass")
        & (ledger["adaptive_pass"].fillna(False).astype(bool))
    ].copy()
    replayed: dict[str, tuple[pd.DataFrame, dict, dict]] = {}
    rows = []
    for record in eligible.itertuples(index=False):
        dataset = str(record.dataset)
        rare = str(record.rare_class)
        if dataset not in DATA_SOURCES:
            continue
        if marker_panel[(marker_panel.dataset == dataset) & (marker_panel.rare_class == rare) & (marker_panel.role == "target")].empty:
            continue
        table, metrics, cached = _replay(dataset, rare)
        print(
            f"[replay] {dataset}: n_test_rare={metrics['n_test_rare']} "
            f"baseline_missed={metrics['baseline_missed']} "
            f"true_rescued={metrics['true_rescued']} false_rescued={metrics['false_rescued']}",
            flush=True,
        )
        replayed[dataset] = (table, metrics, cached)
        source_true = float(record.true_rescues)
        source_false = float(record.false_rescues)
        source_total = float(record.n_rescued)
        for field, expected, observed in (
            ("true_rescues", source_true, metrics["true_rescued"]),
            ("false_rescues", source_false, metrics["false_rescued"]),
            ("n_rescued", source_total, metrics["n_rescued"]),
        ):
            if not np.isclose(expected, observed, atol=0, rtol=0):
                raise AssertionError(f"{dataset}: replay {field}={observed} disagrees with adaptive ledger={expected}")
        rows.append(
            {
                **metrics,
                "ledger_gate_mode": str(record.gate_mode),
                "ledger_variant": str(record.variant),
                "ledger_run_dir": str(record.run_dir),
                "target_panel_available": True,
                "eligible_for_secondary_case": (
                    metrics["n_test_rare"] >= MIN_TEST_RARE
                    and metrics["baseline_missed"] >= MIN_BASELINE_MISSED
                    and metrics["true_rescued"] >= MIN_TRUE_RESCUED
                ),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("no replayable human candidates found")
    candidates["selection_rank"] = np.nan
    candidate_mask = candidates["eligible_for_secondary_case"].astype(bool)
    ordered = candidates.loc[candidate_mask].sort_values(["separability", "dataset"]).reset_index()
    candidates.loc[ordered["index"].to_numpy(), "selection_rank"] = np.arange(1, len(ordered) + 1)
    candidates["selected_case"] = False
    if ordered.empty:
        raise RuntimeError("no human dataset satisfies the preregistered second-case criteria")
    second = str(ordered.loc[0, "dataset"])
    candidates.loc[candidates.dataset == PRIMARY_CASE, "selected_case"] = True
    candidates.loc[candidates.dataset == second, "selected_case"] = True
    candidates["selection_rule"] = (
        f"Primary case fixed to {PRIMARY_CASE}; secondary case is the lowest-separability eligible human dataset "
        f"at seed={SEED}, budget={BUDGET}, with n_test_rare>={MIN_TEST_RARE}, "
        f"baseline_missed>={MIN_BASELINE_MISSED}, true_rescued>={MIN_TRUE_RESCUED}; no F1 gain criterion."
    )
    return candidates, replayed


def _h5ad_source(path: Path, use_raw: bool):
    handle = h5py.File(path, "r")
    root = handle["raw"] if use_raw else handle
    x = root["X"]
    if isinstance(x, h5py.Group):
        shape = tuple(int(v) for v in x.attrs["shape"])
    else:
        shape = tuple(int(v) for v in x.shape)
    index_name = handle["obs"].attrs["_index"]
    obs_ids = _decode(handle["obs"][index_name][:])
    var_index_name = root["var"].attrs["_index"]
    genes = _decode(root["var"][var_index_name][:])
    return handle, root, x, shape, obs_ids, genes


def _read_rows(x, shape: tuple[int, int], rows: np.ndarray) -> sparse.csr_matrix:
    rows = np.asarray(rows, dtype=int)
    if isinstance(x, h5py.Dataset):
        return sparse.csr_matrix(np.asarray(x[rows, :], dtype=np.float32))
    encoding = str(x.attrs.get("encoding-type", ""))
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise ValueError(f"unsupported sparse encoding {encoding!r}")
    indptr = np.asarray(x["indptr"][:], dtype=np.int64)
    unique_rows = np.unique(rows)
    blocks = []
    starts = np.flatnonzero(np.r_[True, np.diff(unique_rows) > 1])
    stops = np.r_[starts[1:] - 1, len(unique_rows) - 1]
    for start_pos, stop_pos in zip(starts, stops):
        first = int(unique_rows[start_pos])
        last = int(unique_rows[stop_pos])
        data_start = int(indptr[first])
        data_stop = int(indptr[last + 1])
        data = np.asarray(x["data"][data_start:data_stop], dtype=np.float32)
        indices = np.asarray(x["indices"][data_start:data_stop], dtype=np.int64)
        local_indptr = indptr[first : last + 2] - data_start
        block_shape = (last - first + 1, shape[1])
        if encoding == "csr_matrix":
            block = sparse.csr_matrix((data, indices, local_indptr), shape=block_shape)
        else:
            block = sparse.csc_matrix((data, indices, local_indptr), shape=block_shape).tocsr()
        blocks.append(block)
    stacked = sparse.vstack(blocks, format="csr")
    position = {int(row): i for i, row in enumerate(unique_rows)}
    take = np.asarray([position[int(row)] for row in rows], dtype=int)
    return stacked[take]


def _normalize_log1p(matrix: sparse.spmatrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(target_sum, library_size, out=np.zeros_like(library_size, dtype=float), where=library_size > 0)
    normalized = matrix.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    normalized.eliminate_zeros()
    return normalized


def _top_variable_genes(matrix: sparse.csr_matrix, n_genes: int, exclude: set[int] | None = None) -> np.ndarray:
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    if exclude:
        variance[np.asarray(sorted(exclude), dtype=int)] = -np.inf
    finite = np.flatnonzero(np.isfinite(variance))
    if len(finite) == 0:
        raise ValueError("no finite genes available for expression analysis")
    order = finite[np.argsort(variance[finite])]
    return order[-min(int(n_genes), len(order)) :]


def _cosine_similarity(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = matrix @ reference
    denominator = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(reference)), 1e-12)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u = float(mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic)
    return 2.0 * u / (len(x) * len(y)) - 1.0


def _bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(p[indices])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def _analyze_case(table: pd.DataFrame, metrics: dict, marker_panel: pd.DataFrame, cached: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    dataset = metrics["dataset"]
    rare = metrics["rare_class"]
    competitor = metrics["competitor_type"]
    source = DATA_SOURCES[dataset]
    panel = marker_panel[(marker_panel.dataset == dataset) & (marker_panel.rare_class == rare)].copy()
    print(f"[expression] {dataset}: opening {source['path'].name}", flush=True)
    if set(panel.loc[panel.role == "competitor", "cell_type"].astype(str)) != {competitor}:
        raise ValueError(f"marker panel competitor does not match baseline confusion for {dataset}: {competitor}")
    target_genes = panel.loc[panel.role == "target", "gene"].astype(str).drop_duplicates().tolist()
    competitor_genes = panel.loc[panel.role == "competitor", "gene"].astype(str).drop_duplicates().tolist()

    h5, root, x, shape, obs_ids, genes = _h5ad_source(source["path"], bool(source["use_raw"]))
    try:
        row_lookup = {str(cell_id): i for i, cell_id in enumerate(obs_ids)}
        test_rows = np.asarray([row_lookup[cell_id] for cell_id in table.cell_id.astype(str)], dtype=int)
        train = cached["train"]
        rare_ref_ids = train.loc[
            train["true_label"].astype(str).eq(rare) & train["is_labeled_for_scanvi"].astype(bool), "cell_id"
        ].astype(str).to_numpy()
        competitor_ref_ids = train.loc[
            train["true_label"].astype(str).eq(competitor) & train["is_labeled_for_scanvi"].astype(bool), "cell_id"
        ].astype(str).to_numpy()
        rare_ref_rows = np.asarray([row_lookup[cell_id] for cell_id in rare_ref_ids], dtype=int)
        competitor_ref_rows = np.asarray([row_lookup[cell_id] for cell_id in competitor_ref_ids], dtype=int)
        if len(rare_ref_rows) == 0 or len(competitor_ref_rows) == 0:
            raise ValueError(f"missing train reference cells for {dataset}")
        all_rows = np.concatenate([test_rows, rare_ref_rows, competitor_ref_rows])
        print(
            f"[expression] {dataset}: reading {len(test_rows)} test + "
            f"{len(rare_ref_rows)} rare-reference + {len(competitor_ref_rows)} competitor-reference rows",
            flush=True,
        )
        expression = _normalize_log1p(_read_rows(x, shape, all_rows))
        print(f"[expression] {dataset}: expression rows loaded", flush=True)
    finally:
        h5.close()

    n_test = len(test_rows)
    n_rare_ref = len(rare_ref_rows)
    n_comp_ref = len(competitor_ref_rows)
    test_expression = expression[:n_test]
    rare_expression = expression[n_test : n_test + n_rare_ref]
    competitor_expression = expression[n_test + n_rare_ref :]
    gene_lookup = {str(gene): i for i, gene in enumerate(genes)}
    missing_target = [gene for gene in target_genes if gene not in gene_lookup]
    missing_competitor = [gene for gene in competitor_genes if gene not in gene_lookup]
    if missing_target or missing_competitor:
        raise ValueError(f"missing marker genes for {dataset}: target={missing_target}, competitor={missing_competitor}")
    target_idx = np.asarray([gene_lookup[gene] for gene in target_genes], dtype=int)
    competitor_idx = np.asarray([gene_lookup[gene] for gene in competitor_genes], dtype=int)
    marker_idx = set(target_idx.tolist()) | set(competitor_idx.tolist())

    # The embedding is built from expression of the held-out test cells only.
    # Marker genes are added only for visibility; they do not affect predictions.
    hvg = _top_variable_genes(test_expression, N_HVG)
    embedding_features = np.unique(np.concatenate([hvg, target_idx, competitor_idx]))
    pca_input = test_expression[:, embedding_features].toarray()
    n_components = min(20, pca_input.shape[1], max(2, pca_input.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=SEED)
    pca_coordinates = pca.fit_transform(pca_input)
    # The standard UMAP package in the local environment attempts to import
    # optional parametric components before exposing the ordinary estimator.
    # For this frozen downstream visualization we use the equally independent
    # PCA/t-SNE embedding instead of changing the model or reusing scANVI's
    # latent space.  The figure labels this panel as an expression embedding.
    reducer = TSNE(
        n_components=2,
        perplexity=min(30.0, max(5.0, (n_test - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        max_iter=300,
        random_state=SEED,
        method="barnes_hut",
        angle=0.5,
        n_jobs=1,
    )
    xy = reducer.fit_transform(pca_coordinates).astype(np.float32)
    embedding_method = "log-normalized expression -> HVG PCA -> t-SNE"
    print(f"[embedding] {dataset}: expression embedding complete", flush=True)

    # Marker module scores are test-set z-scored averages.  The z-scoring is
    # unsupervised and uses no group labels.
    target_expression = test_expression[:, target_idx].toarray()
    competitor_expression_test = test_expression[:, competitor_idx].toarray()
    def zscore(values: np.ndarray) -> np.ndarray:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        return (values - mean) / std
    target_score = zscore(target_expression).mean(axis=1)
    competitor_score = zscore(competitor_expression_test).mean(axis=1)

    # Similarity reference profiles are constructed only from labeled training
    # cells.  Profile genes are selected independently of marker genes so that
    # DeltaSim is not just a restatement of the marker score.
    reference_matrix = sparse.vstack([test_expression, rare_expression, competitor_expression], format="csr")
    profile_idx = _top_variable_genes(reference_matrix, N_PROFILE_GENES, exclude=marker_idx)
    profile_expression = reference_matrix[:, profile_idx].toarray()
    test_profile = profile_expression[:n_test]
    rare_profile = profile_expression[n_test : n_test + n_rare_ref].mean(axis=0)
    competitor_profile = profile_expression[n_test + n_rare_ref :].mean(axis=0)
    similarity_rare = _cosine_similarity(test_profile, rare_profile)
    similarity_competitor = _cosine_similarity(test_profile, competitor_profile)
    delta_similarity = similarity_rare - similarity_competitor

    table = table.copy()
    table["umap1"] = xy[:, 0]
    table["umap2"] = xy[:, 1]
    table["rare_marker_score"] = target_score
    table["competitor_marker_score"] = competitor_score
    table["similarity_rare_reference"] = similarity_rare
    table["similarity_competitor_reference"] = similarity_competitor
    table["delta_similarity"] = delta_similarity
    table["rare_reference_n"] = n_rare_ref
    table["competitor_reference_n"] = n_comp_ref

    group_order = ["Baseline TP", "Rescued TP", "Unrescued FN", "Rescue FP", "Competitor"]
    group_masks = {
        "Baseline TP": table["is_baseline_tp"].to_numpy(dtype=bool),
        "Rescued TP": table["is_rescued_tp"].to_numpy(dtype=bool),
        "Unrescued FN": table["is_unrescued_fn"].to_numpy(dtype=bool),
        "Rescue FP": table["is_rescue_fp"].to_numpy(dtype=bool),
        "Competitor": table["is_competitor"].to_numpy(dtype=bool),
    }
    score_rows = []
    dot_rows = []
    marker_sets = [("Target rare markers", target_genes, target_idx), ("Competing-type markers", competitor_genes, competitor_idx)]
    for group in group_order:
        mask = group_masks[group]
        if not mask.any():
            continue
        for cell_index in np.flatnonzero(mask):
            score_rows.append(
                {
                    "dataset": dataset,
                    "cell_id": table.iloc[cell_index]["cell_id"],
                    "group": group,
                    "rare_marker_score": target_score[cell_index],
                    "competitor_marker_score": competitor_score[cell_index],
                    "similarity_rare_reference": similarity_rare[cell_index],
                    "similarity_competitor_reference": similarity_competitor[cell_index],
                    "delta_similarity": delta_similarity[cell_index],
                }
            )
        for family, marker_names, marker_indices in marker_sets:
            values = test_expression[mask][:, marker_indices].toarray()
            for gene_index, gene in enumerate(marker_names):
                gene_values = values[:, gene_index]
                dot_rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "marker_family": family,
                        "gene": gene,
                        "mean_expression": float(gene_values.mean()),
                        "pct_expressed": float((gene_values > 0).mean()),
                        "n_cells": int(mask.sum()),
                    }
                )

    stats_rows = []
    p_values = []
    for metric in ("rare_marker_score", "competitor_marker_score", "delta_similarity"):
        for right_group in ("Competitor", "Unrescued FN"):
            left_mask = group_masks["Rescued TP"]
            right_mask = group_masks[right_group]
            left = table.loc[left_mask, metric].to_numpy(dtype=float)
            right = table.loc[right_mask, metric].to_numpy(dtype=float)
            if len(left) and len(right):
                test_result = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic")
                p_value = float(test_result.pvalue)
                statistic = float(test_result.statistic)
                median_difference = float(np.median(left) - np.median(right))
                effect = _cliffs_delta(left, right)
            else:
                p_value = statistic = median_difference = effect = float("nan")
            p_values.append(p_value)
            stats_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "left_group": "Rescued TP",
                    "right_group": right_group,
                    "n_left": int(len(left)),
                    "n_right": int(len(right)),
                    "median_left": float(np.median(left)) if len(left) else np.nan,
                    "median_right": float(np.median(right)) if len(right) else np.nan,
                    "median_difference": median_difference,
                    "cliffs_delta": effect,
                    "mannwhitney_u": statistic,
                    "p_value": p_value,
                }
            )
    stats = pd.DataFrame(stats_rows)
    stats["q_value_bh_within_figure"] = _bh_fdr(stats["p_value"].to_numpy())

    summary = {
        **metrics,
        "target_markers": ";".join(target_genes),
        "competitor_markers": ";".join(competitor_genes),
        "target_markers_present": len(target_genes),
        "competitor_markers_present": len(competitor_genes),
        "target_reference_n": n_rare_ref,
        "competitor_reference_n": n_comp_ref,
        "embedding_method": embedding_method,
        "embedding_hvg_n": int(len(hvg)),
        "embedding_feature_n": int(len(embedding_features)),
        "similarity_profile_gene_n": int(len(profile_idx)),
        "test_truth_used_for_model_selection": False,
        "test_truth_used_for_group_definition": True,
    }
    embedding = {
        "xy": xy,
        "cell_id": table["cell_id"].astype(str).to_numpy(dtype=object),
        "is_baseline_tp": table["is_baseline_tp"].to_numpy(dtype=bool),
        "is_rescued_tp": table["is_rescued_tp"].to_numpy(dtype=bool),
        "is_unrescued_fn": table["is_unrescued_fn"].to_numpy(dtype=bool),
        "is_rescue_fp": table["is_rescue_fp"].to_numpy(dtype=bool),
        "is_competitor": table["is_competitor"].to_numpy(dtype=bool),
    }
    return table, pd.DataFrame(score_rows), pd.DataFrame(dot_rows), stats, {"summary": summary, "embedding": embedding}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    marker_panel = pd.read_csv(MARKER_PANEL_PATH)
    required_marker_columns = {"dataset", "rare_class", "role", "cell_type", "gene", "source", "source_url"}
    if not required_marker_columns.issubset(marker_panel.columns):
        raise ValueError(f"marker panel is missing columns: {required_marker_columns - set(marker_panel.columns)}")
    candidates, replayed = _candidate_table(marker_panel)
    selected = candidates[candidates["selected_case"]].copy()
    if set(selected.dataset) != {PRIMARY_CASE, "pancreas_baron"}:
        raise AssertionError(f"unexpected selected cases: {selected.dataset.tolist()}")
    selected = selected.sort_values(by="dataset", key=lambda s: s.map({PRIMARY_CASE: 0, "pancreas_baron": 1}))

    all_tables = []
    all_scores = []
    all_dot = []
    all_stats = []
    summaries = []
    embeddings = {}
    for dataset in selected.dataset.tolist():
        table, metrics, cached = replayed[dataset]
        analyzed_table, score_table, dot_table, stats, extra = _analyze_case(table, metrics, marker_panel, cached)
        all_tables.append(analyzed_table)
        all_scores.append(score_table)
        all_dot.append(dot_table)
        all_stats.append(stats)
        summaries.append(extra["summary"])
        embeddings[dataset] = extra["embedding"]

    pd.concat(all_tables, ignore_index=True).to_csv(OUT / "test_cell_groups.csv", index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(OUT / "group_scores.csv", index=False)
    pd.concat(all_dot, ignore_index=True).to_csv(OUT / "marker_dotplot.csv", index=False)
    pd.concat(all_stats, ignore_index=True).to_csv(OUT / "biological_stats.csv", index=False)
    candidates.to_csv(OUT / "case_selection.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "case_summary.csv", index=False)
    marker_panel[marker_panel.dataset.isin(selected.dataset)].to_csv(OUT / "marker_panels_used.csv", index=False)
    for dataset, embedding in embeddings.items():
        np.savez_compressed(OUT / f"embedding_{dataset}.npz", **embedding)

    metadata = {
        "analysis": "held-out biological plausibility case study",
        "seed": SEED,
        "rare_label_budget": BUDGET,
        "primary_case": PRIMARY_CASE,
        "selected_cases": selected.dataset.tolist(),
        "selection_rule": str(candidates.iloc[0]["selection_rule"]),
        "prediction_source": str(ADAPTIVE_RUN_LEVEL.relative_to(ROOT)),
        "marker_panel_source": str(MARKER_PANEL_PATH.relative_to(ROOT)),
        "expression_sources": {dataset: {"path": str(DATA_SOURCES[dataset]["path"].relative_to(ROOT)), "use_raw": DATA_SOURCES[dataset]["use_raw"]} for dataset in selected.dataset},
        "model_replay": "cached train/validation/test predictions and latent embeddings; fixed-pass branch replayed with src.rescue.conformal_rescue",
        "test_labels": "used only after predictions were frozen to define groups, choose the baseline confusion competitor, and report held-out outcomes",
        "independent_embedding": "log-normalized expression -> unsupervised test-cell HVG selection -> PCA -> t-SNE; scANVI latent was not used",
        "reference_profiles": "labeled train rare cells and labeled train competitor cells; no test expression or labels enter reference construction",
        "marker_scores": "test-set z-scored mean expression for preregistered composite marker panels",
        "similarity": "cosine similarity on non-marker variable genes to labeled train reference profiles",
        "claim_boundary": "transcriptomic biological plausibility, not external wet-lab validation",
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT}")
    print(pd.DataFrame(summaries)[["dataset", "rare_class", "competitor_type", "n_test_rare", "baseline_missed", "true_rescued", "false_rescued", "target_reference_n", "competitor_reference_n"]].to_string(index=False))


if __name__ == "__main__":
    main()
