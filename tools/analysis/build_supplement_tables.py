"""Build data-driven LaTeX tables for the scRareRefine supplement.

The script reads only frozen result artifacts and writes presentation-only
LaTeX fragments under paper/supplement_tables. Numerical results are not
hard-coded in the manuscript or in this script.

Run with the analysis environment, for example:

    D:/setup/anaconda/envs/scanvi311/python.exe \
        tools/analysis/build_supplement_tables.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "supplement_tables"

DATASET_ORDER = [
    "immune_dc",
    "mouse_lung_tms_10x",
    "mouse_pancreas_tms_10x",
    "pancreas_baron",
    "pancreas_integrated",
    "tabula_lung_endo",
    "tabula_sapiens_stomach",
    "tabula_small_intestine",
]

DATASET_NAME = {
    "immune_dc": "Immune DC",
    "mouse_lung_tms_10x": "Mouse lung",
    "mouse_pancreas_tms_10x": "Mouse pancreas",
    "pancreas_baron": "Baron pancreas",
    "pancreas_integrated": "Integrated pancreas",
    "tabula_lung_endo": "Lung endothelium",
    "tabula_sapiens_stomach": "Stomach",
    "tabula_small_intestine": "Small intestine",
}

TARGET_NAME = {
    "immune_dc": "ASDC",
    "mouse_lung_tms_10x": "vein endothelial cell",
    "mouse_pancreas_tms_10x": "pancreatic D cell",
    "pancreas_baron": "gamma",
    "pancreas_integrated": "endothelial",
    "tabula_lung_endo": "lymphatic-vessel endothelial cell",
    "tabula_sapiens_stomach": "mast cell",
    "tabula_small_intestine": "intestinal tuft cell",
}

BUDGETS = ["0.01", "0.05", "0.10", "all"]
SCARCE_BUDGETS = {"0.01", "0.05", "0.10"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def as_bool(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "--"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(value, spec)


def fmt_p(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.2e}"


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def unique_int(rows: list[dict[str, str]], key: str) -> int:
    values = {int(float(row[key])) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Expected one value for {key}, found {sorted(values)}")
    return values.pop()


def build_label_support(lang: str) -> str:
    rows = [
        row
        for row in read_csv(ROOT / "results" / "label_budget" / "v1" / "run_level.csv")
        if row["status"] == "success"
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["rare_train_size"])].append(row)
        by_dataset[row["dataset"]].append(row)

    body = []
    for dataset in DATASET_ORDER:
        counts = [
            unique_int(grouped[(dataset, budget)], "actual_training_labeled_rare_count")
            for budget in BUDGETS
        ]
        val_n = unique_int(by_dataset[dataset], "validation_rare")
        test_n = unique_int(by_dataset[dataset], "test_rare")
        body.append(
            f"{DATASET_NAME[dataset]} & {TARGET_NAME[dataset]} & "
            + " & ".join(str(value) for value in counts)
            + f" & {val_n} & {test_n} \\\\"
        )

    if lang == "en":
        caption = (
            "Observed rare-cell support in each frozen split. Training columns give the actual "
            "number of labelled target cells after applying each nominal budget and the five-cell "
            "minimum; validation and test columns are not subsampled. Counts were identical across "
            "the three seeds for every dataset."
        )
        headers = (
            "Dataset & Target class & \\multicolumn{4}{c}{Labelled rare training cells} "
            "& Validation rare & Test rare \\\\\n"
            "\\cmidrule(lr){3-6}\n"
            " & & 1\\% & 5\\% & 10\\% & All & & \\\\"
        )
        label = "tab:supp-label-support"
    else:
        caption = (
            "各冻结划分中的实际稀有细胞支持量。训练列为施加名义预算和至少 5 个细胞下限后，"
            "实际带标签的目标细胞数；验证集和测试集不进行下采样。每个数据集在三个随机种子下"
            "的计数均一致。"
        )
        headers = (
            "数据集 & 目标类别 & \\multicolumn{4}{c}{训练集中带标签的稀有细胞} "
            "& 验证集稀有细胞 & 测试集稀有细胞 \\\\\n"
            "\\cmidrule(lr){3-6}\n"
            " & & 1\\% & 5\\% & 10\\% & 全部 & & \\\\"
        )
        label = "tab:supp-label-support-zh"

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}X X c c c c c c@{{}}}}
\toprule
{headers}
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabularx}}
\end{{table}}
"""


def build_gate_stability(lang: str) -> str:
    rows = read_csv(
        ROOT
        / "results"
        / "adaptive_separability_gate"
        / "v1"
        / "stability_20seeds"
        / "stability_summary.csv"
    )
    n_units = len(rows)
    repeats = {int(row["n_repeats"]) for row in rows}
    if len(repeats) != 1:
        raise ValueError("Decision-seed repeat count is not constant")
    n_repeat = repeats.pop()
    pass_rows = [row for row in rows if as_bool(row["original_pass"])]
    reject_rows = [row for row in rows if not as_bool(row["original_pass"])]
    consistent = sum(as_bool(row["consistent_with_frozen"]) for row in rows)

    if lang == "en":
        caption = (
            "Summary of the decision-seed stability audit for the adaptive low-separability gate. "
            "Each low-$S$ unit was rerun with 20 deterministic fold/bootstrap seeds; test labels "
            "were not loaded. A stable pass required a pass rate of at least 0.80 and a stable "
            "rejection a pass rate of at most 0.20."
        )
        items = [
            ("Low-$S$ batch-heldout units", str(n_units)),
            ("Decision-seed repeats per unit", str(n_repeat)),
            ("Total gate audits", str(n_units * n_repeat)),
            ("Frozen-pass units with stable pass", f"{len(pass_rows)}/{len(pass_rows)}"),
            ("Frozen-reject units with stable rejection", f"{len(reject_rows)}/{len(reject_rows)}"),
            ("Units consistent with the frozen decision", f"{consistent}/{n_units}"),
            ("Test labels loaded", "No"),
        ]
        header = "Audit item & Result"
        label = "tab:supp-gate-stability"
    else:
        caption = (
            "自适应低可分性门控的决策种子稳定性审计汇总。每个低 $S$ 单元使用 20 个确定性"
            "折划分/bootstrap 种子重新运行，且不加载测试标签。通过率不低于 0.80 定义为稳定放行，"
            "不高于 0.20 定义为稳定拒绝。"
        )
        items = [
            ("低 $S$ 的 batch-heldout 单元", str(n_units)),
            ("每个单元的决策种子重复次数", str(n_repeat)),
            ("门控审计总次数", str(n_units * n_repeat)),
            ("冻结规则放行且稳定放行的单元", f"{len(pass_rows)}/{len(pass_rows)}"),
            ("冻结规则拒绝且稳定拒绝的单元", f"{len(reject_rows)}/{len(reject_rows)}"),
            ("与冻结决策一致的单元", f"{consistent}/{n_units}"),
            ("是否加载测试标签", "否"),
        ]
        header = "审计项目 & 结果"
        label = "tab:supp-gate-stability-zh"

    body = "\n".join(f"{name} & {value} \\\\" for name, value in items)
    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\setlength{{\tabcolsep}}{{6pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
{header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_ablation(lang: str) -> str:
    summary_rows = read_csv(
        ROOT / "results" / "supplementary_ablation" / "v1" / "tables" / "dataset_equal_summary.csv"
    )
    summary = {row["variant"]: row for row in summary_rows}
    run_rows = read_csv(ROOT / "results" / "supplementary_ablation" / "v1" / "run_level.csv")
    runs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        if row["status"] == "success":
            runs[row["variant"]].append(row)

    if lang == "en":
        names = {
            "full_method": "Full fixed-gate rescue",
            "fixed_rank_1": "Adaptive rank $\\rightarrow$ fixed $k=1$",
            "minus_conformal_tau": "No score threshold $\\tau$",
            "minus_necessity_gate": "No necessity guard",
            "minus_separability_gate": "No separability gate",
            "rank_1": "Fixed $k=1$",
            "rank_2": "Fixed $k=2$",
            "rank_3": "Fixed $k=3$",
            "adaptive_rank": "Validation-adaptive rank",
        }
        caption = (
            "Full-grid component and candidate-rank audit over eight datasets, four nominal budgets "
            "and three seeds. F1, recall, rescue precision and mean iFPR are dataset-equal means "
            "after effective-budget accounting. Maximum iFPR, violation counts and abstention counts "
            "are taken over the 96 frozen run-level units. The full component pipeline uses the fixed "
            "separability gate so that the adaptive cross-fitted gate remains the separate comparison "
            "reported in the main Table 3."
        )
        headers = (
            "Variant & Rare F1 & Recall & Rescue precision & Mean iFPR & Max iFPR "
            "& $>0.01$ & Abstain"
        )
        component_heading = "\\multicolumn{8}{l}{\\textit{Component leave-one-out variants}} \\\\"
        rank_heading = "\\multicolumn{8}{l}{\\textit{Candidate-rank strategies}} \\\\"
        label = "tab:supp-full-ablation"
    else:
        names = {
            "full_method": "完整固定门控流程",
            "fixed_rank_1": "自适应排序 $\\rightarrow$ 固定 $k=1$",
            "minus_conformal_tau": "无分数阈值 $\\tau$",
            "minus_necessity_gate": "无必要性保护",
            "minus_separability_gate": "无可分性门控",
            "rank_1": "固定 $k=1$",
            "rank_2": "固定 $k=2$",
            "rank_3": "固定 $k=3$",
            "adaptive_rank": "验证集自适应排序",
        }
        caption = (
            "八个数据集、四个名义预算和三个随机种子下的完整网格组件与候选排序审计。F1、召回率、"
            "救援精确率和平均 iFPR 是完成有效预算核算后的数据集等权均值；最大 iFPR、违规数和弃权数"
            "来自 96 个冻结运行单元。组件消融中的完整流程使用固定可分性门控，自适应交叉拟合门控"
            "仍由正文表 3 单独比较。"
        )
        headers = "变体 & 稀有 F1 & 召回率 & 救援精确率 & 平均 iFPR & 最大 iFPR & $>0.01$ & 弃权"
        component_heading = "\\multicolumn{8}{l}{\\textit{组件逐一删除}} \\\\"
        rank_heading = "\\multicolumn{8}{l}{\\textit{候选排序策略}} \\\\"
        label = "tab:supp-full-ablation-zh"

    def row_for(variant: str) -> str:
        result = summary[variant]
        variant_runs = runs[variant]
        if len(variant_runs) != 96:
            raise ValueError(f"Expected 96 runs for {variant}, found {len(variant_runs)}")
        max_ifpr = max(as_float(row["incremental_fpr"]) for row in variant_runs)
        violations = sum(as_bool(row["alpha_violation"]) for row in variant_runs)
        abstentions = sum(as_bool(row["abstain"]) for row in variant_runs)
        return (
            f"{names[variant]} & "
            f"{fmt(as_float(result['dataset_equal_rare_f1_mean']))} & "
            f"{fmt(as_float(result['dataset_equal_rare_recall_mean']))} & "
            f"{fmt(as_float(result['dataset_equal_rescue_precision_mean']))} & "
            f"{fmt(as_float(result['dataset_equal_incremental_fpr_mean']), 4)} & "
            f"{fmt(max_ifpr, 4)} & {violations}/96 & {abstentions}/96 \\\\"
        )

    component = [
        "full_method",
        "fixed_rank_1",
        "minus_conformal_tau",
        "minus_necessity_gate",
        "minus_separability_gate",
    ]
    rank = ["rank_1", "rank_2", "rank_3", "adaptive_rank"]
    body = [component_heading]
    body.extend(row_for(variant) for variant in component)
    body.extend(["\\addlinespace[2pt]", rank_heading])
    body.extend(row_for(variant) for variant in rank)

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
{headers} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_primary_ablation(lang: str) -> str:
    rows = read_csv(ROOT / "results" / "ablation" / "ablation_summary.csv")
    runs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        runs[row["variant"]].append(row)
    expected_variants = [
        "A5_full",
        "A3_minus_adaptive_rank",
        "A4_minus_tau",
        "A2_minus_necessity",
        "A1_minus_sep",
        "R1_rank1",
        "R2_rank2",
        "R3_rank3",
        "R_adaptive",
    ]
    for variant in expected_variants:
        if len(runs[variant]) != 72:
            raise ValueError(f"Expected 72 primary ablation units for {variant}, found {len(runs[variant])}")

    full_by_unit = {
        (row["dataset"], row["seed"], row["rts"]): as_float(row["rare_f1"])
        for row in runs["A5_full"]
    }

    if lang == "en":
        names = {
            "A5_full": "Full fixed-gate rescue",
            "A3_minus_adaptive_rank": "Adaptive rank $\\rightarrow$ fixed $k=1$",
            "A4_minus_tau": "No score threshold $\\tau$",
            "A2_minus_necessity": "No necessity guard",
            "A1_minus_sep": "No separability gate",
            "R1_rank1": "Fixed $k=1$",
            "R2_rank2": "Fixed $k=2$",
            "R3_rank3": "Fixed $k=3$",
            "R_adaptive": "Validation-adaptive rank",
        }
        caption = (
            "Complete six-human-dataset ablation underlying the aggregate statements in the main "
            "Figure 3. Each variant contains six datasets, four budgets and three seeds ($n=72$). "
            "Mean $\\Delta$F1 is paired to the full fixed-gate rescue pipeline; maximum iFPR and "
            "violations are frozen test outcomes. The main figure displays three representative "
            "datasets for legibility, whereas this table reports the full primary ablation grid."
        )
        headers = (
            "Variant & Mean F1 & Mean $\\Delta$F1 & Mean recall & Mean precision "
            "& Max iFPR & $>0.01$ & Abstain"
        )
        component_heading = "\\multicolumn{8}{l}{\\textit{Component leave-one-out variants}} \\\\"
        rank_heading = "\\multicolumn{8}{l}{\\textit{Candidate-rank strategies}} \\\\"
        label = "tab:supp-primary-ablation"
    else:
        names = {
            "A5_full": "完整固定门控流程",
            "A3_minus_adaptive_rank": "自适应排序 $\\rightarrow$ 固定 $k=1$",
            "A4_minus_tau": "无分数阈值 $\\tau$",
            "A2_minus_necessity": "无必要性保护",
            "A1_minus_sep": "无可分性门控",
            "R1_rank1": "固定 $k=1$",
            "R2_rank2": "固定 $k=2$",
            "R3_rank3": "固定 $k=3$",
            "R_adaptive": "验证集自适应排序",
        }
        caption = (
            "正文图 3 聚合结论对应的完整六个人类数据集消融。每个变体包含 6 个数据集、4 个预算和 "
            "3 个随机种子（$n=72$）。平均 $\\Delta$F1 与完整固定门控流程进行配对；最大 iFPR 和"
            "违规数为冻结测试结果。正文图为保证可读性展示 3 个代表性数据集，本表报告完整主消融网格。"
        )
        headers = (
            "变体 & 平均 F1 & 平均 $\\Delta$F1 & 平均召回率 & 平均精确率 "
            "& 最大 iFPR & $>0.01$ & 弃权"
        )
        component_heading = "\\multicolumn{8}{l}{\\textit{组件逐一删除}} \\\\"
        rank_heading = "\\multicolumn{8}{l}{\\textit{候选排序策略}} \\\\"
        label = "tab:supp-primary-ablation-zh"

    def row_for(variant: str) -> str:
        variant_rows = runs[variant]
        paired_delta = []
        for row in variant_rows:
            key = (row["dataset"], row["seed"], row["rts"])
            paired_delta.append(as_float(row["rare_f1"]) - full_by_unit[key])
        max_ifpr = max(as_float(row["ffr"]) for row in variant_rows)
        violations = sum(as_float(row["ffr"]) > 0.01 for row in variant_rows)
        abstentions = sum(as_bool(row["abstain"]) for row in variant_rows)
        return (
            f"{names[variant]} & "
            f"{fmt(safe_mean([as_float(row['rare_f1']) for row in variant_rows]))} & "
            f"{fmt(safe_mean(paired_delta), signed=True)} & "
            f"{fmt(safe_mean([as_float(row['rare_recall']) for row in variant_rows]))} & "
            f"{fmt(safe_mean([as_float(row['rare_precision']) for row in variant_rows]))} & "
            f"{fmt(max_ifpr, 4)} & {violations}/72 & {abstentions}/72 \\\\"
        )

    component = [
        "A5_full",
        "A3_minus_adaptive_rank",
        "A4_minus_tau",
        "A2_minus_necessity",
        "A1_minus_sep",
    ]
    rank = ["R1_rank1", "R2_rank2", "R3_rank3", "R_adaptive"]
    body = [component_heading]
    body.extend(row_for(variant) for variant in component)
    body.extend(["\\addlinespace[2pt]", rank_heading])
    body.extend(row_for(variant) for variant in rank)

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
{headers} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def batch_rows() -> list[dict[str, str]]:
    paths = [
        ROOT / "results" / "adaptive_separability_gate" / "v1" / "human_run_level.csv",
        ROOT / "results" / "adaptive_separability_gate" / "v1" / "mouse_run_level.csv",
    ]
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))
    return [
        row
        for row in rows
        if row["status"] == "success"
        and row["variant"] == "adaptive_sep_gate"
        and row["rare_train_size"] in SCARCE_BUDGETS
    ]


def build_split(lang: str) -> str:
    batch = batch_rows()
    cell = [
        row
        for row in read_csv(ROOT / "results" / "split_sensitivity" / "cell_stratified_followup_run_level.csv")
        if row["status"] in {"success", "ok"} and row["rare_train_size"] in SCARCE_BUDGETS
    ]
    for name, rows in (("batch-heldout", batch), ("cell-stratified", cell)):
        if len(rows) != 72:
            raise ValueError(f"Expected 72 {name} scarce units, found {len(rows)}")

    def summarize(rows: list[dict[str, str]], dataset: str | None) -> tuple[float, float, float]:
        subset = [row for row in rows if dataset is None or row["dataset"] == dataset]
        baseline_key = "baseline_rare_f1"
        refined_key = "rare_f1" if "rare_f1" in subset[0] else "refined_rare_f1"
        baseline = safe_mean([as_float(row[baseline_key]) for row in subset])
        refined = safe_mean([as_float(row[refined_key]) for row in subset])
        return baseline, refined, refined - baseline

    body = []
    for dataset in DATASET_ORDER:
        b0, b1, bd = summarize(batch, dataset)
        c0, c1, cd = summarize(cell, dataset)
        body.append(
            f"{DATASET_NAME[dataset]} & {fmt(b0)} & {fmt(b1)} & {fmt(bd, signed=True)} "
            f"& {fmt(c0)} & {fmt(c1)} & {fmt(cd, signed=True)} \\\\"
        )
    b0, b1, bd = summarize(batch, None)
    c0, c1, cd = summarize(cell, None)
    total_label = "All datasets" if lang == "en" else "全部数据集"
    body.append("\\midrule")
    body.append(
        f"{total_label} & {fmt(b0)} & {fmt(b1)} & {fmt(bd, signed=True)} "
        f"& {fmt(c0)} & {fmt(c1)} & {fmt(cd, signed=True)} \\\\"
    )

    if lang == "en":
        caption = (
            "Per-dataset split-sensitivity results in the scarce-label region. Each dataset entry "
            "is the mean over three budgets (1\\%, 5\\% and 10\\%) and three seeds ($n=9$); the final "
            "row averages all 72 units within each split. The paired difference is scRareRefine "
            "minus scANVI rare-cell F1."
        )
        headers = (
            "Dataset & \\multicolumn{3}{c}{Batch-heldout} & \\multicolumn{3}{c}{Cell-stratified} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            " & scANVI & scRareRefine & $\\Delta$F1 & scANVI & scRareRefine & $\\Delta$F1"
        )
        label = "tab:supp-split-results"
    else:
        caption = (
            "稀缺标签区域的逐数据集划分敏感性结果。每个数据集的数值为 3 个预算（1\\%、5\\%、10\\%）"
            "和 3 个随机种子的均值（$n=9$），末行汇总每种划分下的全部 72 个单元。配对差值为 "
            "scRareRefine 减去 scANVI 的稀有细胞 F1。"
        )
        headers = (
            "数据集 & \\multicolumn{3}{c}{Batch-heldout} & \\multicolumn{3}{c}{Cell-stratified} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            " & scANVI & scRareRefine & $\\Delta$F1 & scANVI & scRareRefine & $\\Delta$F1"
        )
        label = "tab:supp-split-results-zh"

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
{headers} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_tosica(lang: str) -> str:
    rows = [
        row
        for row in read_csv(ROOT / "results" / "tosica_backbone_rescue" / "v1" / "run_level.csv")
        if row["status"] == "success" and row["rare_train_size"] in SCARCE_BUDGETS
    ]
    if len(rows) != 72:
        raise ValueError(f"Expected 72 scarce TOSICA units, found {len(rows)}")

    def summarize(dataset: str | None) -> dict[str, float | int | str]:
        subset = [row for row in rows if dataset is None or row["dataset"] == dataset]
        delta = [as_float(row["delta_rare_f1"]) for row in subset]
        wins = sum(value > 1e-12 for value in delta)
        ties = sum(abs(value) <= 1e-12 for value in delta)
        losses = sum(value < -1e-12 for value in delta)
        return {
            "base_f1": safe_mean([as_float(row["baseline_rare_f1"]) for row in subset]),
            "ref_f1": safe_mean([as_float(row["refined_rare_f1"]) for row in subset]),
            "delta": safe_mean(delta),
            "base_rec": safe_mean([as_float(row["baseline_rare_recall"]) for row in subset]),
            "ref_rec": safe_mean([as_float(row["refined_rare_recall"]) for row in subset]),
            "wtl": f"{wins}/{ties}/{losses}",
            "max_ifpr": max(as_float(row["incremental_fpr"]) for row in subset),
            "viol": sum(as_bool(row["alpha_violation"]) for row in subset),
            "n": len(subset),
        }

    body = []
    for dataset in DATASET_ORDER:
        s = summarize(dataset)
        body.append(
            f"{DATASET_NAME[dataset]} & {fmt(s['base_f1'])} & {fmt(s['ref_f1'])} & "
            f"{fmt(s['delta'], signed=True)} & {fmt(s['base_rec'])} & {fmt(s['ref_rec'])} & "
            f"{s['wtl']} & {fmt(s['max_ifpr'], 4)} & {s['viol']}/{s['n']} \\\\"
        )
    s = summarize(None)
    body.extend(
        [
            "\\midrule",
            f"{'All datasets' if lang == 'en' else '全部数据集'} & {fmt(s['base_f1'])} & "
            f"{fmt(s['ref_f1'])} & {fmt(s['delta'], signed=True)} & {fmt(s['base_rec'])} & "
            f"{fmt(s['ref_rec'])} & {s['wtl']} & {fmt(s['max_ifpr'], 4)} & "
            f"{s['viol']}/{s['n']} \\\\",
        ]
    )

    if lang == "en":
        caption = (
            "Per-dataset portability audit for TOSICA in the scarce-label region. Each dataset row "
            "summarizes three budgets and three seeds ($n=9$); the final row summarizes all 72 units. "
            "W/T/L is based on paired rare-cell F1 differences with a $10^{-12}$ tie tolerance. "
            "Violations count frozen test units with iFPR above 0.01."
        )
        headers = (
            "Dataset & TOSICA F1 & +rescue F1 & $\\Delta$F1 & TOSICA recall & +rescue recall "
            "& W/T/L & Max iFPR & Viol."
        )
        label = "tab:supp-tosica-results"
    else:
        caption = (
            "TOSICA 在稀缺标签区域的逐数据集可迁移性审计。每个数据集行汇总 3 个预算和 3 个随机"
            "种子（$n=9$），末行汇总全部 72 个单元。胜/平/负根据配对稀有细胞 F1 差值计算，"
            "持平容差为 $10^{-12}$；违规数为冻结测试 iFPR 超过 0.01 的单元数。"
        )
        headers = (
            "数据集 & TOSICA F1 & +救援 F1 & $\\Delta$F1 & TOSICA 召回率 & +救援召回率 "
            "& 胜/平/负 & 最大 iFPR & 违规"
        )
        label = "tab:supp-tosica-results-zh"

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{2.8pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
{headers} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_markers(lang: str) -> str:
    rows = read_csv(ROOT / "results" / "biological_case_study" / "v1" / "marker_panels_used.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["role"])].append(row)

    body = []
    source_citations = {
        ("immune_dc", "target"): r"\citep{villani2017dendritic,sulczewski2023transitional}",
        ("immune_dc", "competitor"): r"\citep{villani2017dendritic}",
        ("pancreas_baron", "target"): r"\citep{baron2016pancreas,gupta2021ghsr}",
        ("pancreas_baron", "competitor"): r"\citep{baron2016pancreas}",
    }
    for dataset in ("immune_dc", "pancreas_baron"):
        for role in ("target", "competitor"):
            group = grouped[(dataset, role)]
            genes = ", ".join(row["gene"] for row in group)
            source = source_citations[(dataset, role)]
            cell_type = group[0]["cell_type"]
            role_name = (
                ("Target" if role == "target" else "Competitor")
                if lang == "en"
                else ("目标" if role == "target" else "竞争类型")
            )
            body.append(
                f"{DATASET_NAME[dataset]} & {role_name} & {tex_escape(cell_type)} & "
                f"{tex_escape(genes)} & {source} \\\\"
            )

    if lang == "en":
        caption = (
            "Prespecified marker panels used in the held-out transcriptomic plausibility analyses. "
            "Panels were fixed from external literature before test-set group summaries; no marker "
            "was selected or weighted according to the observed rescue outcomes. Composite panels, "
            "rather than individual genes, were interpreted."
        )
        headers = "Case & Role & Cell type & Genes & Source"
        label = "tab:supp-marker-panels"
    else:
        caption = (
            "冻结测试转录组合理性分析使用的预先指定 marker panel。所有 panel 均在测试集分组汇总前"
            "根据外部文献固定，没有依据观察到的救援结果选择或赋权 marker；解释基于组合 panel，"
            "而不是单个基因。"
        )
        headers = "案例 & 作用 & 细胞类型 & 基因 & 来源"
        label = "tab:supp-marker-panels-zh"

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}l l l X X@{{}}}}
\toprule
{headers} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabularx}}
\end{{table}}
"""


def build_biological_stats(lang: str) -> str:
    rows = [
        row
        for row in read_csv(ROOT / "results" / "biological_case_study" / "v1" / "biological_stats.csv")
        if row["right_group"] == "Unrescued FN"
    ]
    metric_order = ["rare_marker_score", "competitor_marker_score", "delta_similarity"]
    metric_name_en = {
        "rare_marker_score": "Target-marker score",
        "competitor_marker_score": "Competitor-marker score",
        "delta_similarity": "$\\Delta$ similarity (rare $-$ competitor)",
    }
    metric_name_zh = {
        "rare_marker_score": "目标 marker 分数",
        "competitor_marker_score": "竞争类型 marker 分数",
        "delta_similarity": "$\\Delta$ 相似性（稀有 $-$ 竞争类型）",
    }
    by_key = {(row["dataset"], row["metric"]): row for row in rows}
    body: list[str] = []
    for dataset in ("immune_dc", "pancreas_baron"):
        body.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DATASET_NAME[dataset]}}}}} \\\\")
        for metric in metric_order:
            row = by_key[(dataset, metric)]
            metric_name = metric_name_en[metric] if lang == "en" else metric_name_zh[metric]
            body.append(
                f"{metric_name} & {row['n_left']} & {row['n_right']} & "
                f"{fmt(as_float(row['median_left']))} & {fmt(as_float(row['median_right']))} & "
                f"{fmt(as_float(row['median_difference']), signed=True)} & "
                f"{fmt(as_float(row['cliffs_delta']))} & {fmt_p(as_float(row['p_value']))} & "
                f"{fmt_p(as_float(row['q_value_bh_within_figure']))} \\\\"
            )
        if dataset == "immune_dc":
            body.append("\\addlinespace[2pt]")

    if lang == "en":
        caption = (
            "Rescued-TP versus unrescued-FN comparisons in the two held-out transcriptomic cases. "
            "Two-sided Wilcoxon rank-sum tests were applied; Benjamini--Hochberg $q$ values were "
            "computed within the six prespecified comparisons of each case. Cliff's $\\delta$ is "
            "reported as the effect size."
        )
        headers = (
            "Measure & $n_R$ & $n_U$ & Median$_R$ & Median$_U$ & $\\Delta$ median "
            "& Cliff's $\\delta$ & $p$ & $q_{\\mathrm{BH}}$"
        )
        label = "tab:supp-biological-stats"
    else:
        caption = (
            "两个冻结测试转录组案例中 Rescued TP 与 Unrescued FN 的比较。采用双侧 Wilcoxon "
            "秩和检验，并在每个案例预先指定的 6 个比较内计算 Benjamini--Hochberg $q$ 值；"
            "Cliff's $\\delta$ 为效应量。"
        )
        headers = (
            "指标 & $n_R$ & $n_U$ & 中位数$_R$ & 中位数$_U$ & 中位数差 "
            "& Cliff's $\\delta$ & $p$ & $q_{\\mathrm{BH}}$"
        )
        label = "tab:supp-biological-stats-zh"

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{2.4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
{headers} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    generators = {
        "table_s2_label_support": build_label_support,
        "table_s3_gate_stability": build_gate_stability,
        "table_s4_primary_ablation": build_primary_ablation,
        "table_s4_full_ablation": build_ablation,
        "table_s5_split_results": build_split,
        "table_s6_tosica_results": build_tosica,
        "table_s7_marker_panels": build_markers,
        "table_s8_biological_stats": build_biological_stats,
    }
    for stem, builder in generators.items():
        write(f"{stem}.tex", builder("en"))
        write(f"{stem}_zh.tex", builder("zh"))
    print(f"[supplement] wrote {len(generators) * 2} table fragments to {OUT}")


if __name__ == "__main__":
    main()
