# run_inductive Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `python -m scrare.cli.run_inductive` 默认在现有 CSV 与 baseline artifacts 之外，再稳定产出 root 级资源汇总表与静态聚合图表。

**Architecture:** 在 `src/scrare/workflows/inductive.py` 中为每个 slice 记录 run 级资源摘要，并沿用现有“run 级缓存 + root 级重建”的 stage 模式聚合出 `stages/inductive_methods/resource_summary.csv`。新增 `src/scrare/visualization/inductive.py` 作为纯下游绘图层，只消费 root 级 CSV，并由 workflow 在 stage CSV 重建后统一触发 plot rebuild 到 `stages/inductive_plots/`。

**Tech Stack:** Python 3.10+, pandas, matplotlib, psutil, pytest

---

## File Map

- Modify: `src/scrare/workflows/inductive.py`
  - 为 baseline + method evaluation 外围接入资源监控
  - 写出 run 级 `resource_summary.csv`
  - 重建 root 级 `stages/inductive_methods/resource_summary.csv`
  - 在 `_rebuild_stage_outputs(root)` 之后触发 `_rebuild_plot_outputs(root)`
- Modify: `src/scrare/infra/paths.py`
  - 如有必要补 plot 路径辅助函数，避免 workflow 里手拼目录
- Create: `src/scrare/visualization/inductive.py`
  - 读取 root 级 CSV
  - 校验列
  - 生成 5 张静态 PNG
  - 为空数据生成说明图
- Modify: `tests/workflows/test_inductive_workflow.py`
  - 先写红灯测试锁定资源表写盘与 plot rebuild 调用顺序
- Create: `tests/visualization/test_inductive_plots.py`
  - 锁定绘图输入契约、缺列报错、空数据出图
- Modify: `README.md`
  - 更新 `run_inductive` 输出目录，补 `resource_summary.csv` 与 `stages/inductive_plots/`

---

### Task 1: 为 workflow 增加资源汇总数据链路

**Files:**
- Modify: `src/scrare/workflows/inductive.py`
- Test: `tests/workflows/test_inductive_workflow.py`

- [ ] **Step 1: 写资源汇总的红灯测试**

```python
def test_run_inductive_workflow_writes_run_and_root_resource_summary(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "outputs"

    monkeypatch.setattr(
        inductive,
        "_iter_slices",
        lambda config, args: iter(
            [
                ("ASDC", "batch_heldout", 42, 20, root),
                ("ASDC", "batch_heldout", 43, 20, root),
            ]
        ),
    )
    monkeypatch.setattr(inductive, "_missing_baseline_artifacts", lambda run_dir: [])
    monkeypatch.setattr(inductive, "_load_baseline_bundle", lambda run_dir: {})
    monkeypatch.setattr(
        inductive,
        "_evaluate_method_outputs",
        lambda *args, run_dir: {
            "effect_runs": pd.DataFrame({"run": [run_dir.name], "method_key": ["baseline"], "method": ["scANVI baseline"]}),
            "effect_summary": pd.DataFrame({"run": [run_dir.name], "method": ["scANVI baseline"]}),
            "threshold_curve": pd.DataFrame(),
            "selected_thresholds": pd.DataFrame(),
            "prototype_candidates": pd.DataFrame(),
            "marker_verified_candidates": pd.DataFrame(),
            "fusion_grid": pd.DataFrame(),
        },
    )

    summaries = iter(
        [
            {"wall_time_seconds": 12.5, "peak_rss_mb": 256.0},
            {"wall_time_seconds": 13.0, "peak_rss_mb": 300.0},
        ]
    )

    class DummyMonitor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def summary(self) -> dict[str, float]:
            return next(summaries)

    monkeypatch.setattr(inductive, "ResourceMonitor", DummyMonitor)
    monkeypatch.setattr(inductive, "_rebuild_plot_outputs", lambda root: None)

    inductive.run_inductive_workflow(
        _config(),
        _args(tmp_path, methods="baseline", reuse_baseline_only=False),
    )

    run_one = pd.read_csv(root / "runs" / "batch_heldout_seed_42_rare_20" / "resource_summary.csv")
    run_two = pd.read_csv(root / "runs" / "batch_heldout_seed_43_rare_20" / "resource_summary.csv")
    root_summary = pd.read_csv(root / "stages" / "inductive_methods" / "resource_summary.csv")

    assert run_one.loc[0, "wall_time_seconds"] == pytest.approx(12.5)
    assert run_two.loc[0, "peak_memory_mb"] == pytest.approx(300.0)
    assert set(root_summary["run"]) == {"batch_heldout_seed_42_rare_20", "batch_heldout_seed_43_rare_20"}
```

- [ ] **Step 2: 运行资源测试，确认当前失败**

Run: `PYTHONPATH=src pytest tests/workflows/test_inductive_workflow.py -k resource_summary -v`
Expected: FAIL，提示 `resource_summary.csv` 未生成，或 `_rebuild_plot_outputs` 不存在。

- [ ] **Step 3: 在 workflow 中定义资源表 schema 与写盘辅助函数**

```python
RESOURCE_SUMMARY_FILENAME = "resource_summary.csv"


def _resource_summary_row(
    *,
    run_name: str,
    seed: int,
    rare_train_size: str | int,
    rare_class: str,
    split_mode: str,
    summary: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run": run_name,
                "seed": seed,
                "rare_train_size": str(rare_train_size),
                "rare_class": rare_class,
                "split_mode": split_mode,
                "wall_time_seconds": float(summary["wall_time_seconds"]),
                "peak_memory_mb": float(summary["peak_rss_mb"]),
            }
        ]
    )


def _run_resource_summary_path(run_dir: Path) -> Path:
    return run_dir / RESOURCE_SUMMARY_FILENAME
```

- [ ] **Step 4: 在单个 slice 外围接入 `ResourceMonitor` 并写 run 级资源表**

```python
with ResourceMonitor() as monitor:
    baseline_missing = _missing_baseline_artifacts(run_dir)
    if baseline_missing:
        if reuse_baseline_only:
            _require_existing_baseline(run_dir)
        _run_baseline_slice(
            config,
            args,
            rare_class=rare_class,
            split_mode=split_mode,
            seed=seed,
            rare_train_size=rare_train_size,
            root=root,
            run_dir=run_dir,
        )
    baseline_bundle = _load_baseline_bundle(run_dir)
    outputs = _evaluate_method_outputs(
        config,
        args,
        rare_class=rare_class,
        split_mode=split_mode,
        seed=seed,
        rare_train_size=rare_train_size,
        run_dir=run_dir,
        baseline_bundle=baseline_bundle,
    )

resource_row = _resource_summary_row(
    run_name=run_dir.name,
    seed=seed,
    rare_train_size=rare_train_size,
    rare_class=rare_class,
    split_mode=split_mode,
    summary=monitor.summary(),
)
write_table(resource_row, _run_resource_summary_path(run_dir))
```

- [ ] **Step 5: 增加 root 级资源聚合重建函数**

```python
def _rebuild_resource_summary(root: Path) -> None:
    parts: list[pd.DataFrame] = []
    for run_dir in _run_dirs(root):
        path = _run_resource_summary_path(run_dir)
        if not path.exists():
            continue
        parts.append(read_table(path))
    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    write_table(merged, stage_table_path(root, "inductive_methods", RESOURCE_SUMMARY_FILENAME))
```

- [ ] **Step 6: 在主循环末尾重建 root 级资源表**

```python
selected_outputs = _select_method_rows(outputs, methods)
_write_run_method_outputs(root, selected_outputs, run_dir=run_dir)
_rebuild_stage_outputs(root)
_rebuild_resource_summary(root)
_rebuild_plot_outputs(root)
```

- [ ] **Step 7: 重新运行资源测试，确认转绿**

Run: `PYTHONPATH=src pytest tests/workflows/test_inductive_workflow.py -k resource_summary -v`
Expected: PASS

- [ ] **Step 8: 提交本任务**

```bash
git add tests/workflows/test_inductive_workflow.py src/scrare/workflows/inductive.py
git commit -m "feat: add inductive resource summaries"
```

---

### Task 2: 锁定 workflow 的 plot rebuild 调用顺序

**Files:**
- Modify: `tests/workflows/test_inductive_workflow.py`
- Modify: `src/scrare/workflows/inductive.py`

- [ ] **Step 1: 写调用顺序红灯测试**

```python
def test_run_inductive_workflow_rebuilds_plots_after_stage_and_resource_outputs(monkeypatch, tmp_path: Path) -> None:
    order: list[str] = []
    root = tmp_path / "outputs"

    monkeypatch.setattr(inductive, "_iter_slices", lambda config, args: iter([("ASDC", "batch_heldout", 42, 20, root)]))
    monkeypatch.setattr(inductive, "_missing_baseline_artifacts", lambda run_dir: [])
    monkeypatch.setattr(inductive, "_load_baseline_bundle", lambda run_dir: {})
    monkeypatch.setattr(
        inductive,
        "_evaluate_method_outputs",
        lambda *args, run_dir: {
            "effect_runs": pd.DataFrame({"run": [run_dir.name], "method_key": ["baseline"], "method": ["scANVI baseline"]}),
            "effect_summary": pd.DataFrame({"run": [run_dir.name], "method": ["scANVI baseline"]}),
            "threshold_curve": pd.DataFrame(),
            "selected_thresholds": pd.DataFrame(),
            "prototype_candidates": pd.DataFrame(),
            "marker_verified_candidates": pd.DataFrame(),
            "fusion_grid": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(inductive, "_write_run_method_outputs", lambda *args, **kwargs: order.append("write"))
    monkeypatch.setattr(inductive, "_rebuild_stage_outputs", lambda root: order.append("stage"))
    monkeypatch.setattr(inductive, "_rebuild_resource_summary", lambda root: order.append("resource"))
    monkeypatch.setattr(inductive, "_rebuild_plot_outputs", lambda root: order.append("plot"))

    class DummyMonitor:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb) -> None:
            return None
        def summary(self) -> dict[str, float]:
            return {"wall_time_seconds": 1.0, "peak_rss_mb": 2.0}

    monkeypatch.setattr(inductive, "ResourceMonitor", DummyMonitor)

    inductive.run_inductive_workflow(_config(), _args(tmp_path, methods="baseline", reuse_baseline_only=False))

    assert order == ["write", "stage", "resource", "plot"]
```

- [ ] **Step 2: 运行调用顺序测试，确认当前失败**

Run: `PYTHONPATH=src pytest tests/workflows/test_inductive_workflow.py -k rebuilds_plots_after_stage_and_resource_outputs -v`
Expected: FAIL，顺序不匹配或 `_rebuild_plot_outputs` 缺失。

- [ ] **Step 3: 在 workflow 中声明 plot rebuild 辅助函数接口**

```python
from scrare.visualization.inductive import rebuild_inductive_plots


def _rebuild_plot_outputs(root: Path) -> None:
    rebuild_inductive_plots(root)
```

- [ ] **Step 4: 确保主流程顺序为 stage → resource → plot**

```python
_write_run_method_outputs(root, selected_outputs, run_dir=run_dir)
_rebuild_stage_outputs(root)
_rebuild_resource_summary(root)
_rebuild_plot_outputs(root)
```

- [ ] **Step 5: 重新运行调用顺序测试，确认转绿**

Run: `PYTHONPATH=src pytest tests/workflows/test_inductive_workflow.py -k rebuilds_plots_after_stage_and_resource_outputs -v`
Expected: PASS

- [ ] **Step 6: 提交本任务**

```bash
git add tests/workflows/test_inductive_workflow.py src/scrare/workflows/inductive.py
git commit -m "test: lock inductive plot rebuild order"
```

---

### Task 3: 实现绘图模块的输入契约与空图机制

**Files:**
- Create: `src/scrare/visualization/inductive.py`
- Create: `tests/visualization/test_inductive_plots.py`

- [ ] **Step 1: 写绘图模块红灯测试**

```python
from pathlib import Path

import pandas as pd
import pytest

from scrare.visualization.inductive import rebuild_inductive_plots


def test_rebuild_inductive_plots_creates_expected_pngs_for_minimal_inputs(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages" / "inductive_methods"
    stage_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "method": ["scANVI baseline", "fusion"],
            "overall_accuracy_mean": [0.9, 0.92],
            "macro_f1_mean": [0.8, 0.83],
            "rare_precision_mean": [0.7, 0.75],
            "rare_recall_mean": [0.6, 0.7],
            "rare_f1_mean": [0.65, 0.72],
        }
    ).to_csv(stage_dir / "five_method_effect_summary.csv", index=False)
    pd.DataFrame(
        {
            "threshold": [0.1, 0.2],
            "rare_error_recall": [0.2, 0.3],
            "major_to_rare_false_rescue_rate": [0.01, 0.02],
        }
    ).to_csv(stage_dir / "validation_marker_threshold_curve.csv", index=False)
    pd.DataFrame({"selected_marker_threshold": [0.2]}).to_csv(stage_dir / "selected_marker_thresholds.csv", index=False)
    pd.DataFrame(
        {
            "temperature": [0.5, 1.0],
            "alpha_min": [0.3, 0.5],
            "beta": [1.0, 1.0],
            "rare_f1": [0.6, 0.7],
        }
    ).to_csv(stage_dir / "fusion_validation_grid.csv", index=False)
    pd.DataFrame(
        {
            "rare_train_size": [20, 50],
            "split_mode": ["batch_heldout", "batch_heldout"],
            "wall_time_seconds": [10.0, 18.0],
            "peak_memory_mb": [200.0, 260.0],
        }
    ).to_csv(stage_dir / "resource_summary.csv", index=False)

    rebuild_inductive_plots(tmp_path)

    plot_dir = tmp_path / "stages" / "inductive_plots"
    assert (plot_dir / "five_method_metric_summary.png").exists()
    assert (plot_dir / "marker_threshold_curve.png").exists()
    assert (plot_dir / "fusion_validation_heatmap.png").exists()
    assert (plot_dir / "runtime_summary.png").exists()
    assert (plot_dir / "memory_summary.png").exists()


def test_rebuild_inductive_plots_raises_clear_error_for_missing_required_columns(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages" / "inductive_methods"
    stage_dir.mkdir(parents=True)
    pd.DataFrame({"method": ["fusion"]}).to_csv(stage_dir / "five_method_effect_summary.csv", index=False)

    with pytest.raises(ValueError, match="five_method_effect_summary.csv is missing required columns"):
        rebuild_inductive_plots(tmp_path)
```

- [ ] **Step 2: 运行绘图测试，确认当前失败**

Run: `PYTHONPATH=src pytest tests/visualization/test_inductive_plots.py -v`
Expected: FAIL，因为模块还不存在。

- [ ] **Step 3: 建立绘图模块骨架与通用辅助函数**

```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scrare.infra.io import read_table
from scrare.infra.paths import stage_table_path

PLOT_FILENAMES = {
    "five_method_metric_summary": "five_method_metric_summary.png",
    "marker_threshold_curve": "marker_threshold_curve.png",
    "fusion_validation_heatmap": "fusion_validation_heatmap.png",
    "runtime_summary": "runtime_summary.png",
    "memory_summary": "memory_summary.png",
}


def _plot_dir(root: Path) -> Path:
    return root / "stages" / "inductive_plots"


def _save_empty_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _require_columns(frame: pd.DataFrame, required: set[str], csv_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_name} is missing required columns: {', '.join(missing)}")
```

- [ ] **Step 4: 实现 `five_method_metric_summary.png` 与 `marker_threshold_curve.png`**

```python
def _build_metric_summary(root: Path) -> None:
    frame = read_table(stage_table_path(root, "inductive_methods", "five_method_effect_summary.csv"))
    path = _plot_dir(root) / "five_method_metric_summary.png"
    if frame.empty:
        _save_empty_figure(path, "Five-method metric summary", "No aggregated method summary data available.")
        return
    required = {
        "method",
        "overall_accuracy_mean",
        "macro_f1_mean",
        "rare_precision_mean",
        "rare_recall_mean",
        "rare_f1_mean",
    }
    _require_columns(frame, required, "five_method_effect_summary.csv")
    metrics = [
        "overall_accuracy_mean",
        "macro_f1_mean",
        "rare_precision_mean",
        "rare_recall_mean",
        "rare_f1_mean",
    ]
    melted = frame[["method", *metrics]].melt(id_vars="method", var_name="metric", value_name="value")
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=True)
    for ax, metric in zip(axes, metrics):
        subset = melted[melted["metric"].eq(metric)]
        ax.bar(subset["method"], subset["value"])
        ax.set_title(metric.removesuffix("_mean"))
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

```python
def _build_marker_threshold_curve(root: Path) -> None:
    curve = read_table(stage_table_path(root, "inductive_methods", "validation_marker_threshold_curve.csv"))
    selected = read_table(stage_table_path(root, "inductive_methods", "selected_marker_thresholds.csv"))
    path = _plot_dir(root) / "marker_threshold_curve.png"
    if curve.empty:
        _save_empty_figure(path, "Marker threshold curve", "No marker threshold search results available.")
        return
    _require_columns(curve, {"threshold", "rare_error_recall", "major_to_rare_false_rescue_rate"}, "validation_marker_threshold_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["rare_error_recall"], label="rare_error_recall")
    ax.plot(curve["threshold"], curve["major_to_rare_false_rescue_rate"], label="major_to_rare_false_rescue_rate")
    if not selected.empty and "selected_marker_threshold" in selected.columns:
        ax.axvline(selected["selected_marker_threshold"].iloc[0], linestyle="--", color="black", label="selected")
    ax.legend()
    ax.set_xlabel("threshold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 5: 实现 `fusion_validation_heatmap.png`、`runtime_summary.png`、`memory_summary.png` 与总入口**

```python
def rebuild_inductive_plots(root: str | Path) -> None:
    root = Path(root)
    _build_metric_summary(root)
    _build_marker_threshold_curve(root)
    _build_fusion_heatmap(root)
    _build_runtime_summary(root)
    _build_memory_summary(root)
```

```python
def _build_fusion_heatmap(root: Path) -> None:
    frame = read_table(stage_table_path(root, "inductive_methods", "fusion_validation_grid.csv"))
    path = _plot_dir(root) / "fusion_validation_heatmap.png"
    if frame.empty:
        _save_empty_figure(path, "Fusion validation heatmap", "No fusion validation grid data available.")
        return
    _require_columns(frame, {"temperature", "alpha_min", "beta", "rare_f1"}, "fusion_validation_grid.csv")
    beta_value = frame["beta"].mode(dropna=True).iloc[0]
    pivot = frame[frame["beta"].eq(beta_value)].pivot_table(index="alpha_min", columns="temperature", values="rare_f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
    ax.set_xlabel("temperature")
    ax.set_ylabel("alpha_min")
    ax.set_title(f"beta={beta_value}")
    fig.colorbar(im, ax=ax, label="rare_f1")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

```python
def _build_runtime_summary(root: Path) -> None:
    frame = read_table(stage_table_path(root, "inductive_methods", "resource_summary.csv"))
    path = _plot_dir(root) / "runtime_summary.png"
    if frame.empty:
        _save_empty_figure(path, "Runtime summary", "No resource summary data available.")
        return
    _require_columns(frame, {"rare_train_size", "split_mode", "wall_time_seconds"}, "resource_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for split_mode, group in frame.groupby("split_mode", dropna=False):
        ax.plot(group["rare_train_size"].astype(str), group["wall_time_seconds"], marker="o", label=str(split_mode))
    ax.set_xlabel("rare_train_size")
    ax.set_ylabel("wall_time_seconds")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

```python
def _build_memory_summary(root: Path) -> None:
    frame = read_table(stage_table_path(root, "inductive_methods", "resource_summary.csv"))
    path = _plot_dir(root) / "memory_summary.png"
    if frame.empty:
        _save_empty_figure(path, "Memory summary", "No resource summary data available.")
        return
    _require_columns(frame, {"rare_train_size", "split_mode", "peak_memory_mb"}, "resource_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for split_mode, group in frame.groupby("split_mode", dropna=False):
        ax.plot(group["rare_train_size"].astype(str), group["peak_memory_mb"], marker="o", label=str(split_mode))
    ax.set_xlabel("rare_train_size")
    ax.set_ylabel("peak_memory_mb")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 6: 重新运行绘图测试，确认转绿**

Run: `PYTHONPATH=src pytest tests/visualization/test_inductive_plots.py -v`
Expected: PASS

- [ ] **Step 7: 提交本任务**

```bash
git add tests/visualization/test_inductive_plots.py src/scrare/visualization/inductive.py
git commit -m "feat: add inductive plot generation"
```

---

### Task 4: 将绘图模块接回 workflow 并验证目录布局

**Files:**
- Modify: `src/scrare/workflows/inductive.py`
- Modify: `tests/workflows/test_inductive_workflow.py`

- [ ] **Step 1: 写布局回归测试**

```python
def test_rebuild_plot_outputs_writes_pngs_under_stage_plot_directory(monkeypatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    def fake_rebuild(root: Path) -> None:
        calls.append(root)
        plot_dir = root / "stages" / "inductive_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / "runtime_summary.png").write_bytes(b"png")

    monkeypatch.setattr(inductive, "rebuild_inductive_plots", fake_rebuild)

    inductive._rebuild_plot_outputs(tmp_path)

    assert calls == [tmp_path]
    assert (tmp_path / "stages" / "inductive_plots" / "runtime_summary.png").exists()
    assert not (tmp_path / "runs" / "runtime_summary.png").exists()
```

- [ ] **Step 2: 运行布局测试，确认当前失败或缺接口**

Run: `PYTHONPATH=src pytest tests/workflows/test_inductive_workflow.py -k stage_plot_directory -v`
Expected: FAIL

- [ ] **Step 3: 在 workflow 中完成 plot rebuild 封装导入**

```python
from scrare.visualization.inductive import rebuild_inductive_plots


def _rebuild_plot_outputs(root: Path) -> None:
    rebuild_inductive_plots(root)
```

- [ ] **Step 4: 重新运行布局测试，确认转绿**

Run: `PYTHONPATH=src pytest tests/workflows/test_inductive_workflow.py -k stage_plot_directory -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add tests/workflows/test_inductive_workflow.py src/scrare/workflows/inductive.py

git commit -m "test: verify inductive plot output layout"
```

---

### Task 5: 更新 README 并跑定向测试集

**Files:**
- Modify: `README.md`
- Test: `tests/workflows/test_inductive_workflow.py`
- Test: `tests/visualization/test_inductive_plots.py`

- [ ] **Step 1: 更新 README 的主实验输出目录说明**

```md
- `runs/<run>/resource_summary.csv`：该 slice 的总运行时间与峰值内存摘要
- `stages/inductive_methods/resource_summary.csv`：跨 runs 聚合后的资源表
- `stages/inductive_plots/`：主流程默认生成的 root 级聚合静态图
  - `five_method_metric_summary.png`
  - `marker_threshold_curve.png`
  - `fusion_validation_heatmap.png`
  - `runtime_summary.png`
  - `memory_summary.png`
```

把 audit 输出说明从：

```md
审计命令会把结果写到配置中的 `experiment.output_dir` 下，并在对应根目录下补充表格。
```

改成：

```md
审计命令会直接把结果写到配置中的 `experiment.output_dir` 下，不再额外复制一份到其他目录。
```

- [ ] **Step 2: 运行 workflow 与 visualization 定向测试**

Run: `PYTHONPATH=src pytest tests/workflows/test_inductive_workflow.py tests/visualization/test_inductive_plots.py -v`
Expected: PASS

- [ ] **Step 3: 运行全量测试**

Run: `PYTHONPATH=src pytest -v`
Expected: PASS

- [ ] **Step 4: 提交本任务**

```bash
git add README.md tests/workflows/test_inductive_workflow.py tests/visualization/test_inductive_plots.py src/scrare/workflows/inductive.py src/scrare/visualization/inductive.py

git commit -m "docs: document inductive visualization outputs"
```

---

## Self-Review

### Spec coverage

- run 级与 root 级 `resource_summary.csv`：Task 1
- `src/scrare/visualization/inductive.py`：Task 3
- 5 张默认静态图：Task 3
- workflow 自动触发 plot rebuild：Task 2 + Task 4
- `stages/inductive_plots/` 目录稳定：Task 4
- 缺列报错、空数据说明图：Task 3
- README 输出说明更新：Task 5
- 定向测试与全量验证：Task 5

### Placeholder scan

- 无未决占位内容。
- 每个任务都给出了具体文件、测试代码、命令与预期结果。

### Type consistency

- 资源监控统一从 `ResourceMonitor.summary()` 消费 `peak_rss_mb`，写表时落成 `peak_memory_mb`。
- root 级资源表固定写到 `stage_table_path(root, "inductive_methods", "resource_summary.csv")`。
- plot rebuild 统一通过 `rebuild_inductive_plots(root)` 入口触发。
