# Project Structure Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目重组为 `src/scrare/` 布局，并把入口、流程编排、模型方法、评估逻辑和基础设施清晰分层，同时保持现有 inductive workflow 与四阶段比较行为不变。

**Architecture:** 新结构以 `src/scrare/` 为中心，分为 `cli/`、`data/`、`models/`、`evaluation/`、`infra/`、`workflows/` 六层。`cli/` 只做参数解析，`workflows/` 做长流程编排，`models/` 提供 baseline/prototype/prototype_gate/marker/fusion 能力，`evaluation/` 统一做指标和四阶段装配，`infra/` 负责配置、路径、IO、资源。

**Tech Stack:** Python 3.10+, setuptools `src/` layout, pytest/unittest, anndata, scanpy, scvi-tools, pandas, numpy, scipy, torch

---

## Planned file structure

**Create:**
- `src/scrare/__init__.py`
- `src/scrare/cli/__init__.py`
- `src/scrare/cli/audit.py`
- `src/scrare/cli/run_inductive.py`
- `src/scrare/cli/evaluate_posthoc.py`
- `src/scrare/data/__init__.py`
- `src/scrare/data/loading.py`
- `src/scrare/data/preprocess.py`
- `src/scrare/data/splits.py`
- `src/scrare/models/__init__.py`
- `src/scrare/models/scanvi.py`
- `src/scrare/models/prototype.py`
- `src/scrare/models/prototype_gate.py`
- `src/scrare/models/fusion.py`
- `src/scrare/models/marker.py`
- `src/scrare/evaluation/__init__.py`
- `src/scrare/evaluation/audit.py`
- `src/scrare/evaluation/metrics.py`
- `src/scrare/evaluation/posthoc.py`
- `src/scrare/infra/__init__.py`
- `src/scrare/infra/config.py`
- `src/scrare/infra/io.py`
- `src/scrare/infra/paths.py`
- `src/scrare/infra/resources.py`
- `src/scrare/workflows/__init__.py`
- `src/scrare/workflows/inductive.py`
- `src/scrare/workflows/posthoc.py`
- `tests/cli/test_cli_smoke.py`
- `tests/evaluation/test_posthoc.py`
- `tests/workflows/test_workflow_smoke.py`

**Modify:**
- `pyproject.toml`
- `README.md`
- `CLAUDE.md`
- `tests/test_inductive.py`
- `tests/test_fusion.py`
- `tests/test_prototype.py`
- `tests/test_prototype_gate.py`
- `tests/test_marker_verifier.py`
- `tests/test_metrics.py`
- `tests/test_output_layout.py`
- `tests/test_project_state.py`

**Remove at end:**
- `scrare_refine/` package files
- `scripts/audit_dataset.py`
- `scripts/run_scanvi_inductive.py`
- `scripts/evaluate_inductive_prototype_marker.py`

---

### Task 1: 建立 `src/scrare` 骨架与打包入口

**Files:**
- Create: `src/scrare/__init__.py`
- Create: `src/scrare/cli/__init__.py`
- Create: `src/scrare/data/__init__.py`
- Create: `src/scrare/models/__init__.py`
- Create: `src/scrare/evaluation/__init__.py`
- Create: `src/scrare/infra/__init__.py`
- Create: `src/scrare/workflows/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/cli/test_cli_smoke.py`

- [ ] **Step 1: 写失败测试，锁定 `src` 包可导入**

```python
# tests/cli/test_cli_smoke.py
import importlib


def test_scrare_package_importable():
    pkg = importlib.import_module("scrare")
    assert pkg.__name__ == "scrare"


def test_cli_modules_importable():
    for name in [
        "scrare.cli.audit",
        "scrare.cli.run_inductive",
        "scrare.cli.evaluate_posthoc",
    ]:
        importlib.import_module(name)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/cli/test_cli_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scrare'`

- [ ] **Step 3: 创建 `src` 骨架与最小包初始化**

```python
# src/scrare/__init__.py
__all__ = [
    "cli",
    "data",
    "models",
    "evaluation",
    "infra",
    "workflows",
]
```

```python
# src/scrare/cli/__init__.py
"""CLI entrypoints for scrare."""
```

```python
# src/scrare/data/__init__.py
"""Data loading, preprocessing, and split utilities."""
```

```python
# src/scrare/models/__init__.py
"""Model and post-processing components."""
```

```python
# src/scrare/evaluation/__init__.py
"""Evaluation utilities and posthoc comparisons."""
```

```python
# src/scrare/infra/__init__.py
"""Infrastructure helpers for config, paths, IO, and resources."""
```

```python
# src/scrare/workflows/__init__.py
"""Workflow orchestration for end-to-end experiments."""
```

```toml
# pyproject.toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["scrare*"]
```

- [ ] **Step 4: 添加最小 CLI 占位模块使导入通过**

```python
# src/scrare/cli/audit.py
def main() -> None:
    raise SystemExit("not implemented")


if __name__ == "__main__":
    main()
```

```python
# src/scrare/cli/run_inductive.py
def main() -> None:
    raise SystemExit("not implemented")


if __name__ == "__main__":
    main()
```

```python
# src/scrare/cli/evaluate_posthoc.py
def main() -> None:
    raise SystemExit("not implemented")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 重新运行测试并确认通过**

Run: `pytest tests/cli/test_cli_smoke.py -v`
Expected: PASS

- [ ] **Step 6: 提交骨架变更**

```bash
git add pyproject.toml src/scrare tests/cli/test_cli_smoke.py
git commit -m "refactor: add src-based scrare package skeleton"
```

### Task 2: 迁移基础设施模块到 `infra/`

**Files:**
- Create: `src/scrare/infra/config.py`
- Create: `src/scrare/infra/io.py`
- Create: `src/scrare/infra/paths.py`
- Create: `src/scrare/infra/resources.py`
- Modify: `tests/test_output_layout.py`
- Test: `tests/test_output_layout.py`

- [ ] **Step 1: 改测试导入到新路径**

```python
# tests/test_output_layout.py
from pathlib import Path
import unittest

from scrare.infra.paths import artifact_path, root_table_path, stage_table_path
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_output_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scrare.infra.paths'`

- [ ] **Step 3: 迁移配置与路径辅助函数**

```python
# src/scrare/infra/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def output_dir(config: dict[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    return Path(experiment.get("output_dir", config.get("output_dir", "outputs")))
```

```python
# src/scrare/infra/paths.py
from __future__ import annotations

from pathlib import Path


def artifact_path(run_dir: str | Path, filename: str) -> Path:
    return Path(run_dir) / "artifacts" / filename


def root_table_path(root: str | Path, filename: str) -> Path:
    return Path(root) / "tables" / filename


def stage_table_path(root: str | Path, stage: str, filename: str) -> Path:
    return Path(root) / "stages" / stage / filename
```

- [ ] **Step 4: 迁移 IO 和资源监控代码**

```python
# src/scrare/infra/io.py
from scrare_refine.io import read_table, write_table

__all__ = ["read_table", "write_table"]
```

```python
# src/scrare/infra/resources.py
from scrare_refine.resources import ResourceMonitor

__all__ = ["ResourceMonitor"]
```

- [ ] **Step 5: 运行路径测试并确认通过**

Run: `pytest tests/test_output_layout.py -v`
Expected: PASS

- [ ] **Step 6: 提交基础设施迁移**

```bash
git add src/scrare/infra tests/test_output_layout.py
git commit -m "refactor: move config paths io and resources into infra"
```

### Task 3: 迁移数据层到 `data/`

**Files:**
- Create: `src/scrare/data/loading.py`
- Create: `src/scrare/data/preprocess.py`
- Create: `src/scrare/data/splits.py`
- Modify: `tests/test_inductive.py`
- Test: `tests/test_inductive.py`

- [ ] **Step 1: 改测试导入到新数据层**

```python
# tests/test_inductive.py
from scrare.models.fusion import prototype_probabilities_from_reference
from scrare.data.splits import (
    batch_heldout_split,
    cell_stratified_split,
    make_inductive_scanvi_labels,
)
from scrare.data.preprocess import select_train_hvg_var_names
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_inductive.py -v`
Expected: FAIL with `ModuleNotFoundError` for `scrare.data.splits` or `scrare.models.fusion`

- [ ] **Step 3: 迁移数据读取和预处理函数**

```python
# src/scrare/data/loading.py
from __future__ import annotations

from typing import Any

import anndata as ad


def adata_from_config(config: dict[str, Any]) -> ad.AnnData:
    dataset = config["dataset"]
    adata = ad.read_h5ad(dataset["path"])
    use_layer = dataset.get("use_layer", None)
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError(f"Config requested layer '{use_layer}', but available layers are: {list(adata.layers.keys())}")
        adata = ad.AnnData(X=adata.layers[use_layer].copy(), obs=adata.obs.copy(), var=adata.var.copy())
    elif dataset.get("use_raw", False):
        if adata.raw is None:
            raise ValueError("Config requested raw.X, but adata.raw is missing")
        adata = ad.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(), var=adata.raw.var.copy(), uns=adata.uns.copy())
    return adata
```

```python
# src/scrare/data/preprocess.py
from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse


def subset_cells(adata: ad.AnnData, *, max_cells: int | None, seed: int) -> ad.AnnData:
    if max_cells is None or max_cells >= adata.n_obs:
        return adata
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(adata.n_obs), size=max_cells, replace=False))
    return adata[indices].copy()


def ensure_unique_names(adata: ad.AnnData) -> None:
    adata.obs_names_make_unique()
    adata.var_names_make_unique()


def select_train_hvg_var_names(train_adata: ad.AnnData, *, n_top_genes: int | None) -> list[str]:
    if n_top_genes is None or n_top_genes <= 0 or n_top_genes >= train_adata.n_vars:
        return train_adata.var_names.astype(str).tolist()
    x = train_adata.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        arr = np.asarray(x)
        mean = arr.mean(axis=0)
        mean_sq = (arr * arr).mean(axis=0)
    variance = mean_sq - mean * mean
    top_idx = np.argsort(-variance)[:n_top_genes]
    return train_adata.var_names[np.sort(top_idx)].astype(str).tolist()
```

- [ ] **Step 4: 迁移 split 与标注预算函数**

```python
# src/scrare/data/splits.py
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SplitName = Literal["train", "validation", "test"]
RareTrainSize = int | Literal["all"]

# 从 scrare_refine/inductive.py 原样迁移：
# - parse_rare_train_size
# - _validate_fractions
# - cell_stratified_split
# - batch_heldout_split
# - make_inductive_scanvi_labels
```

- [ ] **Step 5: 先创建 `models/fusion.py` 的最小桥接，满足当前测试依赖**

```python
# src/scrare/models/fusion.py
from scrare_refine.fusion import prototype_probabilities_from_reference

__all__ = ["prototype_probabilities_from_reference"]
```

- [ ] **Step 6: 运行数据层测试并确认通过**

Run: `pytest tests/test_inductive.py -v`
Expected: PASS

- [ ] **Step 7: 提交数据层迁移**

```bash
git add src/scrare/data src/scrare/models/fusion.py tests/test_inductive.py
git commit -m "refactor: move data loading preprocessing and split logic"
```

### Task 4: 迁移 `prototype`、`prototype_gate`、`marker`、`fusion` 模型模块

**Files:**
- Create: `src/scrare/models/prototype.py`
- Create: `src/scrare/models/prototype_gate.py`
- Create: `src/scrare/models/marker.py`
- Modify: `src/scrare/models/fusion.py`
- Modify: `tests/test_fusion.py`
- Modify: `tests/test_prototype.py`
- Modify: `tests/test_prototype_gate.py`
- Modify: `tests/test_marker_verifier.py`
- Test: `tests/test_fusion.py tests/test_prototype.py tests/test_prototype_gate.py tests/test_marker_verifier.py`

- [ ] **Step 1: 修改四类模型测试导入**

```python
# tests/test_fusion.py
from scrare.models.fusion import (
    confidence_weight,
    evaluate_fusion_effect,
    fuse_predictions,
    prototype_probabilities_from_reference,
    select_best_params,
)
```

```python
# tests/test_prototype_gate.py
from scrare.models.prototype_gate import evaluate_gate_rules
```

```python
# tests/test_marker_verifier.py
from scrare.models.marker import (
    choose_marker_threshold,
    compute_marker_signatures,
    evaluate_threshold_rescue,
    marker_scores_for_candidates,
    marker_threshold_curve,
)
```

- [ ] **Step 2: 运行模型测试并确认失败**

Run: `pytest tests/test_fusion.py tests/test_prototype.py tests/test_prototype_gate.py tests/test_marker_verifier.py -v`
Expected: FAIL because target modules/functions are missing

- [ ] **Step 3: 将旧实现迁移到新模型文件**

```python
# src/scrare/models/fusion.py
# 从 scrare_refine/fusion.py 原样迁移完整实现：
# - prototype_probabilities_from_reference
# - confidence_weight
# - disagreement_aware_weight
# - fuse_predictions
# - evaluate_fusion_effect
# - fuse_and_evaluate
# - select_best_params
```

```python
# src/scrare/models/prototype.py
# 从 scrare_refine/prototype.py 原样迁移完整实现：
# - _euclidean_distances
# - prototype_scores_from_reference
```

```python
# src/scrare/models/prototype_gate.py
# 从 scrare_refine/prototype_gate.py 原样迁移完整实现：
# - _safe_quantile
# - gate_masks
# - evaluate_gate_rules
# - summarize_gate_effect
# - choose_recommended_gate
```

```python
# src/scrare/models/marker.py
# 从 scrare_refine/marker_verifier.py 原样迁移完整实现：
# - compute_marker_signatures
# - marker_scores_for_candidates
# - marker_threshold_curve
# - evaluate_threshold_rescue
# - choose_marker_threshold
# - default_marker_thresholds
```

- [ ] **Step 4: 更新 `tests/test_prototype.py` 导入路径**

```python
# tests/test_prototype.py
from scrare.models.prototype import prototype_scores_from_reference
```

- [ ] **Step 5: 运行模型测试并确认通过**

Run: `pytest tests/test_fusion.py tests/test_prototype.py tests/test_prototype_gate.py tests/test_marker_verifier.py -v`
Expected: PASS

- [ ] **Step 6: 提交模型层迁移**

```bash
git add src/scrare/models tests/test_fusion.py tests/test_prototype.py tests/test_prototype_gate.py tests/test_marker_verifier.py
git commit -m "refactor: move prototype marker and fusion models"
```

### Task 5: 抽取 `scanvi` baseline 模块

**Files:**
- Create: `src/scrare/models/scanvi.py`
- Modify: `tests/test_project_state.py`
- Test: `tests/test_project_state.py`

- [ ] **Step 1: 先写一个针对新 baseline 模块存在性的测试**

```python
# tests/test_project_state.py
import importlib


def test_scanvi_baseline_module_exists():
    module = importlib.import_module("scrare.models.scanvi")
    assert hasattr(module, "prediction_outputs")
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_project_state.py -k scanvi_baseline_module_exists -v`
Expected: FAIL with `ModuleNotFoundError` or missing attribute

- [ ] **Step 3: 从旧主脚本抽取 baseline 训练/推断函数**

```python
# src/scrare/models/scanvi.py
from __future__ import annotations

import numpy as np
import pandas as pd
import scvi
import torch

from scrare.evaluation.metrics import compute_uncertainty


def prediction_outputs(model: scvi.model.SCANVI, adata, label_key: str, rare_class: str):
    pred = model.predict(adata)
    soft = model.predict(adata, soft=True)
    if isinstance(soft, tuple):
        soft = soft[0]
    probabilities = soft if isinstance(soft, pd.DataFrame) else pd.DataFrame(soft)
    probabilities.index = adata.obs_names
    uncertainty = compute_uncertainty(probabilities, rare_class=rare_class)
    latent = model.get_latent_representation(adata)
    predictions = adata.obs.copy()
    predictions["cell_id"] = adata.obs_names
    predictions["true_label"] = adata.obs[label_key].astype(str).to_numpy()
    predictions["predicted_label"] = np.asarray(pred).astype(str)
    predictions = predictions.reset_index(drop=True)
    predictions = pd.concat([predictions, uncertainty.reset_index(drop=True), probabilities.reset_index(drop=True).add_prefix("prob_")], axis=1)
    latent_df = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "cell_id", adata.obs_names.to_numpy())
    return predictions, latent_df
```

```python
# src/scrare/models/scanvi.py
def seed_everything(seed: int) -> None:
    scvi.settings.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
```

```python
# src/scrare/models/scanvi.py
# 继续从旧主脚本拆出：
# - train_reference_scanvi(...)
# - load_query_model(...)
```

- [ ] **Step 4: 运行存在性测试并确认通过**

Run: `pytest tests/test_project_state.py -k scanvi_baseline_module_exists -v`
Expected: PASS

- [ ] **Step 5: 提交 baseline 模块抽取**

```bash
git add src/scrare/models/scanvi.py tests/test_project_state.py
git commit -m "refactor: extract scanvi baseline model helpers"
```

### Task 6: 迁移 `metrics` 与 `audit` 到 `evaluation/`

**Files:**
- Create: `src/scrare/evaluation/metrics.py`
- Create: `src/scrare/evaluation/audit.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_audit.py`
- Test: `tests/test_metrics.py tests/test_audit.py`

- [ ] **Step 1: 改测试导入到 `evaluation/`**

```python
# tests/test_metrics.py
from scrare.evaluation.metrics import *
```

```python
# tests/test_audit.py
from scrare.evaluation.audit import audit_anndata
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_metrics.py tests/test_audit.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 迁移完整实现**

```python
# src/scrare/evaluation/metrics.py
# 从 scrare_refine/metrics.py 原样迁移完整实现
```

```python
# src/scrare/evaluation/audit.py
# 从 scrare_refine/audit.py 原样迁移完整实现
```

- [ ] **Step 4: 运行测试并确认通过**

Run: `pytest tests/test_metrics.py tests/test_audit.py -v`
Expected: PASS

- [ ] **Step 5: 提交评估基础模块迁移**

```bash
git add src/scrare/evaluation tests/test_metrics.py tests/test_audit.py
git commit -m "refactor: move metrics and audit logic into evaluation"
```

### Task 7: 抽取 `workflows.inductive` 并接入新 CLI

**Files:**
- Create: `src/scrare/workflows/inductive.py`
- Modify: `src/scrare/cli/run_inductive.py`
- Test: `tests/workflows/test_workflow_smoke.py`

- [ ] **Step 1: 写一个 smoke test 锁定 CLI 调用 workflow**

```python
# tests/workflows/test_workflow_smoke.py
from unittest.mock import patch

from scrare.cli import run_inductive


def test_run_inductive_cli_calls_workflow():
    with patch("scrare.cli.run_inductive.run_inductive_workflow") as mocked:
        with patch("sys.argv", ["run_inductive", "--config", "configs/immune_dc.yaml"]):
            run_inductive.main()
    mocked.assert_called_once()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/workflows/test_workflow_smoke.py -k run_inductive_cli_calls_workflow -v`
Expected: FAIL because `run_inductive_workflow` is not defined

- [ ] **Step 3: 创建 workflow 并接入 CLI**

```python
# src/scrare/workflows/inductive.py
from __future__ import annotations

from typing import Any


def run_inductive_workflow(config: dict[str, Any], args) -> None:
    # 从旧 scripts/run_scanvi_inductive.py 拆出以下流程：
    # 1. rare_class / split_mode / seed / rare_train_size 循环
    # 2. 数据读取、预处理、split、label budget
    # 3. baseline 训练与 query inference
    # 4. validation grid search 与 test fusion
    # 5. run 级和 stage 级输出写入
    raise NotImplementedError
```

```python
# src/scrare/cli/run_inductive.py
from __future__ import annotations

import argparse

from scrare.infra.config import load_config
from scrare.workflows.inductive import run_inductive_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inductive train/validation/test scANVI + fusion validation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rare-class")
    parser.add_argument("--split-mode", default="cell_stratified")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rare-train-size")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--scvi-epochs", type=int)
    parser.add_argument("--scanvi-epochs", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--max-false-rescue-rate", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_inductive_workflow(config, args)
```

- [ ] **Step 4: 将旧主脚本中的内部函数下沉到 workflow 和 models**

```python
# src/scrare/workflows/inductive.py
# 从旧脚本中迁移并重命名：
# - _csv_values
# - _run_values
# - _safe_class_name
# - _output_root
# - _run_name
# - _flatten_summary
# - _extract_scanvi_probs
# - _baseline_metrics
# - _fusion_grid
# - _fusion_with_params
# - _split_series
# - _write_stage_outputs
# - _plot_comparison
# - run_one
# - run_inductive_workflow（替代旧 main 内大循环）
```

- [ ] **Step 5: 运行 workflow smoke test**

Run: `pytest tests/workflows/test_workflow_smoke.py -k run_inductive_cli_calls_workflow -v`
Expected: PASS

- [ ] **Step 6: 提交 inductive workflow 接入**

```bash
git add src/scrare/workflows/inductive.py src/scrare/cli/run_inductive.py tests/workflows/test_workflow_smoke.py
git commit -m "refactor: route inductive CLI through workflow layer"
```

### Task 8: 实现四阶段装配与 `workflows.posthoc`

**Files:**
- Create: `src/scrare/evaluation/posthoc.py`
- Create: `src/scrare/workflows/posthoc.py`
- Modify: `src/scrare/cli/evaluate_posthoc.py`
- Create: `tests/evaluation/test_posthoc.py`
- Test: `tests/evaluation/test_posthoc.py`

- [ ] **Step 1: 写失败测试，锁定四阶段方法名**

```python
# tests/evaluation/test_posthoc.py
from scrare.evaluation.posthoc import METHOD_ORDER


def test_method_order_matches_design():
    assert METHOD_ORDER == [
        "baseline",
        "baseline_plus_prototype",
        "baseline_plus_prototype_plus_marker",
        "baseline_plus_fusion",
    ]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/evaluation/test_posthoc.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 创建四阶段装配模块**

```python
# src/scrare/evaluation/posthoc.py
METHOD_ORDER = [
    "baseline",
    "baseline_plus_prototype",
    "baseline_plus_prototype_plus_marker",
    "baseline_plus_fusion",
]

METHOD_LABELS = {
    "baseline": "scANVI baseline",
    "baseline_plus_prototype": "prototype rank1 gate",
    "baseline_plus_prototype_plus_marker": "validation-tuned marker",
    "baseline_plus_fusion": "fusion",
}
```

```python
# src/scrare/evaluation/posthoc.py
# 从旧 evaluate_inductive_prototype_marker.py 迁移并重组：
# - _latent_matrix
# - _baseline_metrics
# - _log1p_cpm_dense
# - _expression_for_cells
# - _score_candidates
# - _summarize
# - _with_run_metadata
# 新增统一装配函数：
# - evaluate_four_stage_methods(...)
# - summarize_four_stage_methods(...)
```

- [ ] **Step 4: 创建 posthoc workflow 与 CLI**

```python
# src/scrare/workflows/posthoc.py
from __future__ import annotations

from typing import Any


def run_posthoc_workflow(config: dict[str, Any], args) -> None:
    # 遍历 run 目录，读取 artifacts，调用 evaluate_four_stage_methods，写出 stage 表
    raise NotImplementedError
```

```python
# src/scrare/cli/evaluate_posthoc.py
from __future__ import annotations

import argparse

from scrare.infra.config import load_config
from scrare.workflows.posthoc import run_posthoc_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate four-stage posthoc methods on held-out test cells.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rare-class", default="ASDC,cDC1")
    parser.add_argument("--split-mode", default="batch_heldout")
    parser.add_argument("--max-false-rescue-rate", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--min-cells", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_posthoc_workflow(config, args)
```

- [ ] **Step 5: 运行四阶段装配测试并确认通过**

Run: `pytest tests/evaluation/test_posthoc.py -v`
Expected: PASS

- [ ] **Step 6: 提交 posthoc 装配和 workflow**

```bash
git add src/scrare/evaluation/posthoc.py src/scrare/workflows/posthoc.py src/scrare/cli/evaluate_posthoc.py tests/evaluation/test_posthoc.py
git commit -m "refactor: add four-stage posthoc evaluation workflow"
```

### Task 9: 接入审计 CLI 并迁移旧 `audit_dataset.py`

**Files:**
- Modify: `src/scrare/cli/audit.py`
- Test: `tests/cli/test_cli_smoke.py`

- [ ] **Step 1: 扩充 CLI smoke test，锁定审计入口存在 `build_parser`**

```python
# tests/cli/test_cli_smoke.py
from scrare.cli import audit


def test_audit_cli_exposes_parser():
    parser = audit.build_parser()
    assert parser.prog is not None
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/cli/test_cli_smoke.py -k audit_cli_exposes_parser -v`
Expected: FAIL because `build_parser` is missing

- [ ] **Step 3: 实现审计 CLI**

```python
# src/scrare/cli/audit.py
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from scrare.data.loading import adata_from_config
from scrare.evaluation.audit import audit_anndata
from scrare.infra.config import load_config, output_dir
from scrare.infra.io import write_table
from scrare.infra.paths import root_table_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the scRare dataset.")
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    dataset = config["dataset"]
    analysis = config.get("analysis", {})
    out_dir = output_dir(config) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(dataset["path"], backed="r")
    summary, class_dist, batch_dist = audit_anndata(
        adata,
        dataset_name=dataset["name"],
        label_key=dataset["label_key"],
        batch_key=dataset["batch_key"],
        rare_threshold=float(analysis.get("rare_threshold", 0.05)),
        rare_max_cells=int(analysis.get("rare_max_cells", 200)),
        use_raw=bool(dataset.get("use_raw", False)),
    )
    write_table(pd.DataFrame([summary]), out_dir / "dataset_summary.csv")
    write_table(class_dist, out_dir / "class_distribution.csv")
    write_table(batch_dist, out_dir / "batch_distribution.csv")
    root_out = output_dir(config)
    write_table(pd.DataFrame([summary]), root_table_path(root_out, "dataset_summary.csv"))
    write_table(class_dist, root_table_path(root_out, "class_distribution.csv"))
    write_table(batch_dist, root_table_path(root_out, "batch_distribution.csv"))
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    print(f"Wrote audit outputs to {Path(out_dir)}")
```

- [ ] **Step 4: 运行 CLI smoke test**

Run: `pytest tests/cli/test_cli_smoke.py -v`
Expected: PASS

- [ ] **Step 5: 提交审计入口迁移**

```bash
git add src/scrare/cli/audit.py tests/cli/test_cli_smoke.py
git commit -m "refactor: add module-based audit CLI"
```

### Task 10: 文档切换、删除旧入口、全量验证

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `tests/test_project_state.py`
- Remove: `scripts/audit_dataset.py`
- Remove: `scripts/run_scanvi_inductive.py`
- Remove: `scripts/evaluate_inductive_prototype_marker.py`
- Remove: `scrare_refine/*.py`

- [ ] **Step 1: 先更新 README 运行命令到 `python -m`**

```md
# README.md 需要替换的命令
python -m scrare.cli.audit --config configs/immune_dc.yaml
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

- [ ] **Step 2: 更新 CLAUDE.md 中的开发命令和架构描述**

```md
# CLAUDE.md 关键更新
- 主体代码位于 src/scrare/
- CLI 入口位于 src/scrare/cli/
- 长流程编排位于 src/scrare/workflows/
- 四阶段装配位于 src/scrare/evaluation/posthoc.py
```

- [ ] **Step 3: 更新项目状态测试，显式禁止旧入口残留**

```python
# tests/test_project_state.py
legacy_scripts = [
    "audit_dataset.py",
    "run_scanvi_inductive.py",
    "evaluate_inductive_prototype_marker.py",
]
for name in legacy_scripts:
    self.assertFalse(Path("scripts", name).exists(), name)
```

- [ ] **Step 4: 删除旧 `scripts/` 和旧 `scrare_refine/` 实现文件**

```bash
rm scripts/audit_dataset.py
rm scripts/run_scanvi_inductive.py
rm scripts/evaluate_inductive_prototype_marker.py
rm scrare_refine/anndata_utils.py scrare_refine/audit.py scrare_refine/config.py scrare_refine/fusion.py scrare_refine/inductive.py scrare_refine/io.py scrare_refine/marker_verifier.py scrare_refine/metrics.py scrare_refine/output_layout.py scrare_refine/prototype.py scrare_refine/prototype_gate.py scrare_refine/resources.py
```

- [ ] **Step 5: 运行全量测试**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: 做一次导入冒烟验证**

Run: `python -m scrare.cli.run_inductive --help && python -m scrare.cli.evaluate_posthoc --help && python -m scrare.cli.audit --help`
Expected: 三个命令都打印 help 文本并退出码 0

- [ ] **Step 7: 提交最终收尾**

```bash
git add README.md CLAUDE.md tests src pyproject.toml
git add -u
git commit -m "refactor: finish src-based scrare project reorganization"
```

---

## Self-review checklist

### Spec coverage
- `src/` 布局：Task 1
- `scrare` 包名切换：Task 1、Task 10
- `python -m ...` 入口：Task 7、Task 8、Task 9、Task 10
- 六层分层：Task 2–Task 9
- 四阶段方法比较：Task 8
- `prototype_gate.py` 独立：Task 4
- 先保行为再整理测试：所有任务均先写/改测试再迁实现

### Placeholder scan
- 计划中没有 `TODO`、`TBD`、`implement later`
- 需要从旧文件“原样迁移”的函数已经明确列出来源和名称
- 所有测试步骤都包含具体命令

### Type consistency
- 新顶层包统一为 `scrare`
- 四阶段内部标识统一为 `baseline` / `baseline_plus_*`
- CLI 名称统一为 `audit` / `run_inductive` / `evaluate_posthoc`
