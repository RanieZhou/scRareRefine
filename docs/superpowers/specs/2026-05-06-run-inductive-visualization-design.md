# run_inductive 主流程可视化输出设计

## 背景

当前 `python -m scrare.cli.run_inductive` 已经能够产出 baseline artifacts、five-method 汇总表，以及 marker / fusion 相关 CSV，但主流程还不会自动生成任何图表。用户希望主流程在保留表格输出的同时，默认附带可直接查看的静态可视化结果，并额外给出运行时间与峰值内存情况。

现状约束：

- 主流程已有 root 级聚合表：`stages/inductive_methods/*.csv`
- 当前没有统一的资源汇总表，无法直接绘制耗时和内存图
- 当前没有绘图模块，绘图逻辑不能继续堆进 `src/scrare/workflows/inductive.py`

## 目标

1. `run_inductive` 默认自动生成聚合静态图表。
2. 图表优先面向 root 级聚合结果，而不是每个 run 单独出图。
3. 主流程默认输出运行时间和峰值内存图。
4. 图层只消费聚合后的 CSV，不重新训练模型，也不重新计算方法结果。
5. 输出布局清晰稳定，不与现有 run artifacts 或 method tables 混杂。

## 非目标

1. 本次不生成交互式 HTML 图。
2. 本次不为每个 `runs/<run>/` 单独生成图。
3. 本次不引入 UMAP / t-SNE / latent 空间样本投影图。
4. 本次不做细粒度 profiler（baseline/fusion/marker 分阶段资源剖析）。

## 方案选择

候选方案：

1. 主流程默认生成聚合静态图 + 单独资源图。
2. 主流程默认生成聚合静态图，但资源只出表不出图。
3. 主流程同时生成聚合图和单 run 图。

采用方案 1。

原因：

- 与用户偏好一致：聚合图优先、静态图、默认生成、资源也要图
- 不会造成每个 run 下文件数量爆炸
- 可以直接建立在现有 root 级 CSV 聚合层之上，只需补一条薄的资源数据链路

## 输出布局

在现有：

- `runs/<run>/artifacts/`
- `runs/<run>/stages/inductive_methods/`
- `stages/inductive_methods/`

之外，新增：

- `stages/inductive_plots/`

最终结构：

```text
outputs/<dataset>/<inductive_cell|inductive_batch>/<rare_class>/
├── runs/
│   └── <run>/
│       ├── artifacts/
│       ├── stages/inductive_methods/
│       └── resource_summary.csv
├── stages/
│   ├── inductive_methods/
│   │   ├── five_method_effect_runs.csv
│   │   ├── five_method_effect_summary.csv
│   │   ├── validation_marker_threshold_curve.csv
│   │   ├── selected_marker_thresholds.csv
│   │   ├── prototype_test_candidates.csv
│   │   ├── marker_verified_test_candidates.csv
│   │   ├── fusion_validation_grid.csv
│   │   └── resource_summary.csv
│   └── inductive_plots/
│       ├── five_method_metric_summary.png
│       ├── marker_threshold_curve.png
│       ├── fusion_validation_heatmap.png
│       ├── runtime_summary.png
│       └── memory_summary.png
```

## 默认图表清单

### 1. `five_method_metric_summary.png`

作用：聚合比较 5 个方法的核心指标。

建议指标：

- `overall_accuracy`
- `macro_f1`
- `rare_precision`
- `rare_recall`
- `rare_f1`

数据来源：

- `stages/inductive_methods/five_method_effect_summary.csv`

推荐表现形式：

- 方法为 x 轴
- 指标值为 y 轴
- 多指标分面展示
- 使用聚合均值，必要时加误差线

### 2. `marker_threshold_curve.png`

作用：展示 validation 上 marker threshold 搜索轨迹，并标记最终选择的 threshold。

数据来源：

- `stages/inductive_methods/validation_marker_threshold_curve.csv`
- `stages/inductive_methods/selected_marker_thresholds.csv`

推荐表现形式：

- x 轴：threshold
- y 轴：`rare_error_recall`
- 叠加 `major_to_rare_false_rescue_rate`
- 标出最终选中的 threshold

### 3. `fusion_validation_heatmap.png`

作用：展示 fusion 参数搜索结果。

数据来源：

- `stages/inductive_methods/fusion_validation_grid.csv`

推荐表现形式：

- 使用 `temperature × alpha_min` 热图
- `beta` 做分面或选择最常用视图
- 颜色优先表示 `rare_f1`
- 如有需要，保留 `overall_accuracy` 作为备选颜色指标

### 4. `runtime_summary.png`

作用：展示单个 slice 的总运行时间聚合情况。

数据来源：

- `stages/inductive_methods/resource_summary.csv`

推荐表现形式：

- x 轴：`rare_train_size`
- y 轴：`wall_time_seconds`
- 颜色：`split_mode`
- 如图面过挤，可按 rare class 或 split mode 分面

### 5. `memory_summary.png`

作用：展示单个 slice 的峰值内存聚合情况。

数据来源：

- `stages/inductive_methods/resource_summary.csv`

推荐表现形式：

- x 轴：`rare_train_size`
- y 轴：`peak_memory_mb`
- 颜色：`split_mode`

## 数据来源原则

图层只消费聚合后的 CSV：

- 不重新读取模型对象
- 不重新训练模型
- 不重新执行 prototype / marker / fusion 计算
- 不重新从原始 artifacts 拼接实验结论

这样可将绘图层定义为主流程的纯下游消费者，边界清晰，易于测试。

## 资源数据链路

### 资源表需求

当前主流程缺少统一可聚合的资源表，因此新增：

#### run 级资源表

- `runs/<run>/resource_summary.csv`

#### root 级聚合资源表

- `stages/inductive_methods/resource_summary.csv`

### 最小字段集

- `run`
- `seed`
- `rare_train_size`
- `rare_class`
- `split_mode`
- `wall_time_seconds`
- `peak_memory_mb`

### 统计边界

资源统计语义定义为：

> 单个 `(rare_class, split_mode, seed, rare_train_size)` slice 整次 `run_inductive` 主流程的总耗时与峰值内存

不拆分 baseline / fusion / marker / plotting 各子阶段。

原因：

- 用户当前需要的是主流程默认资源概览，而不是 profiler
- 先保证总耗时 / 总峰值内存稳定输出，复杂度最低，价值最高

## 触发时机

主流程顺序调整为：

1. baseline artifacts 写盘
2. run 级 method tables 写盘
3. root 级 `stages/inductive_methods/*.csv` 重建
4. root 级 `stages/inductive_methods/resource_summary.csv` 重建
5. root 级 `stages/inductive_plots/*.png` 重建

建议在 workflow 中新增与 `_rebuild_stage_outputs(root)` 对称的绘图重建步骤，例如概念上：

- `_rebuild_stage_outputs(root)`
- `_rebuild_plot_outputs(root)`

这样当：

- 跑完整实验
- 只跑单个 slice
- 只重算某个方法并复用 baseline

图都能自动重建到最新状态。

## 默认行为

- 默认总是生成图
- 不新增 `--with-plots` 开关
- 不拆出单独的绘图 CLI

用户心智保持为：

> 跑完 `run_inductive`，除了表格和 artifacts，默认也会得到聚合图表。

## 失败语义

### 数据层失败

以下失败直接让命令失败：

- baseline 训练失败
- method 评估失败
- CSV 聚合失败

### 绘图层失败

若绘图阶段失败：

- 已成功写出的 CSV 必须保留
- 命令状态仍然失败
- 错误信息明确说明“主结果已写出，但绘图失败”

这样既不丢主结果，也不会默默吞掉图层失败。

### 条件性空图

若某张图对应 CSV 存在但数据为空，推荐生成带说明文字的空图，而不是静默缺文件。这样：

- 输出目录稳定
- 用户不会因为缺图去猜是 bug 还是正常无数据

## 模块边界

为避免 `src/scrare/workflows/inductive.py` 继续膨胀，绘图逻辑抽到独立模块。

推荐位置：

- `src/scrare/visualization/inductive.py`

原因：

- 比放在 `evaluation/plots.py` 更能表达这是主流程输出层，而不是指标计算层
- 后续若新增 audit 图、posthoc 图，也更容易分模块扩展

workflow 负责：

- 组织路径
- 触发重建
- 传递 root 级 CSV 路径

visualization 模块负责：

- 读取聚合表
- 校验列
- 生成静态图片

## 测试策略

### 1. 单元测试：绘图输入契约

验证：

- 最小 DataFrame 能成功输出图片文件
- 缺关键列时报清楚错误
- 空数据时按设计生成说明图或报清晰错误

### 2. workflow 测试：主流程会触发绘图

在 `tests/workflows/test_inductive_workflow.py` 中补 monkeypatch 测试，锁定调用顺序：

- 先重建 stage CSV
- 再重建 plots

### 3. 布局回归测试

验证：

- `stages/inductive_plots/` 被创建
- 预期文件名存在
- 图不会散落到 `runs/<run>/` 或根目录其他位置

### 4. 资源汇总测试

验证：

- 每个 slice 会生成 run 级 `resource_summary.csv`
- root 级 `resource_summary.csv` 会跨 runs 正确聚合
- 聚合结果能支撑运行时间图和内存图

## 推荐实现顺序

1. 先补 run 级与 root 级 `resource_summary.csv`
2. 新增 `src/scrare/visualization/inductive.py`
3. 先实现 `five_method_metric_summary.png`
4. 再实现 `marker_threshold_curve.png`
5. 再实现 `fusion_validation_heatmap.png`
6. 最后实现 `runtime_summary.png` 与 `memory_summary.png`
7. 更新 README 中主流程输出目录说明

## 成功标准

完成后，用户执行：

```bash
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
```

除了现有 CSV 和 baseline artifacts，还会默认得到：

- root 级聚合方法表
- root 级聚合资源表
- `stages/inductive_plots/*.png` 静态图表

并且：

- 不为每个 run 额外生成单 run 图
- 不生成交互式 HTML
- 绘图只消费聚合 CSV
- 主结果与图表目录结构清晰稳定
