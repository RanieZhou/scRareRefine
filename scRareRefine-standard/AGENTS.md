# AGENTS.md

## 目的

本文档定义 ARIS、Codex、Claude Code 或其他 AI 编程/科研 agent 在 scRareRefine 项目中的工作规则。

当前标准项目根目录是本目录。当前核心 Python 包是 `scrare`，代码位于 `src/scrare/`。

## 项目概要

项目名称：scRareRefine。

项目目标：基于 scANVI 的输出结果，提高稀有细胞类型识别效果。

scRareRefine 当前使用：

1. scANVI 预测概率。
2. scANVI latent embedding。
3. latent space 中的类别 prototype。
4. probability-prototype fusion。
5. prototype gate。
6. marker verification。
7. validation-driven 参数选择。

## 不可违反的规则

### 1. 修改前必须先给计划

修改代码、配置、实验脚本或文档前，agent 必须先输出：

```text
1. 本次修改目标
2. 准备修改的文件
3. 预期输出结果
4. 潜在风险
5. 测试或验证方式
```

除非用户明确说直接执行，否则不能先改后说。

### 2. 禁止修改原始数据

以下内容视为只读：

```text
data/raw/
原始 .h5ad 文件
未备份的原始输入数据
```

禁止覆盖、删除或直接改写原始数据。

### 3. 禁止随意新建混乱目录

顶层目录必须符合 `PROJECT_STRUCTURE.md`。

核心位置：

```text
文档：docs/
配置：configs/
原始数据：data/raw/
中间数据：data/processed/、data/splits/、data/embeddings/
核心代码：src/scrare/
测试：tests/
正式结果：results/
日志：logs/
模型权重：checkpoints/
临时文件：tmp/
```

注意：当前代码仍有过渡性的 `outputs/` 写入逻辑。不要在未单独设计、迁移和验证的情况下，强行把所有输出路径切换到 `results/`。

旧模板包目录 `sc_rare_refine/` 不是当前安装包、运行入口或新代码位置。不要向该模板目录新增逻辑；如需删除该模板目录，必须单独取得用户明确同意。

不要新建 `new_code/`、`final/`、`backup2/`、`test123/`、`临时实验/`、`新版结果/` 等目录。

### 4. 禁止无记录地改变实验设置

agent 不能悄悄改变：

- 随机种子。
- train/validation/test 划分规则。
- rare cell 定义。
- label 列。
- batch 列。
- 预处理流程。
- baseline 设置。
- 指标定义。
- scANVI 训练参数。

任何变化都必须写入对应 config、结果日志或实验记录。

### 5. 禁止未经验证的科研主张

没有实验支持时，不得写出：

```text
显著优于
state-of-the-art
全面超过
临床可用
适用于所有单细胞数据
解决了稀有细胞识别问题
```

## 当前运行入口

正式入口优先使用 Python module 命令：

```bash
python -m scrare.cli.audit --config configs/immune_dc.yaml
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

console scripts 也应指向同一组入口：

```text
scrare-audit = scrare.cli.audit:main
scrare-run-inductive = scrare.cli.run_inductive:main
scrare-evaluate-posthoc = scrare.cli.evaluate_posthoc:main
```

## 当前测试入口

```bash
pytest tests/test_project_state.py tests/cli/test_cli_smoke.py -v
pytest -v
```

修改核心工作流后，优先运行相关子系统测试，再运行全量测试。

## 受限操作

以下操作必须获得明确同意：

1. 安装新 Python 包。
2. 修改环境文件。
3. 运行长时间实验。
4. 微调 scANVI。
5. 改变数据预处理规则。
6. 使用 sudo/admin 权限。
7. git commit。
8. git push。
9. 删除文件。
10. 移动已有实验结果。
11. force push、reset、删除分支、清理大目录等高风险 Git 操作。

## 禁止操作

agent 不得：

1. 修改原始数据。
2. 删除大目录。
3. 伪造结果。
4. 手动编辑结果文件让指标变好。
5. 在训练中错误使用测试标签。
6. 故意削弱 baseline。
7. 只报告成功 seed。
8. 隐藏失败实验。
9. 编造论文结论。
10. 把核心代码散落到项目根目录。

## 每次实验必须记录的信息

每个正式实验必须记录：

```text
experiment_id
日期
数据集
随机种子
方法名
配置文件
git commit hash
label column
batch column
rare-cell definition
评价指标
输出路径
主要观察
失败原因或异常
```

并更新 `RESULTS_LOG.md` 或对应实验记录。
