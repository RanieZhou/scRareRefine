"""scRareRefine 端到端模块化运行管线

通过依次导入 preprocess -> model -> rescue -> utils 等模块函数，
在 Python 进程内完成自适应数据预处理、模型半监督训练、后处理拯救校正以及科学可视化图表的生成。
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# 将项目根目录插入系统搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocess import run_preprocessing
from src.model import run_model_training
from src.rescue import run_post_hoc_rescue, MarkerRescuer, _load_expression_subset, DEFAULT_CONFORMAL_ALPHA
from src.utils import (
    load_config,
    load_adata,
    make_run_dir,
    parse_rare_train_size,
    classification_tables,
    write_table,
    print_classification_report,
    plot_marker_violin,
    plot_method_comparison,
    plot_rescue_effect,
    build_manifest,
    check_manifest,
    ResourceMonitor
)


def _try_load_cached_embeddings(
    run_dir: Path,
    config: dict,
    *,
    seed: int,
    rare_class: str,
    rare_train_size,
    label_column: str,
    batch_key: str,
    split_mode: str,
    force: bool = False,
) -> tuple | None:
    """若 scANVI baseline 已保存嵌入文件且 manifest provenance 校验通过，
    则直接加载并返回 (predictions_dict, latents_dict, selected_genes)；
    否则（force=True / 文件缺失 / manifest 不匹配）返回 None 以触发重新训练。"""
    if force:
        return None

    splits = ["train", "validation", "test"]
    emb_dir = run_dir / "embeddings"
    hvg_file = run_dir / "selected_hvg_genes.csv"

    required = [emb_dir / f"{s}_{t}.csv" for s in splits for t in ("predictions", "latent")]
    required.append(hvg_file)
    if not all(p.exists() for p in required):
        return None

    if not check_manifest(
        run_dir, config, seed=seed, rare_class=rare_class, rare_train_size=rare_train_size,
        label_column=label_column, batch_key=batch_key, split_mode=split_mode,
    ):
        return None

    predictions_dict = {s: pd.read_csv(emb_dir / f"{s}_predictions.csv") for s in splits}
    latents_dict = {s: pd.read_csv(emb_dir / f"{s}_latent.csv") for s in splits}
    selected_genes = pd.read_csv(hvg_file)["gene"].tolist()
    return predictions_dict, latents_dict, selected_genes

# ==============================================================================
# 【全局填空区】 请在此配置您的数据集路径、稀有细胞类别等控制参数（支持被命令行参数覆盖）
# ==============================================================================
CONFIG_PATH = "configs/immune_dc.yaml"  # 默认使用的配置文件路径
SEED = 42                                # 实验随机种子
RARE_CLASS = None                        # 目标稀有类名称（若为 None 则从 yaml 配置的 experiment.rare_class 读取）
LABEL_COL = None                         # 细胞类型真实标签列名（若为 None 则从 yaml 配置的 dataset.label_key 读取）
RARE_TRAIN_SIZE = None                   # 训练集稀有细胞显式标注数量（设为 None 则优先使用 YAML 中配置的列表首元素；也可配 float、int 或 'all'）
SPLIT_MODE = None                        # 样本切分模式（设为 None 则默认 batch_heldout；可选 cell_stratified）
# 后处理拯救所允许的最大误判率阈值（主路径为 conformal，实际作为 conformal_alpha 传入）。
# 与 tools/comparison/run_scrarerefine_comparison.py 共用同一来源常量（src/rescue.py 的
# DEFAULT_CONFORMAL_ALPHA），避免两处分别硬编码导致官方 baseline 对比和主流水线指标不可比。
MAX_FALSE_RESCUE_RATE = DEFAULT_CONFORMAL_ALPHA
# ==============================================================================


def resolve_param(cli_value, config_value, global_value=None):
    """ 参数决策优先级：命令行传入 > 填空区全局变量（若有且非空） > 配置文件参数 """
    if cli_value is not None:
        return cli_value
    if global_value is not None:
        return global_value
    return config_value


def main() -> None:
    # 允许接收命令行参数以覆盖全局填空区
    parser = argparse.ArgumentParser(
        description="scRareRefine 端到端一键主流水线脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子")
    parser.add_argument("--rare_class", default=None, help="目标稀有类名称")
    parser.add_argument("--label_col", default=None, help="真实标签列名")
    parser.add_argument("--rare_train_size", default=None, help="稀有细胞显式标注规模")
    parser.add_argument("--split_mode", default=None, help="切分模式 (batch_heldout | cell_stratified)")
    parser.add_argument("--max_false_rescue_rate", type=float, default=None,
                         help="最大误拯救率（主路径为 conformal，此值会作为 conformal_alpha 传入；"
                              "对 gate_only/gate_marker/fusion 才是直接的 FFR 约束）")
    parser.add_argument("--force", action="store_true",
                         help="忽略已有 embeddings 缓存与 manifest 校验，强制重新训练")
    args = parser.parse_args()

    # 1. 载入配置文件与各参数的合并决策
    cfg_file = args.config if args.config is not None else CONFIG_PATH
    config = load_config(cfg_file)
    exp = config.get("experiment", {})
    
    seed = args.seed if args.seed is not None else SEED
    rare_class = resolve_param(args.rare_class, exp.get("rare_class"), RARE_CLASS)
    label_column = resolve_param(args.label_col, config["dataset"].get("label_key", "label"), LABEL_COL)
    batch_key = config["dataset"].get("batch_key", "batch")
    split_mode = resolve_param(args.split_mode, exp.get("split_mode", "batch_heldout"), SPLIT_MODE)
    
    # 智能读取 YAML 配置中的默认标注规模（若是列表取第一个元素，否则默认 10）
    yaml_sizes = exp.get("rare_train_sizes", [])
    yaml_default_size = yaml_sizes[0] if isinstance(yaml_sizes, list) and len(yaml_sizes) > 0 else 10
    
    raw_size = resolve_param(args.rare_train_size, yaml_default_size, RARE_TRAIN_SIZE)
    parsed_rare_size = parse_rare_train_size(raw_size)
    
    max_false_rescue_rate = resolve_param(args.max_false_rescue_rate, None, MAX_FALSE_RESCUE_RATE)

    # 验证关键参数
    if rare_class is None:
        raise ValueError("未在全局填空区或配置文件中配置 rare_class！")
        
    # 计算并创建本次实验的唯一存储路径
    run_dir = make_run_dir(config, split_mode, seed, rare_class, parsed_rare_size)
    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("      【scRareRefine 一键模块化流水线启动】")
    print(f"  - 配置文件 : {cfg_file}")
    print(f"  - 稀有类型 : {rare_class}")
    print(f"  - 真实标签 : {label_column}")
    print(f"  - 标注规模 : {parsed_rare_size}")
    print(f"  - 随机种子 : {seed}")
    print(f"  - 保存目录 : {run_dir}")
    print("=" * 80 + "\n")

    # 开启资源与耗时监控
    with ResourceMonitor(sample_interval_seconds=1.0) as monitor:
        
        # ==========================================
        # 步骤 1：读取数据、进行数据体检与严格三路划分
        # ==========================================
        print(">>> [步骤 1/4] 加载原始 h5ad 数据并执行自适应预处理...")
        adata_raw = load_adata(config)
        adata, train_idx, val_idx, test_idx = run_preprocessing(
            adata_raw,
            label_column=label_column,
            batch_key=batch_key,
            split_mode=split_mode,
            seed=seed,
            rare_class=rare_class
        )

        # ==========================================
        # 步骤 2：选择高变基因、两阶段模型训练与表征提取
        # ==========================================
        cached = _try_load_cached_embeddings(
            run_dir, config,
            seed=seed, rare_class=rare_class, rare_train_size=parsed_rare_size,
            label_column=label_column, batch_key=batch_key, split_mode=split_mode,
            force=args.force,
        )
        if cached is not None:
            print(">>> [步骤 2/4] 检测到已有 scANVI 嵌入缓存，跳过重新训练，直接加载...")
            predictions_dict, latents_dict, selected_genes = cached
        else:
            print(">>> [步骤 2/4] 启动 scANVI 半监督神经网络训练流程...")
            _, predictions_dict, latents_dict, selected_genes = run_model_training(
                adata,
                train_idx,
                val_idx,
                test_idx,
                label_column=label_column,
                batch_key=batch_key,
                rare_class=rare_class,
                rare_train_size=parsed_rare_size,
                config=config,
                seed=seed
            )
            # 保存 embeddings，供 compare_baselines.py 等脚本复用
            emb_dir = run_dir / "embeddings"
            emb_dir.mkdir(parents=True, exist_ok=True)
            for _split in ["train", "validation", "test"]:
                predictions_dict[_split].to_csv(emb_dir / f"{_split}_predictions.csv", index=False)
                latents_dict[_split].to_csv(emb_dir / f"{_split}_latent.csv", index=False)
            pd.DataFrame({"gene": selected_genes}).to_csv(run_dir / "selected_hvg_genes.csv", index=False)

            # 写入 provenance manifest，供下次复用缓存前校验（与 train_cache.py /
            # run_scrarerefine_comparison.py 共用 src/utils.build_manifest/check_manifest）
            manifest = build_manifest(
                config, cfg_file,
                label_column=label_column, batch_key=batch_key, split_mode=split_mode,
                seed=seed, rare_class=rare_class, rare_train_size=parsed_rare_size,
                predictions_dict=predictions_dict,
                n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f">>> [步骤 2/4] embeddings 已缓存至 {emb_dir}（manifest split_hash={manifest['split_hash']}）")

        # ==========================================
        # 步骤 3：提取预测标签及不确定性特征，进行后处理拯救校正
        # ==========================================
        print(">>> [步骤 3/4] 模型初步推理完成，开始应用 Post-hoc 校正与拯救算法...")
        y_true_test = predictions_dict["test"]["true_label"].astype(str)
        base_pred_test = predictions_dict["test"]["predicted_label"].astype(str)
        
        # 建立 baseline 评估记录
        bl_metrics, _ = classification_tables(y_true_test, base_pred_test, rare_class=rare_class)
        metrics_rows = [{
            "method": "baseline",
            "seed": seed,
            "rare_train_size": str(parsed_rare_size),
            **bl_metrics,
            "n_rescued": 0,
            "n_false_rescues": 0,
            "major_to_rare_false_rescue_rate": 0.0
        }]
        
        # 执行 scRareRefine 自适应融合拯救
        # 主路径为 conformal：conformal 分支只读 conformal_alpha，不读 max_false_rescue_rate
        # （二者在 run_post_hoc_rescue 内语义独立，详见 src/rescue.py 的函数 docstring），
        # 因此把 CLI 暴露的唯一阈值参数映射到当前实际生效的 conformal_alpha 上。
        final_test_pred, summary = run_post_hoc_rescue(
            adata,
            predictions_dict,
            latents_dict,
            selected_genes,
            rare_class=rare_class,
            strategy="conformal",
            conformal_alpha=max_false_rescue_rate
        )

        overall_metrics, _ = classification_tables(y_true_test, final_test_pred, rare_class=rare_class)
        metrics_rows.append({
            "method": "scRareRefine",
            "seed": seed,
            "rare_train_size": str(parsed_rare_size),
            **overall_metrics,
            "n_rescued": summary["n_rescued"],
            "n_false_rescues": summary["n_false_rescues"],
            "major_to_rare_false_rescue_rate": summary["n_false_rescues"] / int(y_true_test.ne(rare_class).sum()) if int(y_true_test.ne(rare_class).sum()) else 0.0
        })
        print_classification_report(y_true_test, final_test_pred, rare_class=rare_class)

        # 汇总策略评估记录并保存
        metrics_df = pd.DataFrame(metrics_rows)
        write_table(metrics_df, out_dir / "final_metrics.csv")
        print(f"-> [保存指标] 成功将各策略的分类对照表写入: {out_dir / 'final_metrics.csv'}")

        # ==========================================
        # 步骤 4：联动可视化绘图，输出小提琴图与对比柱状图
        # ==========================================
        print(">>> [步骤 4/4] 启动绘图工具，生成生信动态分析图表...")
        
        # (1) 自动提取特异 Marker 基因绘制表达量小提琴图
        train_cell_ids = predictions_dict["train"]["cell_id"].astype(str).tolist()
        train_expr = _load_expression_subset(adata, train_cell_ids, selected_genes)
        ref_labels = predictions_dict["train"]["true_label"]
        ref_is_labeled = predictions_dict["train"]["is_labeled_for_scanvi"].astype(bool).to_numpy()
        
        marker_rescuer = MarkerRescuer(rare_class)
        marker_rescuer.compute_marker_signatures(train_expr, selected_genes, ref_labels, ref_is_labeled, top_n=5)
        
        rare_markers = marker_rescuer.signatures.get(rare_class, [])
        if rare_markers:
            plot_marker_violin(adata, label_column, rare_markers[:3], out_dir / "marker_violin.png", rare_class=rare_class)
            
        # (2) 绘制多策略分类性能对比图与拯救效果柱状图
        plot_method_comparison(metrics_df, out_dir / "method_comparison.png", rare_class=rare_class)
        plot_rescue_effect(metrics_df, out_dir / "rescue_effect.png", rare_class=rare_class)

    # 打印最终系统资源报告
    usage = monitor.summary()
    print("\n" + "=" * 80)
    print("      【scRareRefine 运行状态总结】")
    print(f"  - 消耗 Wall-Time  : {usage['wall_time_seconds']:.2f} 秒")
    print(f"  - 峰值物理内存占用 : {usage['peak_rss_mb']:.2f} MB")
    print(f"  - 结果大表保存路径 : {out_dir / 'final_metrics.csv'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
