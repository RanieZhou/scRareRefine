"""对比脚本 RUNS 的统一 CLI 解析（单一来源，9 个 run_*_comparison.py 共用）。

历史上每个脚本各写一份「位置参数 = config 子串过滤」逻辑；多 seed（第十三轮 G01）
需要再加 --seeds 覆盖。为避免 9 处各自实现产生漂移，集中到此函数。

支持两种 CLI：
  - 位置参数（不以 - 开头）：config 子串过滤，如 `... pancreas_integrated`
  - --seeds S1 S2 ...：把 RUNS 的 seed 维度替换为指定 seed 列表
        （按 (config, rts) × seeds 笛卡尔展开；默认不传则沿用 RUNS 内原 seed=42）

例：
  run_scanvi_comparison.py --seeds 43 44                 # 全数据集，seed 43/44
  run_scanvi_comparison.py --seeds 43 44 immune_dc       # 仅 immune_dc，seed 43/44
"""
from __future__ import annotations


def resolve_runs(runs: list[tuple], argv: list[str]) -> list[tuple]:
    """按 CLI 解析 RUNS：先 --seeds 覆盖 seed 维度，再按位置参数做 config 子串过滤。

    runs: [(config_path, seed, rts_str), ...]
    argv: 通常传 sys.argv[1:]
    """
    args = list(argv)

    # 1) 抽取 --seeds 及其后续数字参数，并从 args 移除（避免被当成 config 过滤词）
    seeds: list[int] | None = None
    if "--seeds" in args:
        i = args.index("--seeds")
        j = i + 1
        vals: list[int] = []
        while j < len(args) and args[j].lstrip("-").isdigit():
            vals.append(int(args[j]))
            j += 1
        seeds = vals or None
        del args[i:j]

    # 2) 位置参数 = config 子串过滤
    flt = [a for a in args if not a.startswith("-")]
    out = runs
    if flt:
        out = [r for r in out if any(s in r[0] for s in flt)]

    # 3) --seeds 覆盖：按 (config, rts) × seeds 展开（去重保序）
    if seeds:
        pairs: list[tuple] = []
        seen: set[tuple] = set()
        for (c, _s, r) in out:
            if (c, r) not in seen:
                seen.add((c, r))
                pairs.append((c, r))
        out = [(c, s, r) for s in seeds for (c, r) in pairs]

    return out
