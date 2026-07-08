"""Shared RUNS CLI resolver for all run_*_comparison.py scripts.

Supported forms:
  - positional filters: keep hard-coded RUNS whose config path contains a token
  - --seeds S1 S2 ...: replace the seed dimension
  - --rts R1 R2 ...: keep or construct the requested rare_train_size values
  - --configs C1 C2 ...: construct RUNS from explicit config paths

The explicit --configs mode is used for add-on datasets such as mouse TMS,
without editing nine method scripts independently.
"""

from __future__ import annotations


DEFAULT_RTS = ["0.01", "0.05", "0.10", "all"]


def _consume_values(args: list[str], option: str) -> list[str] | None:
    if option not in args:
        return None
    i = args.index(option)
    j = i + 1
    values: list[str] = []
    while j < len(args) and not args[j].startswith("--"):
        values.append(args[j])
        j += 1
    del args[i:j]
    return values


def _consume_seed_values(args: list[str]) -> list[str] | None:
    option = "--seeds"
    if option not in args:
        return None
    i = args.index(option)
    j = i + 1
    values: list[str] = []
    while j < len(args) and args[j].lstrip("-").isdigit():
        values.append(args[j])
        j += 1
    del args[i:j]
    return values


def _unique_pairs(runs: list[tuple]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for config_path, _seed, rts in runs:
        key = (str(config_path), str(rts))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def resolve_runs(runs: list[tuple], argv: list[str]) -> list[tuple]:
    """Resolve [(config_path, seed, rts_str), ...] from script argv."""
    args = list(argv)

    seed_values = _consume_seed_values(args)
    rts_values = _consume_values(args, "--rts")
    config_values = _consume_values(args, "--configs")

    seeds = [int(s) for s in seed_values] if seed_values else None
    rts_list = [str(r) for r in rts_values] if rts_values else None

    if config_values:
        configs = [str(c) for c in config_values]
        seeds = seeds or [42]
        rts_list = rts_list or DEFAULT_RTS
        out = [(config, seed, rts) for config in configs for seed in seeds for rts in rts_list]
    else:
        out = runs
        if rts_list:
            allowed_rts = set(rts_list)
            out = [r for r in out if str(r[2]) in allowed_rts]

    filters = [a for a in args if not a.startswith("-")]
    if filters:
        out = [r for r in out if any(token in r[0] for token in filters)]

    if seeds:
        out = [(config, seed, rts) for seed in seeds for config, rts in _unique_pairs(out)]

    return out
