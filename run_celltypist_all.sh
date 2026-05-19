#!/bin/bash
# Run CellTypist baseline (Stage 3c) on all existing completed runs.
# Skips runs where celltypist/test_metrics.csv already exists.

set -e
PYTHON="$(which python3)"
SCRIPT="src/03c_celltypist_baseline.py"
OUTPUTS="outputs"

run_ct() {
    local config="$1"
    local seed="$2"
    local rare_class="$3"
    local rare_train_size="$4"
    local split_mode="$5"
    local run_dir="$6"

    if [ -f "${run_dir}/celltypist/test_metrics.csv" ]; then
        echo "  [SKIP] already done: ${run_dir}"
        return 0
    fi

    echo "  Running CellTypist: config=${config} seed=${seed} rare_class='${rare_class}' size=${rare_train_size} mode=${split_mode}"
    "$PYTHON" "$SCRIPT" \
        --config "$config" \
        --seed "$seed" \
        --rare_class "$rare_class" \
        --rare_train_size "$rare_train_size" \
        --split_mode "$split_mode" \
        && echo "  [OK] ${run_dir}" \
        || echo "  [FAIL] ${run_dir}"
}

# immune_dc: ASDC, cDC1
for run_dir in "$OUTPUTS"/immune_dc/*/; do
    [ -f "${run_dir}/split_assignments.csv" ] || continue
    run_id=$(basename "$run_dir")
    seed=$(echo "$run_id" | sed -E 's/.*seed([0-9]+).*/\1/')
    rts=$(echo "$run_id" | sed -E 's/.*_rare([0-9]+|all)$/\1/')
    if echo "$run_id" | grep -q "asdc"; then
        run_ct "configs/immune_dc.yaml" "$seed" "ASDC" "$rts" "batch_heldout" "$run_dir"
    elif echo "$run_id" | grep -q "cdc1"; then
        run_ct "configs/immune_dc.yaml" "$seed" "cDC1" "$rts" "batch_heldout" "$run_dir"
    fi
done

# pancreas: epsilon, gamma
for run_dir in "$OUTPUTS"/pancreas/*/; do
    [ -f "${run_dir}/split_assignments.csv" ] || continue
    run_id=$(basename "$run_dir")
    seed=$(echo "$run_id" | sed -E 's/.*seed([0-9]+).*/\1/')
    rts=$(echo "$run_id" | sed -E 's/.*_rare([0-9]+|all)$/\1/')
    if echo "$run_id" | grep -q "epsilon"; then
        run_ct "configs/pancreas_epsilon.yaml" "$seed" "epsilon" "$rts" "batch_heldout" "$run_dir"
    elif echo "$run_id" | grep -q "gamma"; then
        run_ct "configs/pancreas_gamma.yaml" "$seed" "gamma" "$rts" "batch_heldout" "$run_dir"
    fi
done

# tabula_liver: non-classical monocyte
for run_dir in "$OUTPUTS"/tabula_liver/*/; do
    [ -f "${run_dir}/split_assignments.csv" ] || continue
    run_id=$(basename "$run_dir")
    seed=$(echo "$run_id" | sed -E 's/.*seed([0-9]+).*/\1/')
    rts=$(echo "$run_id" | sed -E 's/.*_rare([0-9]+|all)$/\1/')
    split_mode="cell_stratified"
    echo "$run_id" | grep -q "batch_heldout" && split_mode="batch_heldout"
    run_ct "configs/tabula_liver.yaml" "$seed" "non-classical monocyte" "$rts" "$split_mode" "$run_dir"
done

# tabula_pancreas: type B pancreatic cell
for run_dir in "$OUTPUTS"/tabula_pancreas/*/; do
    [ -f "${run_dir}/split_assignments.csv" ] || continue
    run_id=$(basename "$run_dir")
    seed=$(echo "$run_id" | sed -E 's/.*seed([0-9]+).*/\1/')
    rts=$(echo "$run_id" | sed -E 's/.*_rare([0-9]+|all)$/\1/')
    split_mode="cell_stratified"
    run_ct "configs/tabula_pancreas.yaml" "$seed" "type B pancreatic cell" "$rts" "$split_mode" "$run_dir"
done

echo ""
echo "All CellTypist runs attempted."
