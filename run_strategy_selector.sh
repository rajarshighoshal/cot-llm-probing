#!/bin/bash
# Run all strategy selector experiments (all 4 models x 2 datasets)
# Skips runs that already have results CSVs
# All use MLX backend (deepseek base uses locally converted safetensors)
# Usage: bash run_strategy_selector.sh

echo "=== Strategy Selector: Full Run (4 models x 2 datasets) ==="
echo "Started at: $(date)"
echo ""

for model in qwen_coder_instruct qwen_coder deepseek_coder_instruct deepseek_coder; do
    for dataset in humaneval mbpp; do
        outfile="results/${model}/strategy_selector_${dataset}.csv"
        if [ -f "$outfile" ]; then
            echo "SKIP: $model / $dataset (already exists: $outfile)"
            echo ""
            continue
        fi
        echo "----------------------------------------------"
        echo "Running: $model / $dataset (MLX)"
        echo "Time: $(date)"
        echo "----------------------------------------------"
        python -u -m experiments.strategy_selector --model $model --dataset $dataset --backend mlx
        echo ""
    done
done

echo "=== All done! ==="
echo "Finished at: $(date)"
