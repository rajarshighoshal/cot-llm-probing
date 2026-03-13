#!/bin/bash
# Run selection probe experiments (4 models x 2 datasets)
# Uses --probe_mode selection (direct activation → style prediction)
# Caches activations + generations for reuse across probe modes
# Usage: bash run_selection_probe.sh

echo "=== Selection Probe: Full Run (4 models x 2 datasets) ==="
echo "Started at: $(date)"
echo ""

for model in qwen_coder_instruct qwen_coder deepseek_coder_instruct deepseek_coder; do
    for dataset in humaneval mbpp; do
        outfile="results/${model}/strategy_selector_${dataset}_selection.csv"
        if [ -f "$outfile" ]; then
            echo "SKIP: $model / $dataset (already exists: $outfile)"
            echo ""
            continue
        fi
        echo "----------------------------------------------"
        echo "Running: $model / $dataset (selection probe, MLX)"
        echo "Time: $(date)"
        echo "----------------------------------------------"
        python -u -m experiments.strategy_selector \
            --model $model --dataset $dataset \
            --backend mlx --probe_mode selection
        echo ""
    done
done

echo "=== All done! ==="
echo "Finished at: $(date)"
