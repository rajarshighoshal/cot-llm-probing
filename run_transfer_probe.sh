#!/bin/bash
# Cross-dataset transfer: train probe on HumanEval+MBPP, test on LiveCodeBench
# Demonstrates probe generalizes to unseen dataset distribution
# Reuses existing activation/generation caches for HumanEval and MBPP
# Usage: bash run_transfer_probe.sh

echo "=== Cross-Dataset Transfer: Train on HumanEval+MBPP → Test on LiveCodeBench ==="
echo "Started at: $(date)"
echo ""

for model in qwen_coder_instruct qwen_coder deepseek_coder_instruct deepseek_coder; do
    outfile="results/${model}/strategy_selector_livecodebench_transfer_selection.csv"
    if [ -f "$outfile" ]; then
        echo "SKIP: $model (already exists: $outfile)"
        echo ""
        continue
    fi
    echo "----------------------------------------------"
    echo "Running: $model (transfer: humaneval+mbpp → livecodebench)"
    echo "Time: $(date)"
    echo "----------------------------------------------"
    python -u -m experiments.strategy_selector \
        --model $model \
        --dataset livecodebench \
        --transfer_from humaneval,mbpp \
        --backend mlx \
        --probe_mode selection
    echo ""
done

echo "=== All done! ==="
echo "Finished at: $(date)"
