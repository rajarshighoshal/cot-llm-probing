#!/bin/bash
# Re-run generation for all 4 main models on HumanEval + MBPP
# with token-count tracking for truncation analysis.
#
# Usage: bash run_truncation_analysis.sh
set -e

MODELS=(qwen_coder qwen_coder_instruct deepseek_coder deepseek_coder_instruct)
DATASETS=(humaneval mbpp)

source /Users/rajarshighoshal/miniconda3/etc/profile.d/conda.sh
conda activate cot-probing

# All HumanEval first (smaller, more critical for the table), then MBPP
for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        echo ""
        echo "=============================================="
        echo "  $m / $d"
        echo "=============================================="
        python -m experiments.generation \
            --model "$m" \
            --dataset "$d" \
            --backend mlx \
            --styles direct cot
    done
done

echo ""
echo "All runs complete."
