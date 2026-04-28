#!/bin/bash
# Scale-validation runs on a rented GPU (RunPod / Lambda / Vast.ai).
# REQUIRES A100 80GB or H100 80GB for 14B-class fp16 inference.
#
# Setup once on the pod:
#   pip install -r requirements.txt
#   huggingface-cli login   # only if running Llama-3 (gated)
#
# Usage:
#   bash run_7b_scale.sh humaneval [tier]
#   bash run_7b_scale.sh mbpp [tier]
#
# tier:
#   minimal  — Qwen 14B Base+Instruct only (key 13B+ result)            [~1 hr]
#   core     — adds Qwen 7B Base+Instruct (scale curve)                 [~2.5 hr]
#   full     — adds DeepSeek 6.7B + Llama-3 8B (cross-family)           [~5 hr]
set -e

DATASET="${1:-humaneval}"
TIER="${2:-core}"

case "$TIER" in
    minimal)
        MODELS=(qwen_coder_14b qwen_coder_14b_instruct)
        ;;
    core)
        MODELS=(qwen_coder_14b qwen_coder_14b_instruct
                qwen_coder_7b qwen_coder_7b_instruct)
        ;;
    full)
        MODELS=(qwen_coder_14b qwen_coder_14b_instruct
                qwen_coder_7b qwen_coder_7b_instruct
                deepseek_coder_67b deepseek_coder_67b_instruct
                llama3_8b_instruct)
        ;;
    *)
        echo "Unknown tier: $TIER. Use minimal | core | full"
        exit 1
        ;;
esac

echo "Tier: $TIER  ->  ${#MODELS[@]} models on $DATASET"
for m in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  $m / $DATASET"
    echo "=============================================="
    python -m experiments.generation \
        --model "$m" \
        --dataset "$DATASET" \
        --backend pytorch \
        --styles direct cot
done

echo ""
echo "All scale runs complete: $DATASET / tier=$TIER"
echo "Results in: results/<model>/generation_${DATASET}.csv"
