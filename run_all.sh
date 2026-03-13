#!/bin/bash
# Run all experiments for the workshop resubmission.
# Execute from repo root: bash run_all.sh
#
# Estimated time on M4 Pro 24GB:
#   Tomography:        ~20 min per model x dataset
#   Generation:        ~60 min per model x dataset
#   Strategy Selector: ~120 min per model x dataset (6 styles per problem)
#
# You can run individual experiments:
#   python -m experiments.tomography --model qwen_coder --dataset humaneval
#   python -m experiments.generation --model deepseek_coder --dataset mbpp

set -e

MODELS="qwen_coder deepseek_coder"
DATASETS="humaneval mbpp"

echo "=========================================="
echo "Phase A: Tomography"
echo "=========================================="
for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        echo ">>> ${MODEL} x ${DATASET}"
        python -m experiments.tomography --model "$MODEL" --dataset "$DATASET" --samples 50
    done
done

echo ""
echo "=========================================="
echo "Phase B: Direct vs CoT Generation"
echo "=========================================="
for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        echo ">>> ${MODEL} x ${DATASET}"
        python -m experiments.generation --model "$MODEL" --dataset "$DATASET"
    done
done

echo ""
echo "=========================================="
echo "Phase C: Strategy Selector (on humaneval only)"
echo "=========================================="
for MODEL in $MODELS; do
    echo ">>> ${MODEL} x humaneval"
    python -m experiments.strategy_selector --model "$MODEL" --dataset humaneval
done

echo ""
echo "All experiments complete."
