# Think Less, Code Better

Code for the paper *Think Less, Code Better: Probing When Chain-of-Thought Hurts and How to Route Around It*.

**Accepted at:**
- ICLR 2026 Workshop on Logical Reasoning of LLMs (non-archival)
- ACL 2026 Student Research Workshop (archival)

## Key Findings

- **CoT reversal under instruction tuning:** CoT improves Qwen2.5-Coder base (+13.4%, p<0.001) but degrades the instruction-tuned variant (−15.2%, p<0.001) on the same architecture
- **Architecture-specific:** DeepSeek-Coder is insensitive to CoT regardless of training regime; CodeLlama-7B base is also hurt (−6.7%, p=0.022)
- **Early encoding, divergent behavior:** All models encode prompt type by Layer 1–4 (>90% accuracy) — yet this universal encoding drives opposite behavioral outcomes
- **Output truncation explains part of the gap:** Qwen Instruct under CoT generates +112 tokens on average and hits the 512-token cap 7.6× more often; DeepSeek Instruct's output distribution barely shifts (+2 tokens), explaining its CoT-insensitivity
- **Probe-guided router:** A lightweight MLP probe selects from 12 prompt styles via a single 84ms forward pass, statistically matching the best fixed style in 7/8 settings

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Layer-wise probe accuracy (where does prompt-type info emerge?)
python -m experiments.tomography --model qwen_coder --dataset humaneval

# Direct vs CoT Pass@1 comparison (with token-count + truncation tracking)
python -m experiments.generation --model qwen_coder_instruct --dataset humaneval --backend mlx

# Probe-guided style routing across 12 prompt styles
python -m experiments.strategy_selector --model qwen_coder_instruct --dataset humaneval

# Truncation analysis (after running generation for all models)
python paper/compute_truncation.py
```

Bulk runners:
```bash
bash run_all.sh                 # tomography + generation on all small models
bash run_truncation_analysis.sh # generation with token/truncation tracking
bash run_7b_scale.sh humaneval  # 7B/14B scale validation (requires 24GB+ GPU)
```

See `RUNPOD_SETUP.md` for using a rented GPU.

## Models

| Key | HuggingFace ID | Params | Type |
|-----|----------------|--------|------|
| `qwen_coder` | Qwen/Qwen2.5-Coder-1.5B | 1.5B | Base |
| `qwen_coder_instruct` | Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B | Instruct |
| `qwen_coder_7b` | Qwen/Qwen2.5-Coder-7B | 7B | Base |
| `qwen_coder_7b_instruct` | Qwen/Qwen2.5-Coder-7B-Instruct | 7B | Instruct |
| `qwen_coder_14b` | Qwen/Qwen2.5-Coder-14B | 14B | Base |
| `qwen_coder_14b_instruct` | Qwen/Qwen2.5-Coder-14B-Instruct | 14B | Instruct |
| `deepseek_coder` | deepseek-ai/deepseek-coder-1.3b-base | 1.3B | Base |
| `deepseek_coder_instruct` | deepseek-ai/deepseek-coder-1.3b-instruct | 1.3B | Instruct |
| `deepseek_coder_67b` | deepseek-ai/deepseek-coder-6.7b-base | 6.7B | Base |
| `deepseek_coder_67b_instruct` | deepseek-ai/deepseek-coder-6.7b-instruct | 6.7B | Instruct |
| `codellama` | codellama/CodeLlama-7b-hf | 7B | Base |

1.3–1.5B models tested on Apple M4 Pro 24GB (MPS + MLX). 7B+ require ~24GB VRAM; 13B+ require A100 80GB.

## Datasets

- **HumanEval** (164 problems) — Chen et al., 2021
- **MBPP** (500 problems) — Austin et al., 2021
- **LiveCodeBench** (381 problems) — contamination-free, LeetCode-sourced

## Project Structure

```
experiments/    tomography.py, generation.py, strategy_selector.py
data_loader/    HumanEval, MBPP, LiveCodeBench loaders
models/         Registry + universal HF loader (output_hidden_states)
metrics/        Pass@k, McNemar's test, execution-based eval
results/        Cached CSVs and activation caches
paper/          LaTeX source, figures, analysis scripts (compute_truncation.py)
```

## Reproduce Figures

```bash
python paper/make_figures.py
```

Outputs go to `paper/figures/`.

## Citation

Bibtex will be added once camera-ready is published in the ACL Anthology.
