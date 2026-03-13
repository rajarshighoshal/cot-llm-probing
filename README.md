# Think Less, Code Better

Code for the paper: *Think Less, Code Better: Probing When Chain-of-Thought Hurts and How to Route Around It*

## Key Findings

- **CoT reversal under instruction tuning:** CoT significantly improves Qwen2.5-Coder base (+13.4%, p<0.001) but significantly degrades the instruction-tuned variant (−15.2%, p<0.001) on the same architecture
- **Architecture-specific sensitivity:** DeepSeek-Coder is insensitive to CoT regardless of training regime; CodeLlama-7B base is also hurt (−6.7%, p=0.022)
- **Early encoding, divergent behavior:** All models encode prompt type by Layer 1–4 (>90% accuracy) — yet this universal encoding drives opposite behavioral outcomes depending on training regime
- **Probe-guided router:** A lightweight MLP probe selects from 12 prompt styles via a single 84ms forward pass, statistically matching the best fixed style in 7/8 settings

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Layer-wise probe accuracy
python -m experiments.tomography --model qwen_coder --dataset humaneval

# Direct vs CoT pass rate comparison
python -m experiments.generation --model qwen_coder_instruct --dataset humaneval

# Probe-guided style routing
python -m experiments.strategy_selector --model qwen_coder_instruct --dataset humaneval

# Run everything
bash run_all.sh
```

## Models

| Key | Model | Params | Type |
|-----|-------|--------|------|
| `qwen_coder` | Qwen2.5-Coder-1.5B | 1.5B | Base |
| `qwen_coder_instruct` | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Instruct |
| `deepseek_coder` | DeepSeek-Coder-1.3B | 1.3B | Base |
| `deepseek_coder_instruct` | DeepSeek-Coder-1.3B-Instruct | 1.3B | Instruct |
| `codellama` | CodeLlama-7B | 7B | Base |

All 1.3–1.5B models tested on Apple M4 Pro 24GB (MPS + MLX). CodeLlama-7B requires a GPU.

## Datasets

- **HumanEval** (164 problems)
- **MBPP** (500 problems)
- **LiveCodeBench** (381 problems, contamination-free)

## Project Structure

```
experiments/     tomography.py, generation.py, strategy_selector.py
data_loader/     HumanEval, MBPP, LiveCodeBench loaders
models/          Model registry and loader
metrics/         Pass@k, McNemar's test, execution eval
results/         Cached CSVs and activation caches
paper/           LaTeX source and figures
```

## Regenerate Figures

```bash
python paper/make_figures.py
```
