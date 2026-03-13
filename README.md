# Don't Make LLMs Overthink: When Chain-of-Thought Hurts Code Generation

Mechanistic interpretability study showing that Chain-of-Thought prompting degrades code generation accuracy in base (non-instruction-tuned) code LLMs.

## Key Findings

- CoT prompting degrades Pass@1 by 6.7% on base code models (p=0.02)
- Prompt-type information becomes linearly separable at layer 4 of 32 — very early
- Early emergence is architecture-agnostic across multiple model families
- A probe-guided strategy selector improves over both Direct and CoT baselines

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

Individual experiments:
```bash
# Phase A: Where do prompt-type signals emerge?
python -m experiments.tomography --model qwen_coder --dataset humaneval

# Phase B: Does CoT help or hurt?
python -m experiments.generation --model deepseek_coder --dataset mbpp

# Phase C: Can probes pick the best strategy?
python -m experiments.strategy_selector --model qwen_coder --dataset humaneval
```

Run everything:
```bash
bash run_all.sh
```

## Supported Models

| Key | Model | Params | Fits on M4 Pro |
|-----|-------|--------|----------------|
| `qwen_coder` | Qwen2.5-Coder-1.5B | 1.5B | Yes |
| `deepseek_coder` | DeepSeek-Coder-1.3B | 1.3B | Yes |
| `starcoder2_3b` | StarCoder2-3B | 3B | Yes |
| `phi2` | Phi-2 | 2.7B | Yes |
| `codellama` | CodeLlama-7B | 7B | Needs GPU |

## Datasets

- **HumanEval** (164 problems) — primary benchmark
- **MBPP** (500 problems) — secondary benchmark

## Project Structure

```
experiments/       Unified, device-agnostic experiment runners
data_loader/       Dataset loaders (HumanEval, MBPP)
models/            Model registry and universal loader
metrics/           Pass@k evaluation, McNemar's test
results/           Experiment outputs (CSVs)
paper/             LaTeX source
paper_figures/     Publication figures
```
