"""Unified experiment runners.

All experiments are device-agnostic (CUDA/MPS/CPU) and parameterized
via CLI arguments. No hardcoded paths.

Usage:
    python -m experiments.tomography --model qwen_coder --dataset humaneval
    python -m experiments.generation --model deepseek_coder --dataset mbpp
    python -m experiments.strategy_selector --model qwen_coder --dataset humaneval
"""

PROMPT_STYLES = {
    "direct": lambda p: p,
    "cot": lambda p: "# Let's think step by step\n" + p,
    "plan": lambda p: "# First, let's plan the approach, then implement\n" + p,
    "expert": lambda p: "# As an expert Python programmer\n" + p,
    "simple": lambda p: "# Keep it simple and direct\n" + p,
    "careful": lambda p: "# Be careful about edge cases\n" + p,
}
