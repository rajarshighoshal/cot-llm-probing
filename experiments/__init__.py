"""Unified experiment runners.

All experiments are device-agnostic (CUDA/MPS/CPU) and parameterized
via CLI arguments. No hardcoded paths.

Usage:
    python -m experiments.tomography --model qwen_coder --dataset humaneval
    python -m experiments.generation --model deepseek_coder --dataset mbpp
    python -m experiments.strategy_selector --model qwen_coder --dataset humaneval
"""

PROMPT_STYLES = {
    # Minimal
    "direct": lambda p: p,
    # Reasoning
    "cot": lambda p: "# Let's think step by step\n" + p,
    "plan": lambda p: "# First, let's plan the approach, then implement\n" + p,
    "decompose": lambda p: "# Break this into smaller sub-problems\n" + p,
    # Persona
    "expert": lambda p: "# As an expert Python programmer\n" + p,
    "reviewer": lambda p: "# Write clean, production-ready code\n" + p,
    # Constraint
    "simple": lambda p: "# Keep it simple and direct\n" + p,
    "careful": lambda p: "# Be careful about edge cases\n" + p,
    "efficient": lambda p: "# Use an efficient algorithm\n" + p,
    # Test-aware
    "tdd": lambda p: "# Write code that passes all the test cases\n" + p,
    "defensive": lambda p: "# Handle all possible inputs correctly\n" + p,
    # Structure
    "typed": lambda p: "# Use proper type hints and clear variable names\n" + p,
}
