import re
import subprocess

import numpy as np


def extract_code(text):
    """Extract Python code from markdown code blocks, or return as-is."""
    # Try to find ```python ... ``` blocks
    matches = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text


def calculate_pass_at_k(n, c, k):
    """Standard Pass@k formula.

    n: total samples, c: correct samples, k: k-value.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _fix_indentation(code, prompt):
    """Fix indentation mismatch between prompt and generated code.

    Some backends (e.g. MLX) strip a leading space from the first token,
    producing 3-space indent when the prompt expects 4-space. Only the first
    line is affected — nested lines keep correct indentation.
    """
    if not code or not prompt:
        return code

    # Find expected indent from the prompt (last meaningful line)
    prompt_lines = prompt.rstrip().split("\n")
    expected_indent = ""
    for line in reversed(prompt_lines):
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            expected_indent = line[:len(line) - len(stripped)]
            break

    if not expected_indent:
        return code

    # Fix only the first non-empty line if its indent is short
    lines = code.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped:
            actual_spaces = len(line) - len(stripped)
            expected_spaces = len(expected_indent)
            if actual_spaces < expected_spaces:
                lines[i] = " " * (expected_spaces - actual_spaces) + line
            break

    return "\n".join(lines)


def check_correctness(generated_code, problem, timeout=5):
    """Execute generated code against test cases. Returns True if all pass."""
    # Extract code from markdown blocks if present (instruct models)
    code = extract_code(generated_code)

    # Fix indentation mismatch (e.g. MLX generating 3 spaces instead of 4)
    code = _fix_indentation(code, problem.get("prompt", ""))

    is_leetcode = problem.get("platform") == "leetcode"

    if is_leetcode:
        # LiveCodeBench LeetCode style: class Solution with method
        # Tests are like: assert str(Solution().method(...)) == str(expected)
        if "class Solution" in code:
            # Instruct model returned full class
            full_code = problem["prompt"].split("class Solution")[0] + code
        else:
            # Base model returned method body — prepend prompt
            full_code = problem["prompt"] + code
        full_code += "\n" + problem["test"]
    elif problem.get("entry_point"):
        # HumanEval style
        entry = problem["entry_point"]
        if f"def {entry}" in code:
            full_code = code + "\n" + problem["test"]
        else:
            full_code = problem["prompt"] + code + "\n" + problem["test"]
        full_code += f"\ncheck({entry})"
    else:
        # MBPP style
        if "def " in code:
            full_code = code + "\n" + problem["test"]
        else:
            full_code = problem["prompt"] + code + "\n" + problem["test"]

    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def mcnemar_test(results_a, results_b):
    """McNemar's test for paired binary outcomes.

    results_a, results_b: lists of booleans.
    Returns (chi2, p_value).
    """
    from scipy.stats import chi2 as chi2_dist

    b = sum(a and not bb for a, bb in zip(results_a, results_b))
    c = sum(not a and bb for a, bb in zip(results_a, results_b))

    if b + c == 0:
        return 0.0, 1.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return chi2, p_value
