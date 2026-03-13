import subprocess

import numpy as np


def calculate_pass_at_k(n, c, k):
    """Standard Pass@k formula.

    n: total samples, c: correct samples, k: k-value.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def check_correctness(generated_code, problem, timeout=5):
    """Execute generated code against test cases. Returns True if all pass."""
    if problem.get("entry_point"):
        # HumanEval style
        full_code = problem["prompt"] + generated_code + "\n" + problem["test"]
        full_code += f"\ncheck({problem['entry_point']})"
    else:
        # MBPP style
        full_code = generated_code + "\n" + problem["test"]

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
