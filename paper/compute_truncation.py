#!/usr/bin/env python3
"""Compute truncation statistics for CoT vs Direct from generation CSVs.

Outputs a LaTeX-formatted table with:
- Average token count per condition
- Truncation rate (% hitting 512-token cap without complete code block)
- Pass@1 conditional on truncation status

Usage: python paper/compute_truncation.py
"""

import os
import pandas as pd
import numpy as np

MODELS = [
    ("qwen_coder",              "Qwen Base"),
    ("qwen_coder_instruct",     "Qwen Instruct"),
    ("deepseek_coder",          "DeepSeek Base"),
    ("deepseek_coder_instruct", "DeepSeek Instruct"),
]
DATASETS = [("humaneval", "HE"), ("mbpp", "MB")]


def analyse(df):
    """Return summary stats for one CSV."""
    out = {}
    for style in ["direct", "cot"]:
        n_tok_col = f"{style}_n_tokens"
        trunc_col = f"{style}_truncated"
        pass_col = f"{style}_pass"
        if n_tok_col not in df.columns:
            return None
        out[f"{style}_mean_tok"] = df[n_tok_col].mean()
        out[f"{style}_median_tok"] = df[n_tok_col].median()
        out[f"{style}_trunc_rate"] = df[trunc_col].mean() * 100
        out[f"{style}_pass"] = df[pass_col].mean() * 100
        # Pass rate among truncated and non-truncated
        if df[trunc_col].sum() > 0:
            out[f"{style}_pass_when_trunc"] = df.loc[df[trunc_col], pass_col].mean() * 100
        else:
            out[f"{style}_pass_when_trunc"] = float('nan')
        if (~df[trunc_col]).sum() > 0:
            out[f"{style}_pass_when_clean"] = df.loc[~df[trunc_col], pass_col].mean() * 100
        else:
            out[f"{style}_pass_when_clean"] = float('nan')
    return out


def main():
    rows = []
    for mk, label in MODELS:
        for d, ds in DATASETS:
            path = f"results/{mk}/generation_{d}.csv"
            if not os.path.exists(path):
                print(f"[skip] {path} not found")
                continue
            df = pd.read_csv(path)
            stats = analyse(df)
            if stats is None:
                print(f"[skip] {path} missing token-count columns (re-run needed)")
                continue
            rows.append({"model": label, "ds": ds, **stats})

    if not rows:
        print("No data yet. Run experiments/generation.py first.")
        return

    print()
    print("=" * 110)
    print("TRUNCATION STATISTICS")
    print("=" * 110)
    print(f"{'Model':18s} {'DS':4s} | {'Direct mean tok':>15s} {'D trunc%':>9s} {'D pass%':>8s}"
          f" | {'CoT mean tok':>13s} {'C trunc%':>9s} {'C pass%':>8s}"
          f" | {'Δ tok':>7s} {'Δ trunc%':>9s}")
    print("-" * 110)
    for r in rows:
        delta_tok = r["cot_mean_tok"] - r["direct_mean_tok"]
        delta_trunc = r["cot_trunc_rate"] - r["direct_trunc_rate"]
        print(f"{r['model']:18s} {r['ds']:4s} | "
              f"{r['direct_mean_tok']:>15.1f} {r['direct_trunc_rate']:>9.1f} {r['direct_pass']:>8.1f}"
              f" | {r['cot_mean_tok']:>13.1f} {r['cot_trunc_rate']:>9.1f} {r['cot_pass']:>8.1f}"
              f" | {delta_tok:>+7.1f} {delta_trunc:>+9.1f}")

    # ------------------------------------------------------------------
    # LaTeX output
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("LATEX TABLE")
    print("=" * 110)
    print(r"""\begin{table*}[t]
    \centering
    \caption{\textbf{Output length and truncation rates by prompt style.} Direct/CoT mean token counts and percentage of generations hitting the 512-token cap without producing a complete code block. $\Delta$ columns show CoT $-$ Direct differences. Truncation increases under CoT for instruction-tuned models, consistent with the hypothesis that CoT induces verbose reasoning that crowds out code generation. HE=HumanEval, MB=MBPP.}
    \label{tab:truncation}
    \small
    \begin{tabular}{llcccccccc}
        \toprule
        & & \multicolumn{2}{c}{\textbf{Direct}} & \multicolumn{2}{c}{\textbf{CoT}} & \multicolumn{2}{c}{\textbf{$\Delta$ (CoT $-$ Direct)}} \\
        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
        \textbf{Model} & \textbf{DS} & \textbf{Mean tok} & \textbf{Trunc \%} & \textbf{Mean tok} & \textbf{Trunc \%} & \textbf{$\Delta$ tok} & \textbf{$\Delta$ trunc} \\
        \midrule""")
    for i, r in enumerate(rows):
        delta_tok = r["cot_mean_tok"] - r["direct_mean_tok"]
        delta_trunc = r["cot_trunc_rate"] - r["direct_trunc_rate"]
        sign_tok = "+" if delta_tok >= 0 else ""
        sign_trunc = "+" if delta_trunc >= 0 else ""
        print(f"        {r['model']} & {r['ds']} & "
              f"{r['direct_mean_tok']:.0f} & {r['direct_trunc_rate']:.0f} & "
              f"{r['cot_mean_tok']:.0f} & {r['cot_trunc_rate']:.0f} & "
              f"{sign_tok}{delta_tok:.0f} & {sign_trunc}{delta_trunc:.0f} \\\\")
        # Add midrule between models (every 2 rows for HE+MB)
        if (i + 1) % 2 == 0 and (i + 1) < len(rows):
            print(r"        \midrule")
    print(r"""        \bottomrule
    \end{tabular}
\end{table*}""")

    # ------------------------------------------------------------------
    # Pass@1 split by truncation status
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("PASS@1 CONDITIONAL ON TRUNCATION (does truncation explain failure?)")
    print("=" * 110)
    print(f"{'Model':18s} {'DS':4s} | {'D pass-clean%':>14s} {'D pass-trunc%':>14s}"
          f" | {'C pass-clean%':>14s} {'C pass-trunc%':>14s}")
    print("-" * 110)
    for r in rows:
        print(f"{r['model']:18s} {r['ds']:4s} | "
              f"{r['direct_pass_when_clean']:>14.1f} {r['direct_pass_when_trunc']:>14.1f}"
              f" | {r['cot_pass_when_clean']:>14.1f} {r['cot_pass_when_trunc']:>14.1f}")


if __name__ == "__main__":
    main()
