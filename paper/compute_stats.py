#!/usr/bin/env python3
"""
Compute all statistics for the paper:
  - McNemar's tests + Cohen's h + Wilson CIs for Direct vs CoT (Table 1)
  - McNemar's tests + Wilson CIs for Probe vs all 12 styles + baselines (Table 2)
  - Style variance table and probe ranking (Table 3)
  - Spearman correlation: style variance vs probe improvement

Usage: python paper/compute_stats.py
Outputs LaTeX snippets + console summary.
Run from cot-hurts-v2/ directory.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

RESULTS_DIR = "results"

MODELS = {
    "qwen_coder_instruct": "Qwen-1.5B Instruct",
    "qwen_coder":          "Qwen-1.5B Base",
    "deepseek_coder_instruct": "DeepSeek-1.3B Instruct",
    "deepseek_coder":      "DeepSeek-1.3B Base",
}
DATASETS = ["humaneval", "mbpp", "livecodebench"]
STYLES = ["direct", "cot", "plan", "decompose", "expert", "reviewer",
          "simple", "careful", "efficient", "tdd", "defensive", "typed"]


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    hw = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, max(0, center - hw), min(1, center + hw)


def mcnemar(a, b, continuity=True):
    """
    McNemar's test on two boolean arrays a, b.
    Returns (statistic, p_value, b_count, c_count)
    where b = |A=1,B=0| and c = |A=0,B=1|.
    """
    a = np.array(a, dtype=bool)
    b = np.array(b, dtype=bool)
    b_count = np.sum(a & ~b)   # A passes, B fails
    c_count = np.sum(~a & b)   # A fails, B passes
    n_disc = b_count + c_count
    if n_disc == 0:
        return 0.0, 1.0, 0, 0
    if continuity:
        stat = (abs(b_count - c_count) - 1)**2 / n_disc
    else:
        stat = (b_count - c_count)**2 / n_disc
    p = 1 - stats.chi2.cdf(stat, df=1)
    return stat, p, b_count, c_count


def cohen_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def fmt_p(p):
    """Format p-value for LaTeX."""
    if p < 0.001:
        return "$<$0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def fmt_ci(lo, hi):
    return f"[{lo:.1%}, {hi:.1%}]"


# ---------------------------------------------------------------------------
# Table 1: Direct vs CoT analysis (all problems from generation CSVs)
# ---------------------------------------------------------------------------

def analyze_generation():
    print("\n" + "="*70)
    print("TABLE 1: Direct vs CoT (McNemar's + Cohen's h + Wilson CI)")
    print("="*70)

    rows = []
    for model_key, model_name in MODELS.items():
        for dataset in DATASETS:
            path = os.path.join(RESULTS_DIR, model_key, f"generation_{dataset}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if "direct_pass" not in df.columns or "cot_pass" not in df.columns:
                continue

            n = len(df)
            direct = df["direct_pass"].astype(bool).values
            cot    = df["cot_pass"].astype(bool).values

            d_rate, d_lo, d_hi = wilson_ci(direct.sum(), n)
            c_rate, c_lo, c_hi = wilson_ci(cot.sum(), n)
            delta = c_rate - d_rate

            stat, p, b, c = mcnemar(cot, direct)  # cot vs direct
            h = cohen_h(c_rate, d_rate)
            h_label = ("large" if abs(h) > 0.8 else
                       "medium" if abs(h) > 0.5 else
                       "small" if abs(h) > 0.2 else "negligible")

            rows.append({
                "model": model_name, "dataset": dataset.upper(), "n": n,
                "direct": d_rate, "d_ci": fmt_ci(d_lo, d_hi),
                "cot": c_rate, "c_ci": fmt_ci(c_lo, c_hi),
                "delta": delta, "p": p, "p_fmt": fmt_p(p),
                "h": h, "h_label": h_label,
                "b": b, "c_disc": c,
            })

            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  {model_name:30s} {dataset.upper():15s} "
                  f"Direct={d_rate:.1%} CoT={c_rate:.1%} Δ={delta:+.1%} "
                  f"p={p:.4f}{sig}  h={h:+.2f}({h_label})")

    return rows


# ---------------------------------------------------------------------------
# Table 2 + 3: Probe analysis (strategy selector selection CSVs)
# ---------------------------------------------------------------------------

def analyze_probe():
    print("\n" + "="*70)
    print("TABLE 2/3: Probe vs Baselines (McNemar's + Wilson CI + Rankings)")
    print("="*70)

    variance_data = []   # for Spearman correlation
    all_rows = []

    for model_key, model_name in MODELS.items():
        for dataset in ["humaneval", "mbpp"]:
            path = os.path.join(RESULTS_DIR, model_key,
                                f"strategy_selector_{dataset}_selection.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            n = len(df)

            probe = df["probe_selected_pass"].astype(bool).values
            direct = df["direct_pass"].astype(bool).values
            cot   = df["cot_pass"].astype(bool).values
            rand  = df["random_selected_pass"].astype(bool).values
            oracle = df["oracle_pass"].astype(bool).values

            # Per-style pass rates
            style_rates = {s: df[f"{s}_pass"].astype(bool).mean() for s in STYLES}
            best_style = max(style_rates, key=style_rates.get)
            best_single = df[f"{best_style}_pass"].astype(bool).values

            # Style variance (std of 12 pass rates across styles)
            style_var = np.std(list(style_rates.values()))

            # Probe stats
            probe_rate, p_lo, p_hi = wilson_ci(probe.sum(), n)
            direct_rate = direct.mean()
            cot_rate    = cot.mean()
            oracle_rate = oracle.mean()
            best_rate   = best_single.mean()

            # McNemar's: probe vs each baseline
            _, p_vs_direct, _, _ = mcnemar(probe, direct)
            _, p_vs_cot,    _, _ = mcnemar(probe, cot)
            _, p_vs_best,   _, _ = mcnemar(probe, best_single)
            _, p_vs_rand,   _, _ = mcnemar(probe, rand)

            # Cohen's h: probe vs each baseline
            h_vs_cot    = cohen_h(probe_rate, cot_rate)
            h_vs_direct = cohen_h(probe_rate, direct_rate)

            # Probe ranking among 12 styles
            all_rates = sorted(style_rates.values(), reverse=True)
            # Find where probe_rate falls in the ranking
            probe_rank = sum(1 for r in all_rates if r > probe_rate) + 1

            # Gap recovery: (probe - direct) / (oracle - direct)
            gap = oracle_rate - direct_rate
            gap_recovery = (probe_rate - direct_rate) / gap if gap > 0 else 0.0

            variance_data.append({
                "model": model_name, "dataset": dataset, "n": n,
                "style_var": style_var, "probe_improvement": probe_rate - direct_rate,
                "probe_rate": probe_rate, "direct_rate": direct_rate,
                "cot_rate": cot_rate, "oracle_rate": oracle_rate,
                "best_rate": best_rate, "best_style": best_style,
                "probe_rank": probe_rank, "gap_recovery": gap_recovery,
                "p_vs_cot": p_vs_cot, "p_vs_direct": p_vs_direct,
                "p_vs_best": p_vs_best,
                "h_vs_cot": h_vs_cot, "h_vs_direct": h_vs_direct,
                "p_lo": p_lo, "p_hi": p_hi,
            })

            print(f"\n  {model_name} / {dataset.upper()} (n={n})")
            print(f"    Style variance σ = {style_var:.4f}  |  Best style: {best_style} ({best_rate:.1%})")
            print(f"    Probe rank among 12 styles: {probe_rank}/12")
            print(f"    Pass rates: Probe={probe_rate:.1%} 95%CI{fmt_ci(p_lo, p_hi)}")
            print(f"                CoT={cot_rate:.1%}  Direct={direct_rate:.1%}  "
                  f"Best={best_rate:.1%}  Oracle={oracle_rate:.1%}")
            print(f"    McNemar's: probe vs CoT    p={p_vs_cot:.4f}  h={h_vs_cot:+.3f}")
            print(f"    McNemar's: probe vs Direct p={p_vs_direct:.4f}  h={h_vs_direct:+.3f}")
            print(f"    McNemar's: probe vs Best   p={p_vs_best:.4f}  (ns → statistically tied)")
            print(f"    Gap recovery: {gap_recovery:.1%} of (Oracle - Direct)")

            all_rows.append(variance_data[-1])

    return variance_data


# ---------------------------------------------------------------------------
# Spearman correlation: style variance vs probe improvement
# ---------------------------------------------------------------------------

def analyze_variance_correlation(variance_data):
    print("\n" + "="*70)
    print("CORRELATION: Style Variance vs Probe Improvement")
    print("="*70)

    variances = [d["style_var"] for d in variance_data]
    improvements = [d["probe_improvement"] for d in variance_data]

    rho, p_rho = spearmanr(variances, improvements)
    print(f"  Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")
    print(f"  {'Significant positive correlation' if p_rho < 0.05 and rho > 0 else 'Not significant'}")
    print(f"  Interpretation: {'Higher style variance → greater probe benefit' if rho > 0 else 'No clear trend'}")

    for d in variance_data:
        print(f"    {d['model']:30s} {d['dataset'].upper():8s}  "
              f"σ={d['style_var']:.4f}  Δ={d['probe_improvement']:+.3f}")

    return rho, p_rho


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def print_latex_table1(gen_rows):
    print("\n" + "="*70)
    print("LATEX: Table 1 with Cohen's h")
    print("="*70)
    print(r"""\begin{table}[h]
\centering
\caption{\textbf{Direct vs.\ CoT Pass@1.} $\Delta$=CoT$-$Direct. Bold: $p{<}0.05$. $h$=Cohen's h effect size.}
\label{tab:generation}
\small
\begin{tabular}{llcccccl}
\toprule
\textbf{Model} & \textbf{Dataset} & \textbf{Direct} & \textbf{CoT} & \textbf{$\Delta$} & \textbf{$p$-value} & \textbf{$h$} & \textbf{Effect} \\
\midrule""")
    cur_model = None
    for r in gen_rows:
        model_cell = r["model"] if r["model"] != cur_model else ""
        cur_model = r["model"]
        bold = r["p"] < 0.05
        delta_str = f"\\textbf{{{r['delta']:+.1%}}}" if bold else f"{r['delta']:+.1%}"
        p_str = f"\\textbf{{{r['p_fmt']}}}" if bold else r['p_fmt']
        h_str = f"{r['h']:+.2f}"
        print(f"  {model_cell} & {r['dataset']} & {r['direct']:.1%} & {r['cot']:.1%} "
              f"& {delta_str} & {p_str} & {h_str} & {r['h_label']} \\\\")
        if r["dataset"] in ("MBPP", "LIVECODEBENCH"):
            print(r"  \midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def print_latex_table2(variance_data):
    print("\n" + "="*70)
    print("LATEX: Table 2 — Probe results with CIs and McNemar p-values")
    print("="*70)
    print(r"""\begin{table}[h]
\centering
\caption{\textbf{Probe-guided style routing} (Pass@1, test split). 95\% Wilson CI shown for Probe.
  $p_{\text{CoT}}$/$p_{\text{Dir}}$ = McNemar's test probe vs.\ CoT/Direct.
  $p_{\text{Best}}$ = McNemar's probe vs.\ best single style (n.s.\ = statistically tied).}
\label{tab:strategy}
\small
\begin{tabular}{llcccccccc}
\toprule
\textbf{Model} & \textbf{DS} & \textbf{Probe (95\% CI)} & \textbf{CoT} & \textbf{Direct} & \textbf{Best} & \textbf{Oracle} & $p_{\text{CoT}}$ & $p_{\text{Dir}}$ & $p_{\text{Best}}$ \\
\midrule""")
    cur_model = None
    for d in variance_data:
        mc = d["model"] if d["model"] != cur_model else ""
        cur_model = d["model"]
        ci_str = f"{d['probe_rate']:.1%} [{d['p_lo']:.1%},{d['p_hi']:.1%}]"
        print(f"  {mc} & {d['dataset'].upper()[:2]} & {ci_str} & "
              f"{d['cot_rate']:.1%} & {d['direct_rate']:.1%} & "
              f"{d['best_rate']:.1%} & {d['oracle_rate']:.1%} & "
              f"{fmt_p(d['p_vs_cot'])} & {fmt_p(d['p_vs_direct'])} & "
              f"{fmt_p(d['p_vs_best'])} \\\\")
        if d["dataset"] == "mbpp":
            print(r"  \midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def print_latex_variance_table(variance_data):
    print("\n" + "="*70)
    print("LATEX: Table 3 — Style variance and probe ranking")
    print("="*70)
    print(r"""\begin{table}[h]
\centering
\caption{\textbf{Style sensitivity characterizes probe utility.}
  $\sigma$ = std of Pass@1 across 12 styles. Rank = probe placement among 12 styles.
  $\Delta_\text{probe}$ = probe minus Direct. Gap\% = fraction of Oracle$-$Direct gap recovered.}
\label{tab:variance}
\small
\begin{tabular}{llcccccc}
\toprule
\textbf{Model} & \textbf{DS} & $\sigma$ & \textbf{Rank/12} & \textbf{Best Style} & \textbf{Probe} & \textbf{Direct} & \textbf{Gap\%} \\
\midrule""")
    cur_model = None
    for d in variance_data:
        mc = d["model"] if d["model"] != cur_model else ""
        cur_model = d["model"]
        print(f"  {mc} & {d['dataset'].upper()[:2]} & {d['style_var']:.3f} & "
              f"{d['probe_rank']}/12 & "
              f"{d['best_style']}({d['best_rate']:.1%}) & "
              f"{d['probe_rate']:.1%} & {d['direct_rate']:.1%} & "
              f"{d['gap_recovery']:.0%} \\\\")
        if d["dataset"] == "mbpp":
            print(r"  \midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    gen_rows = analyze_generation()
    variance_data = analyze_probe()
    rho, p_rho = analyze_variance_correlation(variance_data)

    print_latex_table1(gen_rows)
    print_latex_table2(variance_data)
    print_latex_variance_table(variance_data)

    print("\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)
    sig_gen = [(r["model"], r["dataset"], r["delta"], r["p"])
               for r in gen_rows if r["p"] < 0.05]
    print(f"  Significant Direct vs CoT comparisons: {len(sig_gen)}/{len(gen_rows)}")
    for m, d, delta, p in sig_gen:
        print(f"    {m} / {d}: Δ={delta:+.1%}, p={p:.4f}")

    probe_beats_cot = sum(1 for d in variance_data if d["probe_rate"] > d["cot_rate"])
    probe_beats_direct = sum(1 for d in variance_data if d["probe_rate"] >= d["direct_rate"])
    print(f"\n  Probe > CoT: {probe_beats_cot}/{len(variance_data)}")
    print(f"  Probe ≥ Direct: {probe_beats_direct}/{len(variance_data)}")
    print(f"\n  Spearman ρ(σ, probe_improvement) = {rho:.3f}, p={p_rho:.4f}")
    avg_rank = np.mean([d["probe_rank"] for d in variance_data])
    print(f"  Mean probe rank among 12 styles: {avg_rank:.1f}/12")
    high_var = [d for d in variance_data if d["style_var"] > 0.05]
    if high_var:
        avg_rank_hv = np.mean([d["probe_rank"] for d in high_var])
        print(f"  Mean probe rank (high-variance settings only, σ>0.05): {avg_rank_hv:.1f}/12")
