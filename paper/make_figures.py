#!/usr/bin/env python3
"""Generate all paper figures. Run from cot-hurts-v2/ directory.

Usage: python paper/make_figures.py
Output: paper/figures/*.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mtick

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

STYLES = ["direct","cot","plan","decompose","expert","reviewer",
          "simple","careful","efficient","tdd","defensive","typed"]

STYLE_LABELS = {
    "direct": "Direct", "cot": "CoT", "plan": "Plan",
    "decompose": "Decompose", "expert": "Expert", "reviewer": "Reviewer",
    "simple": "Simple", "careful": "Careful", "efficient": "Efficient",
    "tdd": "TDD", "defensive": "Defensive", "typed": "Typed",
}

MODELS = [
    ("qwen_coder",          "Qwen Base"),
    ("qwen_coder_instruct",  "Qwen Instruct"),
    ("deepseek_coder",       "DeepSeek Base"),
    ("deepseek_coder_instruct", "DeepSeek Instruct"),
]

# Color palette (color-blind friendly)
CB = {
    "blue":   "#2166ac",
    "red":    "#d6604d",
    "green":  "#4dac26",
    "orange": "#f4a582",
    "purple": "#762a83",
    "gray":   "#878787",
}


# ---------------------------------------------------------------------------
# Figure 1: Layer-wise probe accuracy curves
# ---------------------------------------------------------------------------
def fig_tomography():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    configs = [
        ("qwen_coder",     "Qwen-1.5B Base", 28,
         [(f"results/qwen_coder/tomography_humaneval.csv", "HumanEval", CB["blue"]),
          (f"results/qwen_coder/tomography_mbpp.csv",      "MBPP",      CB["orange"])]),
        ("deepseek_coder", "DeepSeek-1.3B Base", 24,
         [(f"results/deepseek_coder/tomography_humaneval.csv", "HumanEval", CB["blue"]),
          (f"results/deepseek_coder/tomography_mbpp.csv",      "MBPP",      CB["orange"])]),
    ]

    for ax, (model_key, title, n_layers, files) in zip(axes, configs):
        for fpath, label, color in files:
            df = pd.read_csv(fpath)
            ax.plot(df["layer"], df["accuracy"] * 100,
                    color=color, linewidth=1.8, label=label, marker="o",
                    markersize=2.5, markevery=4)

        # 90% threshold
        ax.axhline(90, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(n_layers - 1, 91.5, "90%", fontsize=7, color="black", alpha=0.7, ha="right")

        # 12.5% depth marker
        probe_layer = max(1, n_layers // 8)
        ax.axvline(probe_layer, color=CB["red"], linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(probe_layer + 0.3, 15, f"L{probe_layer}\n(probe)", fontsize=6.5,
                color=CB["red"], alpha=0.85)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_xlim(0, n_layers - 1)
        ax.set_ylim(0, 103)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(loc="lower right", frameon=False)

    axes[0].set_ylabel("Probe Accuracy")
    fig.suptitle("Prompt-type information emerges in the first layer across all models",
                 fontsize=9, y=1.02)

    plt.savefig(f"{OUT}/fig1_tomography.pdf")
    plt.savefig(f"{OUT}/fig1_tomography.png")
    plt.close()
    print("✓ fig1_tomography")


# ---------------------------------------------------------------------------
# Figure 2: CoT effect reversal heatmap
# ---------------------------------------------------------------------------
def fig_cot_reversal():
    datasets = ["humaneval", "mbpp", "livecodebench"]
    ds_labels = {"humaneval": "HumanEval", "mbpp": "MBPP", "livecodebench": "LiveCB"}

    data = {}  # (model_key, dataset) -> delta
    pvals = {}
    for model_key, _ in MODELS:
        for d in datasets:
            try:
                df = pd.read_csv(f"results/{model_key}/generation_{d}.csv")
                delta = df["cot_pass"].mean() - df["direct_pass"].mean()
                b = int(((~df["direct_pass"]) & df["cot_pass"]).sum())
                c = int((df["direct_pass"] & ~df["cot_pass"]).sum())
                from scipy.stats import binomtest
                p = binomtest(b, b+c, 0.5).pvalue if b+c > 0 else 1.0
                data[(model_key, d)] = delta * 100
                pvals[(model_key, d)] = p
            except:
                data[(model_key, d)] = np.nan
                pvals[(model_key, d)] = 1.0

    model_labels = [label for _, label in MODELS]
    grid = np.array([[data[(mk, d)] for d in datasets] for mk, _ in MODELS])

    vmax = np.nanmax(np.abs(grid))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    im = ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds_labels[d] for d in datasets])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(model_labels)

    for i, (mk, _) in enumerate(MODELS):
        for j, d in enumerate(datasets):
            val = data[(mk, d)]
            p = pvals[(mk, d)]
            if np.isnan(val):
                continue
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:+.1f}%{sig}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if sig else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("CoT − Direct (pp)", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_title("CoT effect reverses with instruction tuning (** p<0.01)",
                 fontsize=9, fontweight="bold")
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_cot_reversal.pdf")
    plt.savefig(f"{OUT}/fig2_cot_reversal.png")
    plt.close()
    print("✓ fig2_cot_reversal")


# ---------------------------------------------------------------------------
# Figure 3: Strategy selector bar chart  (1×4, one panel per model)
# ---------------------------------------------------------------------------
def fig_strategy_selector():
    methods     = ["Direct", "CoT", "Probe", "Best Single", "Oracle"]
    colors      = [CB["gray"], CB["orange"], CB["blue"], CB["green"], CB["purple"]]
    ds_labels   = ["HumanEval", "MBPP"]
    datasets    = ["humaneval", "mbpp"]
    short_names = ["Qwen\nBase", "Qwen\nInstruct", "DeepSeek\nBase", "DeepSeek\nInstruct"]

    # data[model_idx][dataset_idx][method] = value
    data = []
    for mk, _ in MODELS:
        model_data = []
        for d in datasets:
            try:
                df = pd.read_csv(f"results/{mk}/strategy_selector_{d}_selection.csv")
                best_s = max(STYLES, key=lambda s: df[f"{s}_pass"].mean())
                model_data.append({
                    "Direct":      df["direct_pass"].mean() * 100,
                    "CoT":         df["cot_pass"].mean() * 100,
                    "Probe":       df["probe_selected_pass"].mean() * 100,
                    "Best Single": df[f"{best_s}_pass"].mean() * 100,
                    "Oracle":      df["oracle_pass"].mean() * 100,
                })
            except:
                model_data.append(None)
        data.append(model_data)

    n_methods = len(methods)
    n_ds      = len(datasets)
    group_w   = n_methods * 0.13 + 0.1   # width of one dataset group
    bar_w     = 0.13

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8))
    fig.subplots_adjust(wspace=0.38)

    for mi, ax in enumerate(axes):
        for di, (d_label, d_vals) in enumerate(zip(ds_labels, data[mi])):
            if d_vals is None:
                continue
            group_center = di * (group_w + 0.25)
            offsets = np.linspace(-(n_methods - 1) / 2,
                                   (n_methods - 1) / 2, n_methods) * bar_w
            for j, (method, color, offset) in enumerate(zip(methods, colors, offsets)):
                val = d_vals[method]
                bar = ax.bar(group_center + offset, val, width=bar_w * 0.9,
                             color=color, edgecolor="white", linewidth=0.4,
                             zorder=2)
                # Bold outline on Probe bar
                if method == "Probe":
                    bar[0].set_edgecolor(CB["blue"])
                    bar[0].set_linewidth(1.5)
                # Value label only on Probe + Oracle to avoid clutter
                if method in ("Probe", "Oracle"):
                    ax.text(group_center + offset,
                            val + 0.8, f"{val:.0f}",
                            ha="center", va="bottom", fontsize=5.5,
                            color="black" if method == "Oracle" else CB["blue"],
                            fontweight="bold" if method == "Probe" else "normal")

        ax.set_xticks([i * (group_w + 0.25) for i in range(n_ds)])
        ax.set_xticklabels(ds_labels, fontsize=8)
        ax.set_title(short_names[mi], fontsize=8.5, fontweight="bold", pad=4)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ymax = max(
            (d[m] for d in data[mi] if d is not None for m in ["Oracle"]),
            default=80
        )
        ax.set_ylim(0, min(100, ymax * 1.3))
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-group_w * 0.6, (n_ds - 1) * (group_w + 0.25) + group_w * 0.6)

    axes[0].set_ylabel("Pass@1")
    patches = [mpatches.Patch(color=c, label=m) for c, m in zip(colors, methods)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Probe-guided selection vs. baselines (Pass@1, held-out test set)",
                 fontsize=9, fontweight="bold", y=1.02)

    plt.savefig(f"{OUT}/fig3_strategy_selector.pdf")
    plt.savefig(f"{OUT}/fig3_strategy_selector.png")
    plt.close()
    print("✓ fig3_strategy_selector")


# ---------------------------------------------------------------------------
# Figure 4: Per-style pass rate heatmap
# ---------------------------------------------------------------------------
def fig_style_heatmap():
    # Average over humaneval + mbpp for each model
    grid = np.zeros((len(MODELS), len(STYLES)))
    for i, (mk, _) in enumerate(MODELS):
        rates = []
        for d in ["humaneval", "mbpp"]:
            try:
                df = pd.read_csv(f"results/{mk}/strategy_selector_{d}_selection.csv")
                rates.append([df[f"{s}_pass"].mean() for s in STYLES])
            except:
                pass
        if rates:
            grid[i] = np.mean(rates, axis=0)

    model_labels = [label for _, label in MODELS]
    style_labels = [STYLE_LABELS[s] for s in STYLES]

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    im = ax.imshow(grid * 100, cmap="YlOrRd", aspect="auto", vmin=0, vmax=grid.max()*100)

    ax.set_xticks(range(len(STYLES)))
    ax.set_xticklabels(style_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(model_labels, fontsize=8.5)

    for i in range(len(MODELS)):
        for j in range(len(STYLES)):
            val = grid[i, j] * 100
            color = "white" if val > grid.max() * 60 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5, color=color)

    # Highlight best style per model
    for i in range(len(MODELS)):
        best_j = np.argmax(grid[i])
        ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", linewidth=2.0))

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Pass@1 (%)", fontsize=8)
    ax.set_title("Per-style Pass@1 averaged over HumanEval + MBPP (best style boxed per model)",
                 fontsize=9, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_style_heatmap.pdf")
    plt.savefig(f"{OUT}/fig4_style_heatmap.png")
    plt.close()
    print("✓ fig4_style_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: Style variance (σ) vs. probe improvement — scatter plot
# Shows that the probe is most useful when style sensitivity is high
# ---------------------------------------------------------------------------
def fig_oracle_gap():
    short = {"qwen_coder": "Qwen\nBase", "qwen_coder_instruct": "Qwen\nInstruct",
             "deepseek_coder": "DeepSeek\nBase", "deepseek_coder_instruct": "DeepSeek\nInstruct"}
    marker_style = {"qwen_coder": "o", "qwen_coder_instruct": "s",
                    "deepseek_coder": "^", "deepseek_coder_instruct": "D"}
    model_colors = {"qwen_coder": CB["blue"], "qwen_coder_instruct": CB["red"],
                    "deepseek_coder": CB["green"], "deepseek_coder_instruct": CB["purple"]}

    records = []
    for mk, _ in MODELS:
        for d in ["humaneval", "mbpp"]:
            try:
                df = pd.read_csv(f"results/{mk}/strategy_selector_{d}_selection.csv")
                direct = df["direct_pass"].mean() * 100
                probe  = df["probe_selected_pass"].mean() * 100
                # σ = std of Pass@1 across 12 fixed styles
                style_means = [df[f"{s}_pass"].mean() * 100 for s in STYLES]
                sigma = float(np.std(style_means))
                ds_short = "HE" if d == "humaneval" else "MB"
                records.append({
                    "mk": mk, "ds": ds_short,
                    "sigma": sigma,
                    "probe_improvement": probe - direct,
                    "label": ds_short,
                })
            except:
                pass

    if not records:
        return

    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    # Plot each point
    seen_labels = set()
    for r in records:
        mk = r["mk"]
        label = short[mk].replace("\n", " ") if mk not in seen_labels else None
        seen_labels.add(mk)
        ax.scatter(r["sigma"], r["probe_improvement"],
                   color=model_colors[mk], marker=marker_style[mk],
                   s=55, zorder=3, label=label, edgecolors="white", linewidths=0.5)
        # Dataset label offset
        offset_x = 0.001
        offset_y = 0.3
        ax.text(r["sigma"] + offset_x, r["probe_improvement"] + offset_y,
                r["label"], fontsize=7, color=model_colors[mk], ha="left", va="bottom")

    # Reference line at y=0
    x_range = [min(r["sigma"] for r in records), max(r["sigma"] for r in records)]
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.4, zorder=1)
    ax.text(x_range[1] * 0.98, 0.4, "Probe = Direct", fontsize=7,
            color="gray", ha="right", va="bottom")

    # Shading: high-variance region
    ax.axvspan(0.04, x_range[1] * 1.05, alpha=0.06, color=CB["blue"],
               label="High variance (σ > 0.04)")

    ax.set_xlabel("Style Variance σ  (std of Pass@1 across 12 styles)", fontsize=9)
    ax.set_ylabel("Probe improvement over Direct (pp)", fontsize=9)
    ax.set_title("Probe is most useful when style choice matters most",
                 fontsize=9, fontweight="bold")

    # Legend: model markers only
    handles, labels = ax.get_legend_handles_labels()
    # Filter out the shaded region from main legend
    model_handles = [(h, l) for h, l in zip(handles, labels) if "High variance" not in l]
    shade_handles = [(h, l) for h, l in zip(handles, labels) if "High variance" in l]
    leg1 = ax.legend([h for h, _ in model_handles], [l for _, l in model_handles],
                     loc="upper left", fontsize=7.5, frameon=False,
                     handletextpad=0.4, borderpad=0.3)
    if shade_handles:
        ax.legend([h for h, _ in shade_handles], [l for _, l in shade_handles],
                  loc="lower right", fontsize=7.5, frameon=False)
        ax.add_artist(leg1)

    ax.set_xlim(max(0, x_range[0] - 0.003), x_range[1] + 0.006)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_oracle_gap.pdf")
    plt.savefig(f"{OUT}/fig5_oracle_gap.png")
    plt.close()
    print("✓ fig5_oracle_gap (scatter: σ vs probe improvement)")


if __name__ == "__main__":
    fig_tomography()
    fig_cot_reversal()
    fig_strategy_selector()
    fig_style_heatmap()
    fig_oracle_gap()
    print(f"\nAll figures saved to {OUT}/")
