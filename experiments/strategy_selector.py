#!/usr/bin/env python3
"""
Phase C: Probe-Guided Strategy Selector

For each problem, extracts activations under all prompt styles,
uses a trained probe to predict which style will succeed, and selects
the best one. Compares against baselines (Direct, CoT, Random, Oracle).

Restructured for efficiency: collects all activations first (PyTorch),
then generates all code (MLX or PyTorch), then trains probe and evaluates.

Supports MLX backend for ~17x faster generation on Apple Silicon.
Supports resume via activation and generation caches.

Usage:
    python -m experiments.strategy_selector --model qwen_coder --dataset humaneval --backend mlx
    python -m experiments.strategy_selector --model qwen_coder --dataset mbpp --backend mlx
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_dataset
from experiments import PROMPT_STYLES
from experiments.generation import load_for_generation, generate_code, cleanup_gen
from metrics.evaluator import check_correctness
from models import MODEL_REGISTRY
from models.loader import (
    get_device, load_model, get_last_token_activation, cleanup, format_prompt,
)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Linear(d_model, 2)

    def forward(self, x):
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class SelectionProbe(nn.Module):
    """Predicts which style(s) will pass from a single problem activation.

    Input: direct-prompt activation (d_model,)
    Output: per-style logits (n_styles,)
    Trained with BCE loss (multi-label: multiple styles can pass).
    """
    def __init__(self, d_model, n_styles, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_styles),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(activations, labels, probe_type="linear", d_model=None,
                n_epochs=200, lr=1e-3):
    """Train pass/fail probe. Returns trained probe and accuracy."""
    X = torch.stack(activations)
    y = torch.tensor(labels).long()

    if d_model is None:
        d_model = X.shape[1]

    if probe_type == "mlp":
        probe = MLPProbe(d_model)
        n_epochs = 300
    else:
        probe = LinearProbe(d_model)

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(probe(X), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (probe(X).argmax(1) == y).float().mean().item()

    return probe, acc


def train_selection_probe(activations, labels_multi, n_styles, d_model=None,
                          n_epochs=500, lr=1e-3):
    """Train selection probe: problem activation → which styles pass.

    activations: list of tensors (one per problem, from direct prompt)
    labels_multi: list of lists, each inner list has 1/0 per style
    Returns trained probe and per-style accuracy.
    """
    X = torch.stack(activations)
    y = torch.tensor(labels_multi).float()

    if d_model is None:
        d_model = X.shape[1]

    probe = SelectionProbe(d_model, n_styles)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(probe(X), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(probe(X)) > 0.5).float()
        acc = (preds == y).float().mean().item()

    return probe, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase C: Probe-Guided Strategy Selector")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp", "livecodebench"])
    parser.add_argument("--probe_type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--probe_mode", default="selection", choices=["passfail", "selection"],
                        help="passfail: per-style pass/fail probe (old). "
                             "selection: predict best style from direct activation (new)")
    parser.add_argument("--probe_layer", type=int, default=None,
                        help="Layer to extract activations from (default: ~12.5%% depth)")
    parser.add_argument("--train_split", type=int, default=None,
                        help="Number of problems for probe training (rest = eval)")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "mlx"],
                        help="Generation backend (mlx is ~17x faster on Apple Silicon)")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    styles = list(PROMPT_STYLES.keys())

    # Load dataset
    ds = get_dataset(args.dataset)
    problems = ds.get_problems()

    # Train/test split
    if args.train_split is None:
        args.train_split = len(problems) // 2
    train_problems = problems[:args.train_split]
    test_problems = problems[args.train_split:]

    # Output/cache paths
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    suffix_parts = []
    if args.probe_mode != "passfail":
        suffix_parts.append(args.probe_mode)
    if args.probe_type != "linear":
        suffix_parts.append(args.probe_type)
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_path = os.path.join(out_dir, f"strategy_selector_{args.dataset}{suffix}.csv")
    # Caches are shared across probe modes (same data regardless of probe type)
    act_cache = os.path.join(out_dir, f"_act_cache_{args.dataset}.pt")
    gen_cache = os.path.join(out_dir, f"_gen_cache_{args.dataset}.json")

    # -------------------------------------------------------------------
    # Step 1: Collect activations (always PyTorch — needs hidden states)
    # -------------------------------------------------------------------
    if os.path.exists(act_cache):
        print("--- Step 1: Loading cached activations ---")
        cached = torch.load(act_cache, weights_only=False)
        all_activations = cached["activations"]
        meta = cached["meta"]
        probe_layer = cached["probe_layer"]
        print(f"  Loaded {len(all_activations)} cached activations")
        model_pt = None
    else:
        print("--- Step 1: Collecting activations (PyTorch) ---")
        device = get_device()
        model_pt, tokenizer_pt, meta = load_model(args.model, device)
        probe_layer = args.probe_layer or max(1, meta["num_layers"] // 8)

        all_activations = {}
        for i, prob in enumerate(tqdm(problems, desc="Activations")):
            for style in styles:
                prompt = PROMPT_STYLES[style](prob["prompt"])
                prompt_fmt = format_prompt(prompt, tokenizer_pt, meta)
                act = get_last_token_activation(
                    model_pt, tokenizer_pt, prompt_fmt, probe_layer, device
                )
                all_activations[(str(prob["id"]), style)] = act
            cleanup(device)

            if (i + 1) % 20 == 0:
                torch.save({"activations": all_activations, "meta": meta,
                            "probe_layer": probe_layer}, act_cache)

        torch.save({"activations": all_activations, "meta": meta,
                    "probe_layer": probe_layer}, act_cache)

    print(f"  Probe layer: {probe_layer}")
    print(f"  Probe type: {args.probe_type}")
    print(f"  Prompt library: {len(styles)} styles")
    print(f"  Train: {len(train_problems)}, Test: {len(test_problems)}")

    # -------------------------------------------------------------------
    # Step 2: Generate code for all problems x styles
    # -------------------------------------------------------------------
    all_gen_results = {}
    if os.path.exists(gen_cache):
        with open(gen_cache) as f:
            all_gen_results = json.load(f)
        print(f"--- Step 2: Loaded {len(all_gen_results)} cached generations ---")

    # Find problems still needing generation
    done_ids = set()
    for key in all_gen_results:
        pid = key.rsplit("|", 1)[0]
        if all(f"{pid}|{s}" in all_gen_results for s in styles):
            done_ids.add(pid)
    remaining = [p for p in problems if str(p["id"]) not in done_ids]

    if remaining:
        print(f"--- Step 2: Generating {len(remaining)} problems ({args.backend}) ---")

        # Free PyTorch model before loading MLX
        if args.backend == "mlx" and model_pt is not None:
            del model_pt, tokenizer_pt
            model_pt = None
            cleanup(device)

        # Load generation model
        if args.backend == "mlx":
            model_gen, tok_gen, meta_gen, gen_device = load_for_generation(
                args.model, "mlx"
            )
        elif model_pt is not None:
            # Reuse already-loaded PyTorch model
            model_gen, tok_gen, gen_device = model_pt, tokenizer_pt, device
            meta_gen = meta
        else:
            model_gen, tok_gen, meta_gen, gen_device = load_for_generation(
                args.model, "pytorch"
            )

        for i, prob in enumerate(tqdm(remaining, desc=f"Generating ({args.backend})")):
            for style in styles:
                prompt = PROMPT_STYLES[style](prob["prompt"])
                prompt_fmt = format_prompt(prompt, tok_gen, meta)
                generated = generate_code(
                    model_gen, tok_gen, prompt_fmt, gen_device,
                    backend=args.backend,
                )
                passed = check_correctness(generated, prob)
                all_gen_results[f"{prob['id']}|{style}"] = bool(passed)
            cleanup_gen(gen_device, args.backend)

            if (i + 1) % 10 == 0:
                with open(gen_cache, "w") as f:
                    json.dump(all_gen_results, f)

        with open(gen_cache, "w") as f:
            json.dump(all_gen_results, f)

        del model_gen, tok_gen
        if args.backend == "mlx":
            cleanup_gen(gen_device, "mlx")
        else:
            cleanup(gen_device if 'gen_device' in dir() else None)
    else:
        # Free PyTorch model if still loaded
        if model_pt is not None:
            del model_pt, tokenizer_pt
            cleanup(device)
        print("  All generations cached.")

    # Helper
    def get_passed(prob_id, style):
        return all_gen_results[f"{prob_id}|{style}"]

    # -------------------------------------------------------------------
    # Step 3: Train probe
    # -------------------------------------------------------------------
    print(f"\n--- Step 3: Training {args.probe_mode}/{args.probe_type} probe ---")

    if args.probe_mode == "selection":
        # Selection probe: direct-prompt activation → predict which styles pass
        train_acts = []
        train_labels_multi = []
        for prob in train_problems:
            # Use direct-prompt activation as problem representation
            train_acts.append(all_activations[(str(prob["id"]), "direct")])
            label_vec = [1 if get_passed(prob["id"], s) else 0 for s in styles]
            train_labels_multi.append(label_vec)

        probe, train_acc = train_selection_probe(
            train_acts, train_labels_multi,
            n_styles=len(styles),
            d_model=meta["hidden_size"],
        )
        print(f"  Mode: selection (direct activation → style prediction)")
        print(f"  Train problems: {len(train_acts)}, element-wise accuracy: {train_acc:.4f}")
    else:
        # Original pass/fail probe
        train_acts = []
        train_labels = []
        for prob in train_problems:
            for style in styles:
                train_acts.append(all_activations[(str(prob["id"]), style)])
                train_labels.append(1 if get_passed(prob["id"], style) else 0)

        probe, train_acc = train_probe(
            train_acts, train_labels,
            probe_type=args.probe_type,
            d_model=meta["hidden_size"],
        )
        print(f"  Mode: passfail (per-style pass/fail classification)")
        print(f"  Train samples: {len(train_acts)}, accuracy: {train_acc:.4f}")

    # -------------------------------------------------------------------
    # Step 4: Evaluate on test set
    # -------------------------------------------------------------------
    print(f"\n--- Step 4: Evaluating ({len(test_problems)} test problems) ---")
    results = []

    for prob in tqdm(test_problems, desc="Scoring"):
        row = {"id": prob["id"]}

        # Record per-style pass results
        for style in styles:
            row[f"{style}_pass"] = get_passed(prob["id"], style)

        if args.probe_mode == "selection":
            # Selection: one activation → scores for all styles
            act = all_activations[(str(prob["id"]), "direct")]
            with torch.no_grad():
                logits = probe(act.unsqueeze(0))[0]
            for i, style in enumerate(styles):
                row[f"{style}_score"] = logits[i].item()
            best_style = styles[logits.argmax().item()]
        else:
            # Pass/fail: score each style's activation independently
            best_style = None
            best_score = -float("inf")
            for style in styles:
                act = all_activations[(str(prob["id"]), style)]
                with torch.no_grad():
                    score = probe(act.unsqueeze(0))[0, 1].item()
                row[f"{style}_score"] = score
                if score > best_score:
                    best_score = score
                    best_style = style

        row["probe_selected_style"] = best_style
        row["probe_selected_pass"] = get_passed(prob["id"], best_style)

        rand_style = random.choice(styles)
        row["random_selected_pass"] = get_passed(prob["id"], rand_style)

        row["oracle_pass"] = any(get_passed(prob["id"], s) for s in styles)
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # Keep caches for reuse across probe modes
    # To clean up: rm results/<model>/_act_cache_* results/<model>/_gen_cache_*

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    pass_cols = [f"{s}_pass" for s in styles]
    best_single_rate = max(df[c].mean() for c in pass_cols)
    best_single_name = max(styles, key=lambda s: df[f"{s}_pass"].mean())

    print(f"\n{'='*60}")
    print(f"  {args.model} on {args.dataset} (test={len(test_problems)})")
    print(f"  Probe: {args.probe_mode}/{args.probe_type}, Layer: {probe_layer}")
    print(f"  Prompt library: {len(styles)} styles")
    print(f"{'='*60}")
    print(f"  Oracle (ceiling):   {df['oracle_pass'].mean():.1%} Pass@1")
    print(f"  Probe-selected:     {df['probe_selected_pass'].mean():.1%} Pass@1")
    print(f"  Best single ({best_single_name}): {best_single_rate:.1%} Pass@1")
    print(f"  Random:             {df['random_selected_pass'].mean():.1%} Pass@1")
    print(f"  Direct:             {df['direct_pass'].mean():.1%} Pass@1")
    print(f"  CoT:                {df['cot_pass'].mean():.1%} Pass@1")

    oracle_rate = df["oracle_pass"].mean()
    probe_rate = df["probe_selected_pass"].mean()
    if oracle_rate > 0:
        print(f"\n  Probe captures {probe_rate/oracle_rate:.0%} of oracle ceiling")

    # Style distribution
    print(f"\n  Probe style distribution:")
    for style in styles:
        count = (df["probe_selected_style"] == style).sum()
        if count > 0:
            print(f"    {style:12s}: {count} ({count/len(df):.0%})")

    # Per-style pass rates
    print(f"\n  Per-style Pass@1:")
    for style in styles:
        rate = df[f"{style}_pass"].mean()
        print(f"    {style:12s}: {rate:.1%}")

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
