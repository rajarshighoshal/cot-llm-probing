#!/usr/bin/env python3
"""
Phase C: Probe-Guided Strategy Selector

For each problem, extracts activations under all 6 prompt styles,
uses a trained probe to predict which style will succeed, and selects
the best one. Compares against baselines (Direct, CoT, Random).

Usage:
    python -m experiments.strategy_selector --model qwen_coder --dataset humaneval
    python -m experiments.strategy_selector --model deepseek_coder --dataset mbpp --train_split 200
"""

import argparse
import gc
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_dataset
from experiments import PROMPT_STYLES
from experiments.generation import generate_code
from metrics.evaluator import check_correctness
from models import MODEL_REGISTRY
from models.loader import get_device, load_model, get_last_token_activation


# ---------------------------------------------------------------------------
# Probe for pass/fail prediction
# ---------------------------------------------------------------------------

class PassFailProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Linear(d_model, 2)

    def forward(self, x):
        return self.net(x)


def train_probe(activations, labels, n_epochs=100, lr=1e-3):
    """Train pass/fail probe. Returns trained probe and accuracy."""
    X = torch.stack(activations)
    y = torch.tensor(labels).long()

    probe = PassFailProbe(X.shape[1])
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase C: Probe-Guided Strategy Selector")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--probe_layer", type=int, default=None,
                        help="Layer to extract activations from (default: ~12.5%% depth)")
    parser.add_argument("--train_split", type=int, default=None,
                        help="Number of problems for probe training (rest = eval)")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, meta = load_model(args.model, device)
    styles = list(PROMPT_STYLES.keys())

    # Default probe layer: ~12.5% depth (early emergence point)
    if args.probe_layer is None:
        args.probe_layer = max(1, meta["num_layers"] // 8)
    print(f"Probe layer: {args.probe_layer}")

    # Load dataset
    ds = get_dataset(args.dataset)
    problems = ds.get_problems()

    # Train/test split
    if args.train_split is None:
        args.train_split = len(problems) // 2
    train_problems = problems[:args.train_split]
    test_problems = problems[args.train_split:]
    print(f"Train: {len(train_problems)}, Test: {len(test_problems)}")

    # -----------------------------------------------------------------------
    # Phase 1: Generate training data (activations + pass/fail for each style)
    # -----------------------------------------------------------------------
    print("\n--- Phase 1: Collecting training data ---")
    train_activations = []
    train_labels = []

    for prob in tqdm(train_problems, desc="Training data"):
        for style in styles:
            prompt = PROMPT_STYLES[style](prob["prompt"])

            # Get activation
            act = get_last_token_activation(
                model, tokenizer, prompt, args.probe_layer, device
            )

            # Generate and check
            generated = generate_code(model, tokenizer, prompt, device)
            passed = check_correctness(generated, prob)

            train_activations.append(act)
            train_labels.append(1 if passed else 0)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Train probe
    print(f"\nTraining probe ({len(train_activations)} samples)...")
    probe, train_acc = train_probe(train_activations, train_labels)
    print(f"Probe train accuracy: {train_acc:.4f}")

    # -----------------------------------------------------------------------
    # Phase 2: Evaluate on test set
    # -----------------------------------------------------------------------
    print("\n--- Phase 2: Evaluating on test set ---")
    results = []

    for prob in tqdm(test_problems, desc="Evaluating"):
        row = {"id": prob["id"]}

        # Score each style with the probe
        best_style = None
        best_score = -float("inf")
        style_results = {}

        for style in styles:
            prompt = PROMPT_STYLES[style](prob["prompt"])

            # Probe score
            act = get_last_token_activation(
                model, tokenizer, prompt, args.probe_layer, device
            )
            with torch.no_grad():
                logits = probe(act.unsqueeze(0))
                score = logits[0, 1].item()  # P(pass)

            # Actually generate and check
            generated = generate_code(model, tokenizer, prompt, device)
            passed = check_correctness(generated, prob)

            style_results[style] = {"score": score, "passed": passed}
            row[f"{style}_pass"] = passed
            row[f"{style}_score"] = score

            if score > best_score:
                best_score = score
                best_style = style

        # Probe selection
        row["probe_selected_style"] = best_style
        row["probe_selected_pass"] = style_results[best_style]["passed"]

        # Random baseline
        rand_style = random.choice(styles)
        row["random_selected_pass"] = style_results[rand_style]["passed"]

        results.append(row)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    # Save
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"strategy_selector_{args.dataset}.csv")
    df.to_csv(out_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.model} on {args.dataset} (test={len(test_problems)})")
    print(f"{'='*60}")
    print(f"  Probe-selected: {df['probe_selected_pass'].mean():.1%} Pass@1")
    print(f"  Direct:         {df['direct_pass'].mean():.1%} Pass@1")
    print(f"  CoT:            {df['cot_pass'].mean():.1%} Pass@1")
    print(f"  Random:         {df['random_selected_pass'].mean():.1%} Pass@1")

    # Style distribution
    print(f"\n  Probe style distribution:")
    for style in styles:
        count = (df["probe_selected_style"] == style).sum()
        print(f"    {style:12s}: {count} ({count/len(df):.0%})")

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
