#!/usr/bin/env python3
"""
Phase D: Probe-Guided Prompt Auto-Correction

Given ANY user prompt, the system:
1. Scores the raw prompt via a trained probe
2. Tries wrapping it with style prefixes from a learned library
3. Early-exit validates top candidates (generate ~20 tokens, re-probe)
4. Generates full completion with the best candidate

The user never sees the library. They give a prompt, the system fixes it.

Usage:
    python -m experiments.auto_corrector --model qwen_coder --dataset humaneval
    python -m experiments.auto_corrector --model qwen_coder --dataset mbpp --train_split 250 --probe_type mlp
"""

import argparse
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
from metrics.evaluator import check_correctness
from models import MODEL_REGISTRY
from models.loader import get_device, load_model, get_last_token_activation, cleanup


# ---------------------------------------------------------------------------
# Probes (same as strategy_selector)
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


def train_probe(activations, labels, probe_type="linear", d_model=None,
                n_epochs=200, lr=1e-3):
    X = torch.stack(activations)
    y = torch.tensor(labels).long()
    if d_model is None:
        d_model = X.shape[1]

    probe = MLPProbe(d_model) if probe_type == "mlp" else LinearProbe(d_model)
    if probe_type == "mlp":
        n_epochs = 300

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
# Early-exit scoring: generate a few tokens, probe the trajectory
# ---------------------------------------------------------------------------

def generate_partial(model, tokenizer, prompt, device, n_tokens=20):
    """Generate a few tokens and return (partial_text, final_hidden_state)."""
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    partial_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return partial_text


def score_with_probe(model, tokenizer, text, probe, probe_layer, device):
    """Get probe P(pass) score for a text."""
    act = get_last_token_activation(model, tokenizer, text, probe_layer, device)
    with torch.no_grad():
        logits = probe(act.unsqueeze(0))
        return logits[0, 1].item()


def generate_full(model, tokenizer, prompt, device, max_new_tokens=512):
    """Full greedy generation."""
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, tokens["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Auto-correction pipeline
# ---------------------------------------------------------------------------

def auto_correct(model, tokenizer, prompt, probe, probe_layer, device,
                 top_k=3, early_exit_tokens=20):
    """
    Auto-correct a prompt using probe-guided selection + early-exit validation.

    Returns: (best_prompt, method_used, timing_info)
    """
    styles = list(PROMPT_STYLES.keys())
    timing = {}

    # Step 1: Score raw prompt + all augmented versions
    t0 = time.time()
    candidates = [("raw", prompt)]
    for style in styles:
        candidates.append((style, PROMPT_STYLES[style](prompt)))

    scores = []
    for name, augmented_prompt in candidates:
        s = score_with_probe(model, tokenizer, augmented_prompt, probe,
                             probe_layer, device)
        scores.append((name, augmented_prompt, s))
    timing["probe_scoring"] = time.time() - t0

    # Sort by probe score, take top-k
    scores.sort(key=lambda x: x[2], reverse=True)
    top_candidates = scores[:top_k]

    # Step 2: Early-exit validation on top-k
    t1 = time.time()
    early_scores = []
    for name, augmented_prompt, initial_score in top_candidates:
        partial = generate_partial(model, tokenizer, augmented_prompt, device,
                                   n_tokens=early_exit_tokens)
        # Re-probe after partial generation
        trajectory_score = score_with_probe(model, tokenizer, partial, probe,
                                            probe_layer, device)
        early_scores.append((name, augmented_prompt, initial_score,
                             trajectory_score))
    timing["early_exit"] = time.time() - t1

    # Pick best by trajectory score
    early_scores.sort(key=lambda x: x[3], reverse=True)
    best_name, best_prompt, best_initial, best_trajectory = early_scores[0]

    timing["total_selection"] = timing["probe_scoring"] + timing["early_exit"]

    return best_name, best_prompt, timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase D: Probe-Guided Prompt Auto-Correction")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--probe_type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--probe_layer", type=int, default=None)
    parser.add_argument("--train_split", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of candidates for early-exit validation")
    parser.add_argument("--early_exit_tokens", type=int, default=20,
                        help="Tokens to generate for early-exit scoring")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, meta = load_model(args.model, device)
    styles = list(PROMPT_STYLES.keys())

    if args.probe_layer is None:
        args.probe_layer = max(1, meta["num_layers"] // 8)
    print(f"Probe: {args.probe_type}, layer {args.probe_layer}")
    print(f"Early-exit: top-{args.top_k}, {args.early_exit_tokens} tokens")

    # Load dataset
    ds = get_dataset(args.dataset)
    problems = ds.get_problems()

    if args.train_split is None:
        args.train_split = len(problems) // 2
    train_problems = problems[:args.train_split]
    test_problems = problems[args.train_split:]
    print(f"Train: {len(train_problems)}, Test: {len(test_problems)}")

    # -----------------------------------------------------------------------
    # Phase 1: Train probe on training set
    # -----------------------------------------------------------------------
    print("\n--- Phase 1: Training probe ---")
    train_activations = []
    train_labels = []

    for prob in tqdm(train_problems, desc="Training data"):
        for style in styles:
            prompt = PROMPT_STYLES[style](prob["prompt"])
            act = get_last_token_activation(
                model, tokenizer, prompt, args.probe_layer, device
            )
            generated = generate_full(model, tokenizer, prompt, device)
            passed = check_correctness(generated, prob)
            train_activations.append(act)
            train_labels.append(1 if passed else 0)

        cleanup(device)

    print(f"\nTraining {args.probe_type} probe ({len(train_activations)} samples)...")
    probe, train_acc = train_probe(
        train_activations, train_labels,
        probe_type=args.probe_type, d_model=meta["hidden_size"],
    )
    print(f"Probe train accuracy: {train_acc:.4f}")

    # -----------------------------------------------------------------------
    # Phase 2: Evaluate auto-correction on test set
    # -----------------------------------------------------------------------
    print("\n--- Phase 2: Evaluating auto-correction ---")
    results = []

    for prob in tqdm(test_problems, desc="Evaluating"):
        row = {"id": prob["id"]}

        # Auto-correct the raw prompt
        t_start = time.time()
        selected_style, corrected_prompt, timing = auto_correct(
            model, tokenizer, prob["prompt"], probe, args.probe_layer, device,
            top_k=args.top_k, early_exit_tokens=args.early_exit_tokens,
        )

        # Generate with corrected prompt
        generated = generate_full(model, tokenizer, corrected_prompt, device)
        passed = check_correctness(generated, prob)
        t_total = time.time() - t_start

        row["auto_corrected_style"] = selected_style
        row["auto_corrected_pass"] = passed
        row["selection_time_s"] = timing["total_selection"]
        row["total_time_s"] = t_total

        # Baselines: generate with raw prompt (direct) and CoT
        raw_gen = generate_full(model, tokenizer, prob["prompt"], device)
        row["direct_pass"] = check_correctness(raw_gen, prob)

        cot_prompt = PROMPT_STYLES["cot"](prob["prompt"])
        cot_gen = generate_full(model, tokenizer, cot_prompt, device)
        row["cot_pass"] = check_correctness(cot_gen, prob)

        results.append(row)

        cleanup(device)

    df = pd.DataFrame(results)

    # Save
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"auto_corrector_{args.dataset}_{args.probe_type}.csv")
    df.to_csv(out_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.model} on {args.dataset} (test={len(test_problems)})")
    print(f"  Probe: {args.probe_type}, top-{args.top_k} early-exit")
    print(f"{'='*60}")
    print(f"  Auto-corrected: {df['auto_corrected_pass'].mean():.1%} Pass@1")
    print(f"  Direct (raw):   {df['direct_pass'].mean():.1%} Pass@1")
    print(f"  CoT:            {df['cot_pass'].mean():.1%} Pass@1")

    # Style distribution
    print(f"\n  Auto-selected styles:")
    for style in ["raw"] + list(styles):
        count = (df["auto_corrected_style"] == style).sum()
        if count > 0:
            print(f"    {style:12s}: {count} ({count/len(df):.0%})")

    # How often did auto-correction change the prompt?
    raw_kept = (df["auto_corrected_style"] == "raw").sum()
    modified = len(df) - raw_kept
    print(f"\n  Prompt modified: {modified}/{len(df)} ({modified/len(df):.0%})")
    print(f"  Prompt kept raw: {raw_kept}/{len(df)} ({raw_kept/len(df):.0%})")

    # Latency
    avg_sel = df["selection_time_s"].mean()
    avg_total = df["total_time_s"].mean()
    print(f"\n  Latency:")
    print(f"    Avg selection (probe + early-exit): {avg_sel:.2f}s")
    print(f"    Avg total (selection + generation): {avg_total:.2f}s")
    print(f"    Overhead: {avg_sel/avg_total*100:.1f}% of total")

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
