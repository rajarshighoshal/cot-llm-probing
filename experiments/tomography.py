#!/usr/bin/env python3
"""
Phase A: Layer-wise Tomography

Trains linear probes at each layer to classify Direct vs CoT prompts.
Tests whether prompt-type information emerges early in the network.

Usage:
    python -m experiments.tomography --model qwen_coder --dataset humaneval
    python -m experiments.tomography --model deepseek_coder --dataset mbpp --samples 100
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_dataset
from experiments import PROMPT_STYLES
from models import MODEL_REGISTRY
from models.loader import get_device, load_model, get_last_token_activation, cleanup


class LinearProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Linear(d_model, 2)

    def forward(self, x):
        return self.net(x)


def extract_activations(model, tokenizer, texts, layer, device):
    """Extract last-token activations at a given layer for a batch of texts."""
    acts = []
    for text in texts:
        act = get_last_token_activation(model, tokenizer, text, layer, device)
        acts.append(act)
    return torch.stack(acts)


def train_and_eval_probe(X_direct, X_cot, n_epochs=50, lr=1e-3):
    """Train a linear probe and return train accuracy."""
    X = torch.cat([X_direct, X_cot])
    y = torch.cat([torch.zeros(len(X_direct)), torch.ones(len(X_cot))]).long()

    probe = LinearProbe(X.shape[1])
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(probe(X), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (probe(X).argmax(1) == y).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser(description="Phase A: Layer-wise Tomography")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp", "livecodebench"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--layer_step", type=int, default=1)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, meta = load_model(args.model, device)

    # Load dataset
    ds = get_dataset(args.dataset)
    prompts = ds.get_prompts()[:args.samples]
    cot_prompts = [PROMPT_STYLES["cot"](p) for p in prompts]
    print(f"Using {len(prompts)} problems from {args.dataset}")

    # Run tomography
    layers = list(range(0, meta["num_layers"], args.layer_step))
    results = []

    for layer in tqdm(layers, desc="Scanning layers"):
        X_direct = extract_activations(model, tokenizer, prompts, layer, device)
        X_cot = extract_activations(model, tokenizer, cot_prompts, layer, device)
        acc = train_and_eval_probe(X_direct, X_cot)
        results.append({"layer": layer, "accuracy": acc})
        print(f"  Layer {layer:3d}: {acc:.4f}")

        del X_direct, X_cot
        cleanup(device)

    # Save
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tomography_{args.dataset}.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # Summary
    peak = df.loc[df["accuracy"].idxmax()]
    emergence = df[df["accuracy"] > 0.9]
    e_layer = int(emergence.iloc[0]["layer"]) if len(emergence) > 0 else "none"

    print(f"\n{'='*60}")
    print(f"  Model:      {args.model} ({meta['num_layers']} layers)")
    print(f"  Dataset:    {args.dataset} ({len(prompts)} samples)")
    print(f"  Peak:       {peak['accuracy']:.4f} at layer {int(peak['layer'])}")
    print(f"  Emergence:  layer {e_layer} (first >90%)")
    print(f"  Saved:      {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
