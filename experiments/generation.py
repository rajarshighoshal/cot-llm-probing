#!/usr/bin/env python3
"""
Phase B: Direct vs CoT Code Generation Comparison

Generates code with multiple prompt styles, evaluates Pass@1 via execution,
and runs McNemar's test for statistical significance.

Usage:
    python -m experiments.generation --model qwen_coder --dataset humaneval
    python -m experiments.generation --model deepseek_coder --dataset mbpp
    python -m experiments.generation --model qwen_coder --dataset humaneval --styles direct cot plan expert
"""

import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from data_loader import get_dataset
from experiments import PROMPT_STYLES
from metrics.evaluator import check_correctness, mcnemar_test
from models import MODEL_REGISTRY
from models.loader import get_device, load_model


def generate_code(model, tokenizer, prompt, device, max_new_tokens=512):
    """Greedy decode for Pass@1."""
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


def main():
    parser = argparse.ArgumentParser(description="Phase B: Direct vs CoT Comparison")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--styles", nargs="+", default=["direct", "cot"],
                        choices=list(PROMPT_STYLES.keys()))
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, meta = load_model(args.model, device)

    ds = get_dataset(args.dataset)
    problems = ds.get_problems()
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"Evaluating {len(problems)} problems, styles: {args.styles}")

    # Generate and evaluate
    results = []
    for prob in tqdm(problems, desc="Generating"):
        row = {"id": prob["id"]}
        for style in args.styles:
            prompt = PROMPT_STYLES[style](prob["prompt"])
            generated = generate_code(model, tokenizer, prompt, device)
            passed = check_correctness(generated, prob)
            row[f"{style}_pass"] = passed
            row[f"{style}_output"] = generated[:500]
        results.append(row)

    df = pd.DataFrame(results)

    # Save
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generation_{args.dataset}.csv")
    df.to_csv(out_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.model} on {args.dataset} ({len(problems)} problems)")
    print(f"{'='*60}")
    for style in args.styles:
        col = f"{style}_pass"
        rate = df[col].mean()
        n = df[col].sum()
        print(f"  {style:12s}: {rate:.1%} Pass@1 ({n:.0f}/{len(df)})")

    # McNemar's test for pairwise comparisons
    if len(args.styles) >= 2:
        s1, s2 = args.styles[0], args.styles[1]
        try:
            chi2, p = mcnemar_test(df[f"{s1}_pass"].tolist(), df[f"{s2}_pass"].tolist())
            print(f"\n  McNemar ({s1} vs {s2}): chi2={chi2:.3f}, p={p:.4f}")
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  => {sig} (p {'<' if p < 0.05 else '>='} 0.05)")
        except ImportError:
            print("\n  (install scipy for McNemar's test)")

    # Disagreement counts
    if "direct" in args.styles and "cot" in args.styles:
        d_wins = ((df["direct_pass"]) & (~df["cot_pass"])).sum()
        c_wins = ((~df["direct_pass"]) & (df["cot_pass"])).sum()
        print(f"\n  Direct wins (Direct OK, CoT fail): {d_wins}")
        print(f"  CoT wins (CoT OK, Direct fail):    {c_wins}")

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
