#!/usr/bin/env python3
"""
Phase B: Direct vs CoT Code Generation Comparison

Generates code with multiple prompt styles, evaluates Pass@1 via execution,
and runs McNemar's test for statistical significance.

Supports resume and MLX backend for fast Apple Silicon inference.

Usage:
    python -m experiments.generation --model qwen_coder --dataset humaneval
    python -m experiments.generation --model qwen_coder --dataset humaneval --backend mlx
    python -m experiments.generation --model deepseek_coder --dataset mbpp
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from data_loader import get_dataset
from experiments import PROMPT_STYLES
from metrics.evaluator import check_correctness, mcnemar_test
from models import MODEL_REGISTRY, get_model_name


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------

def _load_pytorch(model_key):
    import torch
    from models.loader import get_device, load_model
    device = get_device()
    model, tokenizer, meta = load_model(model_key, device)
    return model, tokenizer, meta, device


def _generate_pytorch(model, tokenizer, prompt, device, max_new_tokens=512):
    import torch
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)

    stop_strings = ["```\n", "\n```"]
    stop_ids = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.append(ids[0])
    eos_ids = [tokenizer.eos_token_id] + stop_ids

    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    generated = output[0, tokens["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _cleanup_pytorch(device):
    from models.loader import cleanup
    cleanup(device)


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon native — 2-5x faster than PyTorch MPS)
# ---------------------------------------------------------------------------

MLX_LOCAL_PATHS = {
    "deepseek_coder": "/tmp/deepseek-coder-1.3b-base-mlx",
}


def _load_mlx(model_key):
    from mlx_lm import load
    model_name = MLX_LOCAL_PATHS.get(model_key, get_model_name(model_key))
    print(f"Loading {model_name} with MLX...")
    model, tokenizer = load(model_name)

    # MLX models store args, not config
    args = None
    for attr in ["args", "config"]:
        if hasattr(model, attr):
            args = getattr(model, attr)
            break
        if hasattr(model, "model") and hasattr(model.model, attr):
            args = getattr(model.model, attr)
            break

    meta = {
        "num_layers": getattr(args, "num_hidden_layers", 0),
        "hidden_size": getattr(args, "hidden_size", 0),
        "num_params": 0,
        "instruct": MODEL_REGISTRY.get(model_key, {}).get("instruct", False),
    }
    print(f"  Loaded with MLX backend")
    return model, tokenizer, meta, "mlx"


def _generate_mlx(model, tokenizer, prompt, device, max_new_tokens=512):
    import re
    from mlx_lm import stream_generate

    text = ""
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_new_tokens):
        text += response.text
        # Early stop: if a complete code block is present (opening + content + closing ```)
        # Safe: only triggers when extract_code would succeed, so no premature cutoff
        if re.search(r'```(?:python)?\s*\n.+?\n```', text, re.DOTALL):
            break

    return text


def _cleanup_mlx(device):
    pass  # MLX handles memory automatically


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def load_for_generation(model_key, backend="pytorch"):
    if backend == "mlx":
        return _load_mlx(model_key)
    return _load_pytorch(model_key)


def generate_code(model, tokenizer, prompt, device, backend="pytorch",
                  max_new_tokens=512):
    if backend == "mlx":
        return _generate_mlx(model, tokenizer, prompt, device, max_new_tokens)
    return _generate_pytorch(model, tokenizer, prompt, device, max_new_tokens)


def cleanup_gen(device, backend="pytorch"):
    if backend == "mlx":
        return _cleanup_mlx(device)
    return _cleanup_pytorch(device)


def format_prompt_gen(prompt, tokenizer, meta):
    """Wrap prompt in chat template for instruct models."""
    if not meta.get("instruct", False):
        return prompt
    messages = [{"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase B: Direct vs CoT Comparison")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp", "livecodebench"])
    parser.add_argument("--styles", nargs="+", default=["direct", "cot"],
                        choices=list(PROMPT_STYLES.keys()))
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "mlx"],
                        help="Inference backend (mlx is faster on Apple Silicon)")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    model, tokenizer, meta, device = load_for_generation(args.model, args.backend)

    ds = get_dataset(args.dataset)
    problems = ds.get_problems()
    if args.max_problems:
        problems = problems[:args.max_problems]

    # Resume support
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generation_{args.dataset}.csv")

    completed_ids = set()
    results = []
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        completed_ids = set(existing["id"].tolist())
        results = existing.to_dict("records")
        print(f"Resuming: {len(completed_ids)} problems already done, "
              f"{len(problems) - len(completed_ids)} remaining")

    remaining = [p for p in problems if p["id"] not in completed_ids]
    print(f"Evaluating {len(remaining)} problems, styles: {args.styles}, "
          f"backend: {args.backend}")

    # Generate and evaluate
    for prob in tqdm(remaining, desc="Generating"):
        row = {"id": prob["id"]}
        for style in args.styles:
            prompt = PROMPT_STYLES[style](prob["prompt"])
            prompt = format_prompt_gen(prompt, tokenizer, meta)
            generated = generate_code(model, tokenizer, prompt, device,
                                      backend=args.backend)
            passed = check_correctness(generated, prob)
            row[f"{style}_pass"] = passed
            row[f"{style}_output"] = generated[:500]
        results.append(row)
        cleanup_gen(device, args.backend)

        # Checkpoint every 10 problems
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    df = pd.DataFrame(results)
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

    # McNemar's test
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
