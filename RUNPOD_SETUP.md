# Pod Setup for 13B-14B Scale Experiments

Step-by-step to run Qwen2.5-Coder 14B (and optionally 7B + DeepSeek + Llama-3) on a rented GPU for camera-ready scale validation.

## Why A100 80GB / H100 80GB?

14B fp16 needs ~28 GB just for weights, plus KV cache headroom — A100 40GB (let alone RTX 4090 24GB) is tight. **A100 80GB is the floor; H100 80GB is faster for similar total cost.**

## Recommended Provider: RunPod Secure Cloud

Best balance of reliability + price for one-shot deadline runs.

| GPU | Hourly | Eviction risk |
|---|---|---|
| RunPod Secure A100 80GB | ~$1.89/hr | None |
| RunPod Secure H100 80GB | ~$2.79/hr | None |
| Lambda Labs A100 80GB | $2.79/hr | None, but capacity often unavailable |
| Vast.ai verified A100 80GB | ~$1.20-1.50/hr | Low (avoid interruptible) |

**Skip:** RunPod Community (small but real eviction risk), Vast.ai interruptible, Modal (mismatch with sequential loop), Paperspace (overpriced).

## 1. Deploy a pod

Go to https://runpod.io/console/pods → **+ Deploy** → pick **A100 80GB** (or H100 80GB if you want it faster) under Secure Cloud.

Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or any recent PyTorch image).

Disk: 50 GB persistent + 50 GB ephemeral. (14B model ~28 GB + datasets ~1 GB = comfortable.)

## 2. Connect

Click **Connect → Web Terminal** in the pod page (or copy the SSH command for a real terminal).

## 3. Clone repo + install

```bash
cd /workspace
git clone https://github.com/rajarshighoshal/cot-llm-probing.git
cd cot-llm-probing
pip install -r requirements.txt
```

## 4. (Optional) HF login for Llama-3-8B

Llama-3 is gated. Skip if you don't run the `full` tier.

```bash
huggingface-cli login
# Paste your HF token (https://huggingface.co/settings/tokens)
```

## 5. Run experiments

Three tiers, pick based on time/budget:

```bash
# Minimal: just Qwen 14B Base+Instruct on HumanEval (~1 hour, ~$2-3)
bash run_7b_scale.sh humaneval minimal

# Core: + Qwen 7B Base+Instruct (scale curve, ~2.5 hours, ~$5-7)
bash run_7b_scale.sh humaneval core

# Full: + DeepSeek 6.7B + Llama-3-8B (cross-family, ~5 hours, ~$10-14)
bash run_7b_scale.sh humaneval full
```

Add MBPP separately if time/budget remain (replace `humaneval` with `mbpp`).

## 6. Download results back to laptop

From your laptop:

```bash
# Find the SSH command on the pod's connection page
scp -r -P <port> root@<pod-ip>:/workspace/cot-llm-probing/results ./results_scale
```

Or use RunPod's file browser to download `results/qwen_coder_14b/`, etc.

## 7. STOP THE POD

RunPod charges by the second while running. Click **Stop** or **Terminate** as soon as you're done — easy to forget and burn $20+ overnight.

## Cost estimates (HumanEval only)

| Tier | A100 80GB ($1.89) | H100 80GB ($2.79) |
|---|---|---|
| Minimal (Qwen 14B B+I) | ~$2 (1 hr) | ~$2 (40 min) |
| Core (+ Qwen 7B B+I) | ~$5 (2.5 hr) | ~$5 (1.5 hr) |
| Full (+ DeepSeek + Llama-3) | ~$10 (5 hr) | ~$10 (3 hr) |

H100 is ~40% faster for ~50% more $/hr → **roughly equal total cost, faster wall-clock**. For a deadline run, H100 wins.

## Troubleshooting

**OOM on 14B even with 80GB?** Check no other process is using GPU: `nvidia-smi`. Reduce KV cache by setting `max_new_tokens=384` (we use 512 default).

**Generation slow (>10s/problem on A100)?** Switch to vLLM:
```bash
pip install vllm
```
Then modify `experiments/generation.py` `_generate_pytorch` to use vLLM (5-10× speedup).

**Resume after disconnect?** Same command — it auto-resumes from the last saved checkpoint (every 10 problems).
