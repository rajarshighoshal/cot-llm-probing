"""Model registry.

All models use HuggingFace Transformers with output_hidden_states=True
for universal activation extraction. No TransformerLens dependency.
"""

MODEL_REGISTRY = {
    "qwen_coder": {
        "name": "Qwen/Qwen2.5-Coder-1.5B",
        "family": "qwen2",
        "params": "1.5B",
    },
    "qwen_coder_instruct": {
        "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "family": "qwen2",
        "params": "1.5B",
        "instruct": True,
    },
    "deepseek_coder": {
        "name": "deepseek-ai/deepseek-coder-1.3b-base",
        "family": "llama",
        "params": "1.3B",
    },
    "deepseek_coder_instruct": {
        "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "family": "llama",
        "params": "1.3B",
        "instruct": True,
    },
    "codellama": {
        "name": "codellama/CodeLlama-7b-hf",
        "family": "llama",
        "params": "7B",
    },
    "starcoder2_3b": {
        "name": "bigcode/starcoder2-3b",
        "family": "starcoder2",
        "params": "3B",
    },
    "phi2": {
        "name": "microsoft/phi-2",
        "family": "phi",
        "params": "2.7B",
    },
    # --- 7B scale models ---
    "qwen_coder_7b": {
        "name": "Qwen/Qwen2.5-Coder-7B",
        "family": "qwen2",
        "params": "7B",
    },
    "qwen_coder_7b_instruct": {
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "family": "qwen2",
        "params": "7B",
        "instruct": True,
    },
    "deepseek_coder_67b": {
        "name": "deepseek-ai/deepseek-coder-6.7b-base",
        "family": "llama",
        "params": "6.7B",
    },
    "deepseek_coder_67b_instruct": {
        "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "family": "llama",
        "params": "6.7B",
        "instruct": True,
    },
    # --- 13B-14B scale models (need A100 80GB or H100) ---
    "qwen_coder_14b": {
        "name": "Qwen/Qwen2.5-Coder-14B",
        "family": "qwen2",
        "params": "14B",
    },
    "qwen_coder_14b_instruct": {
        "name": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "family": "qwen2",
        "params": "14B",
        "instruct": True,
    },
    "codellama_13b": {
        "name": "codellama/CodeLlama-13b-hf",
        "family": "llama",
        "params": "13B",
    },
    "codellama_13b_instruct": {
        "name": "codellama/CodeLlama-13b-Instruct-hf",
        "family": "llama",
        "params": "13B",
        "instruct": True,
    },
    # --- Cross-family validation ---
    "llama3_8b_instruct": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "family": "llama",
        "params": "8B",
        "instruct": True,
    },
}


def get_model_name(key):
    """Return HuggingFace model name for a registry key."""
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {key}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]["name"]
