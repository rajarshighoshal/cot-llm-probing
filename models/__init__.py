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
}


def get_model_name(key):
    """Return HuggingFace model name for a registry key."""
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {key}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]["name"]
