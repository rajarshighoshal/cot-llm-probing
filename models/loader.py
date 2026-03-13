"""Universal model loader.

Loads any HuggingFace causal LM with proper device/dtype handling.
Activation extraction uses output_hidden_states=True — works with
every HuggingFace model without TransformerLens.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import get_model_name


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_key, device=None):
    """Load model + tokenizer. Returns (model, tokenizer, metadata).

    metadata dict contains: num_layers, hidden_size, num_params.
    """
    if device is None:
        device = get_device()

    model_name = get_model_name(model_key)
    print(f"Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    metadata = {
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    print(f"  {metadata['num_layers']} layers, d={metadata['hidden_size']}, "
          f"{metadata['num_params']/1e6:.0f}M params")

    return model, tokenizer, metadata


def get_hidden_states(model, tokenizer, text, device):
    """Extract hidden states from all layers for a single text.

    Returns tuple of (num_layers + 1) tensors.
    Index 0 = embedding output, index i = layer i-1 output.
    Each tensor shape: (1, seq_len, hidden_size).
    """
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    return outputs.hidden_states


def get_last_token_activation(model, tokenizer, text, layer, device):
    """Extract last-token activation at a specific layer.

    Returns tensor of shape (hidden_size,).
    """
    hidden_states = get_hidden_states(model, tokenizer, text, device)
    # layer+1 because index 0 is embedding output
    return hidden_states[layer + 1][0, -1, :].detach().float().cpu()
