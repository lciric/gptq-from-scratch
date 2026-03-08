"""
Model utilities: loading, calibration data preparation, block-wise processing.
Supports GPT-2, LLaMA, OPT and other architectures via arch_config.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from arch_config import get_arch_config


def load_model(model_name="gpt2", device="cuda", token=None):
    """Load a causal LM and tokenizer in FP16 (CUDA) or FP32 (CPU)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, token=token,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def get_transformer_blocks(model):
    """
    Get architecture config and transformer blocks from any supported model.

    Returns:
        arch: ArchConfig for this model family.
        blocks: nn.ModuleList of transformer blocks.
    """
    arch = get_arch_config(model)
    blocks = arch.get_blocks(model)
    return arch, blocks


def get_calibration_data(tokenizer, n_samples=128, seq_len=2048, seed=42,
                         dataset_name="wikitext2"):
    """
    Extract calibration data from a training set.

    Args:
        tokenizer: HuggingFace tokenizer.
        n_samples: Number of calibration segments.
        seq_len: Tokens per segment.
        seed: Random seed for reproducibility.
        dataset_name: "wikitext2" or "c4".

    Returns a list of (n_samples) tensors, each of shape (1, seq_len).
    """
    if dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        samples = []
        for doc in dataset:
            toks = tokenizer.encode(doc["text"])
            if len(toks) >= seq_len:
                samples.append(torch.tensor(toks[:seq_len], dtype=torch.long).unsqueeze(0))
                if len(samples) >= n_samples:
                    break
        return samples
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(dataset["text"])
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)

    rng = torch.Generator()
    rng.manual_seed(seed)

    samples = []
    for _ in range(n_samples):
        start = torch.randint(0, len(tokens) - seq_len, (1,), generator=rng).item()
        segment = tokens[start : start + seq_len].unsqueeze(0)
        samples.append(segment)

    return samples


@torch.no_grad()
def get_block_inputs(model, calibration_data, device="cpu"):
    """
    Run calibration data through embeddings only, capturing inputs to block 0.

    Uses the architecture config to compute embeddings for any supported model.

    Returns:
        List of tensors, each shape (1, seq_len, hidden_dim).
    """
    arch = get_arch_config(model)

    block_inputs = []
    for batch in calibration_data:
        batch = batch.to(device)
        hidden = arch.compute_embeddings(model, batch, device)
        block_inputs.append(hidden)

    return block_inputs


def get_weight_and_type(layer):
    """
    Get weight in (out_features, in_features) format, handling Conv1D.

    Returns:
        W: weight tensor in (out, in) format
        is_conv1d: bool
    """
    if type(layer).__name__ == "Conv1D":
        return layer.weight.data.T.clone(), True
    else:
        return layer.weight.data.clone(), False


def set_weight(layer, Q, is_conv1d):
    """Write quantized weight back, handling Conv1D transpose."""
    if is_conv1d:
        layer.weight.data = Q.T.to(layer.weight.dtype)
    else:
        layer.weight.data = Q.to(layer.weight.dtype)
