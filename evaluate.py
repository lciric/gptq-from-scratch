"""
Perplexity evaluation on WikiText-2 test set.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm

from arch_config import get_arch_config


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, device="cuda", stride=512, max_len=None):
    """
    Evaluate perplexity on WikiText-2 test set using sliding window.

    Standard approach (cf. HuggingFace perplexity docs):
    - Tokenize full test set as one long sequence
    - Slide a context window of size max_len with given stride
    - At each step, mask context overlap so loss only counts new tokens

    Args:
        model: Causal LM model.
        tokenizer: Corresponding tokenizer.
        device: Device string.
        stride: Number of new tokens evaluated per window.
        max_len: Context window size override (default: model's max sequence length).

    Returns:
        Perplexity (float).
    """
    arch = get_arch_config(model)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)  # (1, total_tokens)

    if max_len is None:
        max_len = arch.get_max_seq_len(model)
    total_len = input_ids.size(1)

    nlls = []
    n_tokens = 0

    for begin in tqdm(range(0, total_len, stride), desc="Evaluating perplexity"):
        end = min(begin + max_len, total_len)
        input_chunk = input_ids[:, begin:end]

        # Build labels: mask overlap context with -100
        labels = input_chunk.clone()
        n_ctx = 0 if begin == 0 else max_len - stride
        labels[:, :n_ctx] = -100

        outputs = model(input_chunk, labels=labels)

        # Count valid (non-masked) label tokens
        # HF loss = mean cross-entropy over tokens where label != -100
        n_valid = (labels != -100).sum().item()
        if n_valid > 0:
            nlls.append(outputs.loss.float() * n_valid)
            n_tokens += n_valid

        if end == total_len:
            break

    avg_nll = torch.stack(nlls).sum() / n_tokens
    perplexity = torch.exp(avg_nll).item()
    return perplexity
