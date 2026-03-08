"""
GPTQ algorithm implementation (Frantar et al., 2022).

Efficient block-by-block approach:
  1. Run calibration data through embeddings -> get inputs to block 0
  2. For each transformer block:
     a. Run block inputs through the block, capturing activations via hooks
     b. Quantize all Linear/Conv1D layers in the block using GPTQ
     c. Re-run the block to get outputs -> these become inputs for the next block
  This requires only n_blocks forward passes, not n_layers * n_samples.

Supports GPT-2, LLaMA, OPT and other architectures via arch_config.

Optimizations:
  - Grouping: per-group-of-g-columns scales instead of per-row (group_size=128)
  - Act-order: quantize columns sorted by descending Hessian diagonal
  - True-sequential: quantize sub-layers sequentially within each block
"""

import time

import torch
from tqdm import tqdm

from quantize import compute_row_scales, quantize_column


def compute_hessian(X, damp_pct=0.01):
    """
    Hessian approximation H = X^T X / n + damping * I.

    Args:
        X: Input activations, shape (n_tokens, in_features). Float32.
        damp_pct: Damping as fraction of mean diagonal.

    Returns:
        H: shape (in_features, in_features). Float32.
    """
    n = X.shape[0]
    H = X.T @ X / n

    damp = damp_pct * torch.mean(torch.diag(H))
    diag_idx = range(H.shape[0])
    H[diag_idx, diag_idx] += damp

    return H


def gptq_quantize_layer(W, H, n_bits=4, block_size=128, group_size=-1, act_order=False):
    """
    GPTQ quantization of a single weight matrix (Algorithm 1 from the paper).

    Args:
        W: Weight matrix, shape (out_features, in_features). Float32.
        H: Hessian, shape (in_features, in_features). Float32.
        n_bits: Quantization bit-width.
        block_size: Columns per block (B in the paper).
        group_size: Columns per quantization group (-1 = per-row scales).
        act_order: If True, quantize columns in descending activation order.

    Returns:
        Q: Quantized weights (dequantized to float), same shape as W.
        loss: Scalar quantization loss.
    """
    W = W.clone().float()
    n_rows, n_cols = W.shape

    # Act-order: permute columns by descending Hessian diagonal
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]

    # Invert H via Cholesky
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        extra_damp = 0.1 * torch.mean(torch.diag(H))
        H_reg = H.clone()
        diag_idx = range(H.shape[0])
        H_reg[diag_idx, diag_idx] += extra_damp
        L = torch.linalg.cholesky(H_reg)
        H_inv = torch.cholesky_inverse(L)

    # Scales: per-group (recomputed at group boundaries) or per-row (fixed)
    if group_size > 0:
        current_group = -1
        scales = None
    else:
        scales = compute_row_scales(W, n_bits)

    Q = torch.zeros_like(W)
    loss = 0.0

    for i1 in range(0, n_cols, block_size):
        i2 = min(i1 + block_size, n_cols)
        count = i2 - i1

        W_blk = W[:, i1:i2].clone()
        Q_blk = torch.zeros_like(W_blk)
        Err = torch.zeros_like(W_blk)
        H_inv_blk = H_inv[i1:i2, i1:i2]

        for j in range(count):
            col_idx = i1 + j

            # Recompute scales at group boundaries
            if group_size > 0:
                g = col_idx // group_size
                if g != current_group:
                    current_group = g
                    g_start = g * group_size
                    g_end = min(g_start + group_size, n_cols)
                    scales = compute_row_scales(W[:, g_start:g_end], n_bits)

            w_col = W_blk[:, j]
            h_jj = H_inv_blk[j, j]

            q_col = quantize_column(w_col, scales, n_bits)
            Q_blk[:, j] = q_col

            err = (w_col - q_col) / h_jj
            Err[:, j] = err

            loss += ((w_col - q_col) ** 2 / h_jj).sum().item()

            if j < count - 1:
                W_blk[:, j + 1:] -= err.unsqueeze(1) * H_inv_blk[j, j + 1:].unsqueeze(0)

        Q[:, i1:i2] = Q_blk

        if i2 < n_cols:
            W[:, i2:] -= Err @ H_inv[i1:i2, i2:]

    # Inverse permute if act-order was used
    if perm is not None:
        inv_perm = torch.argsort(perm)
        Q = Q[:, inv_perm]

    return Q, loss


def _capture_and_quantize(block, layers, hidden_states, arch, block_kwargs,
                          block_idx, device, n_bits, block_size, group_size, act_order):
    """
    Hook layers, run forward passes to capture Hessians, then quantize.

    Returns:
        stats: Dict mapping full layer name -> {loss, time}.
    """
    from model_utils import get_weight_and_type, set_weight

    layer_H = {name: None for name, _ in layers}
    layer_n = {name: 0 for name, _ in layers}

    hooks = []
    for name, module in layers:
        def make_hook(layer_name):
            def hook_fn(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                H = x.T @ x
                if layer_H[layer_name] is None:
                    layer_H[layer_name] = H
                else:
                    layer_H[layer_name] += H
                layer_n[layer_name] += x.shape[0]
            return hook_fn
        hooks.append(module.register_forward_hook(make_hook(name)))

    for h in hidden_states:
        arch.block_forward(block, h.to(device), **block_kwargs)

    for handle in hooks:
        handle.remove()

    stats = {}
    for name, module in layers:
        t0 = time.time()
        full_name = f"{arch.layer_name_prefix}.{block_idx}.{name}"

        H = layer_H[name] / layer_n[name]
        damp = 0.01 * torch.mean(torch.diag(H))
        H[range(H.shape[0]), range(H.shape[0])] += damp

        W, is_conv1d = get_weight_and_type(module)
        Q, loss = gptq_quantize_layer(
            W.float(), H, n_bits=n_bits, block_size=block_size,
            group_size=group_size, act_order=act_order,
        )
        set_weight(module, Q, is_conv1d)

        elapsed = time.time() - t0
        stats[full_name] = {"loss": loss, "time": elapsed}
        del H, W, Q
        layer_H[name] = None

    del layer_H, layer_n
    torch.cuda.empty_cache()

    return stats


@torch.no_grad()
def quantize_model(model, calibration_data, n_bits=4, block_size=128,
                   group_size=-1, act_order=False, true_sequential=False,
                   device="cpu"):
    """
    Apply GPTQ to all layers, processing one transformer block at a time.

    Args:
        model: HuggingFace causal LM (GPT-2, LLaMA, OPT, etc.).
        calibration_data: List of input tensors (each shape (1, seq_len)).
        n_bits: Bit-width.
        block_size: GPTQ block size.
        group_size: Columns per quantization group (-1 = per-row).
        act_order: Quantize columns in descending activation order.
        true_sequential: Quantize sub-layers sequentially within each block.
        device: Device string.

    Returns:
        stats: Dict mapping layer name -> {loss, time}.
    """
    from model_utils import get_transformer_blocks, get_block_inputs

    arch, blocks = get_transformer_blocks(model)

    sample_ids = calibration_data[0]
    block_kwargs = arch.get_block_kwargs(model, sample_ids, device)

    print("Computing embedding outputs for calibration data...")
    hidden_states = get_block_inputs(model, calibration_data, device=device)
    hidden_states = [h.cpu() for h in hidden_states]
    torch.cuda.empty_cache()
    print(f"Got {len(hidden_states)} calibration samples")

    stats = {}

    for block_idx, block in enumerate(tqdm(blocks, desc=f"GPTQ {n_bits}-bit blocks")):
        # Find all quantizable layers in this block
        layers_to_quantize = []
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear) or type(module).__name__ == "Conv1D":
                layers_to_quantize.append((name, module))

        if true_sequential and arch.sublayer_groups is not None:
            # Process sub-layer groups sequentially, re-capturing activations each time
            quantized_names = set()
            for group_names in arch.sublayer_groups:
                layers_in_group = [(n, m) for n, m in layers_to_quantize if n in group_names]
                if not layers_in_group:
                    continue
                group_stats = _capture_and_quantize(
                    block, layers_in_group, hidden_states, arch, block_kwargs,
                    block_idx, device, n_bits, block_size, group_size, act_order,
                )
                stats.update(group_stats)
                quantized_names.update(n for n, _ in layers_in_group)

            # Quantize any remaining layers not covered by sublayer_groups
            remaining = [(n, m) for n, m in layers_to_quantize if n not in quantized_names]
            if remaining:
                rem_stats = _capture_and_quantize(
                    block, remaining, hidden_states, arch, block_kwargs,
                    block_idx, device, n_bits, block_size, group_size, act_order,
                )
                stats.update(rem_stats)
        else:
            # Standard: hook all layers at once, run once, quantize all
            block_stats = _capture_and_quantize(
                block, layers_to_quantize, hidden_states, arch, block_kwargs,
                block_idx, device, n_bits, block_size, group_size, act_order,
            )
            stats.update(block_stats)

        # Propagate hidden states through the (now quantized) block
        new_hidden = []
        for h in hidden_states:
            new_hidden.append(arch.block_forward(block, h.to(device), **block_kwargs).detach().cpu())
        hidden_states = new_hidden

    return stats
