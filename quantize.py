"""
Quantization and dequantization utilities.

Symmetric uniform quantization to n-bit integers.
"""

import torch


def quantize_tensor(w, n_bits=4):
    """
    Per-tensor symmetric uniform quantization.

    Args:
        w: Weight tensor (any shape), float.
        n_bits: Number of bits (2, 3, or 4).

    Returns:
        Dequantized weight tensor (same shape, float).
        Scale factor used.
    """
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))

    scale = w.abs().max() / qmax
    scale = scale.clamp(min=1e-10)

    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    w_hat = q * scale

    return w_hat, scale


def compute_row_scales(W, n_bits=4):
    """
    Compute per-row (per-output-channel) quantization scales.

    Args:
        W: Weight matrix, shape (out_features, in_features).
        n_bits: Quantization bit-width.

    Returns:
        scales: Shape (out_features,).
    """
    qmax = 2 ** (n_bits - 1) - 1
    scales = W.abs().amax(dim=1) / qmax
    scales = scales.clamp(min=1e-10)
    return scales


def quantize_column(w_col, scales, n_bits=4):
    """
    Quantize a weight column using pre-computed per-row scales.

    Used inside GPTQ: each element w_col[i] is quantized with scales[i].

    Args:
        w_col: Column vector, shape (out_features,).
        scales: Per-row scales, shape (out_features,).
        n_bits: Quantization bit-width.

    Returns:
        Dequantized column, shape (out_features,).
    """
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))

    q = torch.clamp(torch.round(w_col / scales), qmin, qmax)
    return q * scales


def round_to_nearest(w, n_bits=4):
    """
    Naive round-to-nearest quantization (no Hessian compensation).
    Baseline for comparison with GPTQ.
    """
    w_hat, _ = quantize_tensor(w, n_bits)
    return w_hat
