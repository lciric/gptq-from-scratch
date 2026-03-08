"""
Architecture-specific configuration for multi-model GPTQ support.

Supports GPT-2, LLaMA (covers Llama-2/3 + Mistral), and OPT model families.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, List, Optional

import torch


@dataclass
class ArchConfig:
    """Architecture-specific accessors for a causal LM."""
    get_blocks: Callable       # model -> nn.ModuleList of transformer blocks
    get_final_ln: Callable     # model -> final LayerNorm
    compute_embeddings: Callable  # (model, input_ids, device) -> hidden_states
    block_forward: Callable    # (block, hidden, **kwargs) -> hidden_out
    get_block_kwargs: Callable # (model, input_ids, device) -> dict of extra kwargs
    get_max_seq_len: Callable  # model -> int
    layer_name_prefix: str     # e.g. "transformer.h" or "model.layers"
    sublayer_groups: Optional[List[List[str]]] = field(default=None)  # for true-sequential


# ---------------------------------------------------------------------------
# GPT-2
# ---------------------------------------------------------------------------

def _gpt2_get_blocks(model):
    return model.transformer.h

def _gpt2_get_final_ln(model):
    return model.transformer.ln_f

def _gpt2_compute_embeddings(model, input_ids, device):
    t = model.transformer
    position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
    return t.wte(input_ids) + t.wpe(position_ids)

def _gpt2_block_forward(block, hidden, **kwargs):
    return block(hidden, **kwargs)[0]

def _gpt2_get_block_kwargs(model, input_ids, device):
    return {}

def _gpt2_get_max_seq_len(model):
    return model.config.n_positions

GPT2_CONFIG = ArchConfig(
    get_blocks=_gpt2_get_blocks,
    get_final_ln=_gpt2_get_final_ln,
    compute_embeddings=_gpt2_compute_embeddings,
    block_forward=_gpt2_block_forward,
    get_block_kwargs=_gpt2_get_block_kwargs,
    get_max_seq_len=_gpt2_get_max_seq_len,
    layer_name_prefix="transformer.h",
    sublayer_groups=[
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"],
    ],
)


# ---------------------------------------------------------------------------
# LLaMA (covers Llama-2, Llama-3, Mistral, etc.)
# ---------------------------------------------------------------------------

def _llama_get_blocks(model):
    return model.model.layers

def _llama_get_final_ln(model):
    return model.model.norm

def _llama_compute_embeddings(model, input_ids, device):
    return model.model.embed_tokens(input_ids)

def _llama_block_forward(block, hidden, **kwargs):
    # transformers >= 4.44 returns a tensor; older versions return a tuple
    out = block(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out

def _llama_get_block_kwargs(model, input_ids, device):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
    kwargs = {"position_ids": position_ids}
    # position_embeddings=(cos, sin) is mandatory in transformers >= 4.44
    # cos, sin shape: (batch=1, seq_len, head_dim)
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        dtype = next(model.parameters()).dtype
        dummy = torch.empty(1, seq_len, 1, device=device, dtype=dtype)
        cos, sin = rotary_emb(dummy, position_ids)
        kwargs["position_embeddings"] = (cos, sin)
    return kwargs

def _llama_get_max_seq_len(model):
    return getattr(model.config, "max_position_embeddings", 4096)

LLAMA_CONFIG = ArchConfig(
    get_blocks=_llama_get_blocks,
    get_final_ln=_llama_get_final_ln,
    compute_embeddings=_llama_compute_embeddings,
    block_forward=_llama_block_forward,
    get_block_kwargs=_llama_get_block_kwargs,
    get_max_seq_len=_llama_get_max_seq_len,
    layer_name_prefix="model.layers",
    sublayer_groups=[
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ],
)


# ---------------------------------------------------------------------------
# OPT
# ---------------------------------------------------------------------------

def _opt_get_blocks(model):
    return model.model.decoder.layers

def _opt_get_final_ln(model):
    # OPT may or may not have a final layer norm depending on version
    decoder = model.model.decoder
    if hasattr(decoder, "final_layer_norm"):
        return decoder.final_layer_norm
    return None

def _opt_compute_embeddings(model, input_ids, device):
    decoder = model.model.decoder
    inputs_embeds = decoder.embed_tokens(input_ids)
    # OPT uses a learned positional embedding with offset=2
    attention_mask = torch.ones_like(input_ids)
    pos_embeds = decoder.embed_positions(attention_mask)
    hidden = inputs_embeds + pos_embeds
    if hasattr(decoder, "project_in") and decoder.project_in is not None:
        hidden = decoder.project_in(hidden)
    return hidden

def _opt_block_forward(block, hidden, **kwargs):
    return block(hidden, **kwargs)[0]

def _opt_get_block_kwargs(model, input_ids, device):
    return {}

def _opt_get_max_seq_len(model):
    return getattr(model.config, "max_position_embeddings", 2048)

OPT_CONFIG = ArchConfig(
    get_blocks=_opt_get_blocks,
    get_final_ln=_opt_get_final_ln,
    compute_embeddings=_opt_compute_embeddings,
    block_forward=_opt_block_forward,
    get_block_kwargs=_opt_get_block_kwargs,
    get_max_seq_len=_opt_get_max_seq_len,
    layer_name_prefix="model.decoder.layers",
    sublayer_groups=[
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"],
    ],
)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

_CONFIG_CLASS_MAP = {
    "GPT2Config": GPT2_CONFIG,
    "LlamaConfig": LLAMA_CONFIG,
    "MistralConfig": LLAMA_CONFIG,
    "OPTConfig": OPT_CONFIG,
}


def get_arch_config(model) -> ArchConfig:
    """Detect model architecture and return the corresponding config."""
    config_name = model.config.__class__.__name__
    if config_name in _CONFIG_CLASS_MAP:
        return _CONFIG_CLASS_MAP[config_name]
    raise ValueError(
        f"Unsupported architecture: {config_name}. "
        f"Supported: {list(_CONFIG_CLASS_MAP.keys())}"
    )
