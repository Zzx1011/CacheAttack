from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch

@dataclass
class SemShareContext:
    """
    Runtime context for SemShareKV prefill patch.

    injected_past_kv:
      tuple(layers)->(k,v), aligned to target prompt length
      k/v: [bs, n_kv_heads, seq_len, head_dim]
    """
    enabled: bool = False
    injected_past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None

    # layer -> LongTensor of token positions to recompute in that layer
    layer_recompute_idx: Dict[int, torch.LongTensor] = field(default_factory=dict)

    # layer -> attention-based importance score [bs, seq_len]
    score_store: Dict[int, torch.Tensor] = field(default_factory=dict)

    # hyperparams
    attn_recovery: float = 0.55
    recompute_hot_ratio: float = 0.5
    recompute_cold_ratio: float = 0.1
    score_window: int = 32

_GLOBAL_CTX = SemShareContext()

def set_semshare_context(ctx: SemShareContext):
    global _GLOBAL_CTX
    _GLOBAL_CTX = ctx

def get_semshare_context() -> SemShareContext:
    return _GLOBAL_CTX

def disable_semshare():
    global _GLOBAL_CTX
    _GLOBAL_CTX.enabled = False
    _GLOBAL_CTX.injected_past_kv = None
    _GLOBAL_CTX.layer_recompute_idx = {}
    _GLOBAL_CTX.score_store = {}
