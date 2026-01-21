from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

@dataclass
class PrunePolicy:
    keep_ratio: float = 0.6
    min_keep: int = 256

class PrunableDynamicCache(DynamicCache):
    """
    DynamicCache + pruning hook.
    Generation uses Cache.update(...) internally; we prune right after update and write back. :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, config, prune_policy: Optional[PrunePolicy] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.prune_policy = prune_policy or PrunePolicy()
        self._keep_idx: Dict[int, torch.LongTensor] = {}
        self._scores: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def seed_from_legacy_tuple(self, past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]):
        """
        Initialize internal key_cache/value_cache from legacy tuple format.
        """
        self.key_cache = []
        self.value_cache = []
        for (k, v) in past_key_values:
            self.key_cache.append(k)
            self.value_cache.append(v)

    @torch.no_grad()
    def set_keep_indices(self, layer_idx: int, keep_idx: torch.LongTensor):
        self._keep_idx[layer_idx] = keep_idx

    @torch.no_grad()
    def set_scores(self, layer_idx: int, scores: torch.Tensor):
        self._scores[layer_idx] = scores

    @torch.no_grad()
    def _auto_keep(self, layer_idx: int, seq_len: int, device) -> Optional[torch.LongTensor]:
        if layer_idx not in self._scores:
            return None
        s = self._scores[layer_idx]
        if s.dim() == 2:
            s = s.mean(dim=0)
        s = s.to(device=device).float()

        k = max(int(seq_len * self.prune_policy.keep_ratio), self.prune_policy.min_keep)
        k = min(k, seq_len)
        _, idx = torch.topk(s, k=k, largest=True, sorted=False)
        return torch.sort(idx).values

    @torch.no_grad()
    def _apply_keep(self, k: torch.Tensor, v: torch.Tensor, keep_idx: torch.LongTensor):
        keep_idx = keep_idx.to(device=k.device)
        k2 = k.index_select(dim=-2, index=keep_idx).contiguous()
        v2 = v.index_select(dim=-2, index=keep_idx).contiguous()
        return k2, v2

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keep the same signature as HF Cache.update(). :contentReference[oaicite:4]{index=4}
        """
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)

        seq_len = k_out.shape[-2]
        keep_idx = self._keep_idx.get(layer_idx, None)
        if keep_idx is None:
            keep_idx = self._auto_keep(layer_idx, seq_len=seq_len, device=k_out.device)

        if keep_idx is not None:
            keep_idx = keep_idx[:seq_len]  # safety
            k_pruned, v_pruned = self._apply_keep(k_out, v_out, keep_idx)
            self.key_cache[layer_idx] = k_pruned
            self.value_cache[layer_idx] = v_pruned
            return k_pruned, v_pruned

        return k_out, v_out
