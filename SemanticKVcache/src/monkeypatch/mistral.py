"""
SemShareKV prefill monkeypatch for MistralAttention.

Key fixes for newer Transformers:
- Some MistralAttention instances DO NOT expose `num_heads`, `num_key_value_heads`, or `rotary_emb`.
  We infer heads from projection output dims and rely on `position_embeddings=(cos,sin)` when provided.
- New cache API uses `past_key_values` (Cache/DynamicCache) updated in-place via `.update(...)`,
  so attention forward typically returns (attn_output, attn_weights_or_None), not (attn_output, present).

Note: Mistral uses sliding-window attention; DynamicCache can stop growing past the window for those layers.
"""
import torch
import torch.nn.functional as F

from ..semshare_context import get_semshare_context


def _token_importance_from_attn(attn_weights: torch.Tensor, window: int = 32) -> torch.Tensor:
    """
    attn_weights: [bs, heads, q_len, k_len]
    return: [bs, k_len] importance score
    """
    w = min(window, attn_weights.shape[-2])
    tail = attn_weights[:, :, -w:, :]                 # [bs, heads, w, k_len]
    return tail.mean(dim=2).mean(dim=1)               # [bs, k_len]


def patch_mistral_attention():
    from transformers.models.mistral.modeling_mistral import MistralAttention, apply_rotary_pos_emb

    # repeat_kv may exist; if not, fallback
    try:
        from transformers.models.mistral.modeling_mistral import repeat_kv  # type: ignore
    except Exception:
        def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
            # x: [bs, kv_heads, seq, head_dim] -> [bs, kv_heads*n_rep, seq, head_dim]
            if n_rep == 1:
                return x
            return x.repeat_interleave(n_rep, dim=1)

    orig_forward = MistralAttention.forward

    def semshare_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,          # legacy name (deprecated)
        past_key_values=None,         # new Cache API
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,     # NEW in many HF versions: (cos, sin)
        **kwargs,
    ):
        ctx = get_semshare_context()
        if (not ctx.enabled) or (ctx.injected_past_kv is None):
            # passthrough — keep signature compatibility
            return orig_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        layer_idx = getattr(self, "layer_idx", None)
        if layer_idx is None:
            return orig_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device

        # ---- infer head counts robustly ----
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            raise AttributeError("MistralAttention has no `head_dim`; cannot infer head layout safely.")

        # q_proj output = num_heads * head_dim
        q_proj_out = self.q_proj(hidden_states)                     # [bs, q_len, hidden]
        num_heads = q_proj_out.shape[-1] // head_dim

        query_states = q_proj_out.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)  # [bs, h, q, d]

        # ---- RoPE for queries (prefer position_embeddings passed from upper model) ----
        cos = sin = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif hasattr(self, "rotary_emb"):
            # old-style fallback
            cos, sin = self.rotary_emb(query_states, position_ids)

        if (cos is not None) and (sin is not None):
            # rotate queries; we pass q as both q/k and discard rotated "k"
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids)

        # ---- start from injected KV (already shaped like HF cache tensors) ----
        inj_k, inj_v = ctx.injected_past_kv[layer_idx]
        key_states = inj_k.to(device)     # [bs, kvh, k_len, d]
        value_states = inj_v.to(device)

        # ---- optionally recompute a subset of KV from current hidden_states ----
        # indices are token positions along sequence dimension
        if layer_idx == 0:
            recompute_idx = torch.arange(q_len, device=device, dtype=torch.long)
        else:
            recompute_idx = ctx.layer_recompute_idx.get(
                layer_idx, torch.tensor([], device=device, dtype=torch.long)
            )

        # clamp & unique (avoid out-of-range)
        if recompute_idx.numel() > 0:
            recompute_idx = recompute_idx.unique()
            recompute_idx = recompute_idx[(recompute_idx >= 0) & (recompute_idx < q_len)]

        if recompute_idx.numel() > 0:
            hs_sub = hidden_states.index_select(dim=1, index=recompute_idx)  # [bs, r, hidden]

            k_sub = self.k_proj(hs_sub)
            v_sub = self.v_proj(hs_sub)

            num_kv_heads = k_sub.shape[-1] // head_dim
            k_sub = k_sub.view(bsz, -1, num_kv_heads, head_dim).transpose(1, 2)  # [bs, kvh, r, d]
            v_sub = v_sub.view(bsz, -1, num_kv_heads, head_dim).transpose(1, 2)

            # apply RoPE to recomputed keys (so they match current positions)
            if (cos is not None) and (sin is not None):
                if position_ids is not None:
                    pos_sub = position_ids.index_select(dim=1, index=recompute_idx)
                else:
                    pos_sub = None
                # rotate keys; pass k as both q/k and discard rotated "q"
                _, k_sub = apply_rotary_pos_emb(k_sub, k_sub, cos, sin, pos_sub)

            # write back
            key_states = key_states.clone()
            value_states = value_states.clone()
            key_states[:, :, recompute_idx, :] = k_sub
            value_states[:, :, recompute_idx, :] = v_sub

        # ---- update HF Cache object in-place (if provided) ----
        cache_obj = past_key_values if past_key_values is not None else past_key_value
        if use_cache and (cache_obj is not None) and hasattr(cache_obj, "update"):
            cache_kwargs = {}
            if (cos is not None) and (sin is not None):
                cache_kwargs.update({"sin": sin, "cos": cos})
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position

            # update returns possibly-sliced states (e.g., static cache / sliding window)
            key_states, value_states = cache_obj.update(key_states, value_states, layer_idx, cache_kwargs)

        # ---- attention compute (matmul path; works regardless of SDPA/flash availability) ----
        kv_heads = key_states.shape[1]
        num_key_value_groups = max(1, num_heads // kv_heads)

        key_states_full = repeat_kv(key_states, num_key_value_groups)       # [bs, h, k, d]
        value_states_full = repeat_kv(value_states, num_key_value_groups)   # [bs, h, k, d]

        attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) / (head_dim ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states_full)         # [bs, h, q, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # store per-layer token scores for pruning
        with torch.no_grad():
            ctx.score_store[layer_idx] = _token_importance_from_attn(attn_weights, window=ctx.score_window).detach()

        # IMPORTANT: match HF new return convention for MistralAttention used by your stacktrace:
        # decoder layer does: hidden_states, _ = self.self_attn(...)
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None

    MistralAttention.forward = semshare_forward
    return orig_forward
