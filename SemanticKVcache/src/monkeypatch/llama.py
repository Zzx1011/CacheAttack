"""
SemShareKV monkeypatch for HF LlamaAttention (transformers >= 4.56).
- Newer Transformers uses Cache API: past_key_values: Cache and past_key_values.update(...)
- LlamaAttention no longer exposes num_heads; use config / head_dim instead.
"""
import torch
import torch.nn.functional as F

from ..semshare_context import get_semshare_context


def _token_importance_from_attn(attn_weights: torch.Tensor, window: int = 32) -> torch.Tensor:
    # attn_weights: [bs, heads, q_len, k_len]
    w = min(window, attn_weights.shape[-2])
    tail = attn_weights[:, :, -w:, :]
    return tail.mean(dim=2).mean(dim=1)  # [bs, k_len]


def patch_llama_attention():
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        apply_rotary_pos_emb,
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )

    orig_forward = LlamaAttention.forward

    def semshare_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        # compat for old/deprecated callers
        past_key_value=None,
        output_attentions=False,
        use_cache=None,
        **kwargs,
    ):
        # --- normalize args ---
        if past_key_values is None and past_key_value is not None:
            past_key_values = past_key_value

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)

        ctx = get_semshare_context()
        if (not ctx.enabled) or (ctx.injected_past_kv is None) or (position_embeddings is None):
            # fall back to HF
            return orig_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        layer_idx = getattr(self, "layer_idx", None)
        if layer_idx is None:
            return orig_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # --- compute q/k/v as HF does ---
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --- inject KV (must match shapes: [bs, kv_heads, q_len, head_dim]) ---
        inj_k, inj_v = ctx.injected_past_kv[layer_idx]
        if inj_k is not None and inj_v is not None:
            inj_k = inj_k.to(hidden_states.device)
            inj_v = inj_v.to(hidden_states.device)

            if inj_k.shape == key_states.shape and inj_v.shape == value_states.shape:
                recompute_idx = ctx.layer_recompute_idx.get(
                    layer_idx,
                    torch.tensor([], device=hidden_states.device, dtype=torch.long),
                )

                if recompute_idx.numel() > 0:
                    k_new = inj_k.clone()
                    v_new = inj_v.clone()
                    k_new[:, :, recompute_idx, :] = key_states[:, :, recompute_idx, :]
                    v_new[:, :, recompute_idx, :] = value_states[:, :, recompute_idx, :]
                    key_states, value_states = k_new, v_new
                else:
                    key_states, value_states = inj_k, inj_v

        # --- update Cache via HF API (important for correctness) ---
        if past_key_values is not None:
            # sin/cos/cache_position are what HF expects for RoPE models
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)

        # --- attention backend selection ---
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        attention_interface = eager_attention_forward if attn_impl == "eager" else ALL_ATTENTION_FUNCTIONS[attn_impl]

        # match HF's call style
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        with torch.no_grad():
            # store token importance scores for pruning / analysis
            ctx.score_store[layer_idx] = _token_importance_from_attn(attn_weights, window=ctx.score_window).detach()

        # LlamaDecoderLayer expects 2 returns: (attn_output, attn_weights)
        return attn_output, attn_weights

    LlamaAttention.forward = semshare_forward
    return orig_forward
