import torch

@torch.no_grad()
def keep_indices_from_scores(
    scores: torch.Tensor,
    attn_recovery: float = 0.55,
    cold_keep_ratio: float = 0.2,
    min_keep: int = 256,
) -> torch.LongTensor:
    """
    scores: [bs, seq_len] or [seq_len]
      - Larger means more important.
    Strategy:
      1) "Hot" tokens: smallest set whose cumulative mass >= attn_recovery
      2) From remaining "Cold": keep an extra budget = cold_keep_ratio * (seq_len - hot)
      3) Always keep at least min_keep total.

    Returns sorted LongTensor indices to keep.
    """
    if scores.dim() == 2:
        s = scores.mean(dim=0)
    else:
        s = scores
    s = s.float()
    seq_len = s.numel()

    # hot = top tokens by score until cumulative >= attn_recovery
    vals, idx = torch.sort(s, descending=True)
    mass = vals / (vals.sum() + 1e-8)
    c = torch.cumsum(mass, dim=0)
    n_hot = int((c <= attn_recovery).sum().item())
    n_hot = min(n_hot + 1, seq_len)  # include first that crosses threshold
    hot_idx = idx[:n_hot]

    # cold budget
    remaining = seq_len - n_hot
    n_cold = int(cold_keep_ratio * remaining)
    # total min keep
    target_total = max(min_keep, n_hot + n_cold)
    target_total = min(target_total, seq_len)

    keep_idx = idx[:target_total]
    return torch.sort(keep_idx).values
