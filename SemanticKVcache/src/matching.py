import torch
import torch.nn.functional as F

# 你现在的代码里如果用到了 LSH（FaissLSHIndex），这里可以保留 import
# 没用到也不影响
try:
    from .lsh import FaissLSHIndex  # optional
except Exception:
    FaissLSHIndex = None


@torch.no_grad()
def prepare_fuzzy_mapping(ref_e: torch.Tensor, tgt_e: torch.Tensor, use_rope: bool = True, topk: int = 64):
    """
    Return:
      {
        "map": LongTensor [Lt]    # for each target position, which ref position to reuse
      }

    关键：所有候选索引/取向量的步骤在 CPU 上做，避免 GPU 上的非法索引触发 device-side assert。
    """
    # [Lr, D], [Lt, D]
    assert ref_e.dim() == 2 and tgt_e.dim() == 2
    Lr, D = ref_e.shape
    Lt, D2 = tgt_e.shape
    assert D == D2

    # ---- move to CPU for safe indexing ----
    ref = ref_e.detach().float().cpu().contiguous()
    tgt = tgt_e.detach().float().cpu().contiguous()

    # normalize for cosine similarity
    ref_n = F.normalize(ref, dim=-1)
    tgt_n = F.normalize(tgt, dim=-1)

    # brute-force topk cosine (Lt x Lr) might be heavy if L is huge; but in your demo L is not crazy.
    # if you have a real LSH index, you can swap this block with LSH candidate retrieval.
    sims = tgt_n @ ref_n.t()  # [Lt, Lr]
    k = min(topk, Lr)
    topv, topi = torch.topk(sims, k=k, dim=-1)  # [Lt,k]

    # pick best ref index for each tgt token
    best = topi[:, 0].to(torch.long)  # [Lt]

    # ---- safety clamp (double insurance) ----
    best = torch.clamp(best, 0, Lr - 1)

    # return mapping on ORIGINAL device (so rearrange_past_kv can use it)
    return {"map": best.to(device=ref_e.device)}


@torch.no_grad()
def rearrange_past_kv(ref_pkv, mapping: torch.Tensor):
    """
    ref_pkv: tuple[(k,v)] per layer, each: [bs, kvh, Lr, d]
    mapping: [Lt] long, each entry in [0, Lr-1]
    Return injected pkv aligned to Lt: [bs, kvh, Lt, d]
    """
    mapping = mapping.to(dtype=torch.long, device=ref_pkv[0][0].device)
    out = []
    for (k, v) in ref_pkv:
        # index along seq dimension (dim=2)
        k2 = k.index_select(2, mapping)
        v2 = v.index_select(2, mapping)
        out.append((k2, v2))
    return tuple(out)
