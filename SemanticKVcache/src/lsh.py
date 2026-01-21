import numpy as np
import faiss

# Paper Appendix A.3
LSH_DIST_MIN = 0.0
LSH_DIST_MAX = 30.0


# def _paper_sim_from_dist(dist: float) -> float:
#     """sim = clip(1 - (dist-min)/(max-min), 0, 1) with min=0 max=30."""
#     d_norm = (float(dist) - LSH_DIST_MIN) / (LSH_DIST_MAX - LSH_DIST_MIN)
#     return float(np.clip(1.0 - d_norm, 0.0, 1.0))


def lsh_token_match_and_sim(
    ref_e: np.ndarray,
    tgt_e: np.ndarray,
    nbits: int = 256,
    seed: int = 1234,  # kept for API compatibility; faiss.IndexLSH doesn't expose seed in Python
):
    """
    ref_e: [Lr, D] float32
    tgt_e: [Lt, D] float32

    Returns:
      mapping: [Lt] each tgt token -> nearest ref token index under LSH/Hamming
      sim: scalar in [0,1] using paper's min=0 max=30 normalization
    """
    ref = np.asarray(ref_e, dtype=np.float32)
    tgt = np.asarray(tgt_e, dtype=np.float32)

    d = ref.shape[1]
    index = faiss.IndexLSH(d, int(nbits))  # LSH -> binary codes -> Hamming distance :contentReference[oaicite:1]{index=1}

    # Some FAISS examples call train() for IndexLSH; it's safe to do so.
    # It is typically a no-op for many flat indexes, but calling it won't hurt.
    if hasattr(index, "train") and not index.is_trained:
        index.train(ref)  # :contentReference[oaicite:2]{index=2}

    index.add(ref)

    # FAISS python API: D, I = index.search(x_query, k) :contentReference[oaicite:3]{index=3}
    D, I = index.search(tgt, 1)  # D: [Lt,1], I: [Lt,1]
    mapping = I.reshape(-1).astype(np.int64)

    mean_hamming = float(np.asarray(D).mean())
    # sim = clip(1 - mean_hamming/nbits, 0, 1)
    sim = float(np.clip(1.0 - (mean_hamming / float(nbits)), 0.0, 1.0))
    
    
    return mapping, sim
