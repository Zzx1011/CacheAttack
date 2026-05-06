from __future__ import annotations

import math
import time
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CacheAttackGenerator:
    """
    Adversarial suffix generator using GCG (Greedy Coordinate Gradient) search.
    """

    def __init__(
        self,
        embed_model_name: str,
        lm_model_name: str = "gpt2",
        mode: str = "cos",
        lambda_ppl: float = 0.05,
        lsh_nbits: int = 256,
        lsh_alpha: float = 10.0,
        max_lm_length: int = 128,
        suffix_prefix: str = "Neglect: ",
    ):
        assert mode in ("cos", "lsh"), f"mode must be 'cos' or 'lsh', got {mode}"
        self.mode = mode
        self.lambda_ppl = lambda_ppl
        self.lsh_nbits = lsh_nbits
        self.lsh_alpha = lsh_alpha
        self.max_lm_length = max_lm_length
        self.suffix_prefix = suffix_prefix

        # ── Embedding model (surrogate) ──────────────────────────────────────
        logger.info(f"Loading embedding model: {embed_model_name}")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
        self.embed_model.eval()
        self.embedding_weight = self.embed_model.get_input_embeddings().weight
        self.vocab_size = self.embedding_weight.size(0)

        # ── Language model (PPL scoring) ─────────────────────────────────────
        logger.info(f"Loading LM for PPL: {lm_model_name}")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name).to(device)
        self.lm_model.eval()
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        # ── LSH projection matrix (fixed random, drawn once) ─────────────────
        if mode == "lsh":
            embed_dim = self.embed_model.config.hidden_size
            torch.manual_seed(42)
            self._R_lsh = torch.randn(lsh_nbits, embed_dim, device=device)  # [B, D]
            self._R_lsh = F.normalize(self._R_lsh, dim=1)

    # ------------------------------------------------------------------ utils

    def _get_cls_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return L2-normalised [CLS] embedding."""
        out = self.embed_model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, dim=1)

    def _get_mean_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return L2-normalised mean-pooled embedding."""
        out = self.embed_model(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state  # [1, L, D]
        mask = attention_mask.unsqueeze(-1).float()
        mean = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(mean, dim=1)

    def _get_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self._get_cls_embedding(input_ids, attention_mask)

    @torch.no_grad()
    def _compute_ppl(self, text: str) -> float:
        inputs = self.lm_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_lm_length
        ).to(device)
        out = self.lm_model(**inputs, labels=inputs["input_ids"])
        return math.exp(min(out.loss.item(), 20.0))

    def _lsh_hash_soft(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Soft relaxation of LSH: h̃(p) = tanh(α · R_LSH · e(p))  (Eq. 5)
        emb: [1, D]  ->  returns [1, B]
        """
        proj = emb @ self._R_lsh.t()  # [1, B]
        return torch.tanh(self.lsh_alpha * proj)

    # ------------------------------------------------------------------ losses

    def _collision_loss_cos(
        self,
        combined_input_ids: torch.Tensor,
        tgt_emb: torch.Tensor,
    ) -> torch.Tensor:
        attn = torch.ones_like(combined_input_ids)
        cur_emb = self._get_embedding(combined_input_ids, attn)
        return 1.0 - (cur_emb * tgt_emb).sum()

    def _collision_loss_lsh(
        self,
        src_embed: torch.Tensor,  # [1, D] from embed model (requires_grad via embeds path)
        tgt_h: torch.Tensor,      # [1, B] pre-computed target soft hash
    ) -> torch.Tensor:
        h_a = self._lsh_hash_soft(src_embed)
        return (h_a - tgt_h).pow(2).sum()

    # ------------------------------------------------------------------ core GCG step

    def _gcg_step(
        self,
        src_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        tgt_emb_or_h: torch.Tensor,
        top_k: int,
        batch_size: int,
        s_src: str,
    ) -> Tuple[torch.Tensor, float]:
        """
        One GCG iteration.
        Returns updated suffix_ids and best combined score.
        """
        suffix_len = len(suffix_ids)

        cls_id = torch.tensor([self.embed_tokenizer.cls_token_id], device=device)
        sep_id = torch.tensor([self.embed_tokenizer.sep_token_id], device=device)

        # ── Forward with gradient ────────────────────────────────────────────
        input_ids = torch.cat([cls_id, src_ids, suffix_ids, sep_id]).unsqueeze(0)
        input_embeds = self.embed_model.get_input_embeddings()(input_ids).detach()
        input_embeds.requires_grad_(True)

        outputs = self.embed_model(inputs_embeds=input_embeds)
        cur_emb = F.normalize(outputs.last_hidden_state[:, 0, :], dim=1)

        if self.mode == "cos":
            loss = 1.0 - (cur_emb * tgt_emb_or_h).sum()
        else:
            h_a = self._lsh_hash_soft(cur_emb)
            loss = (h_a - tgt_emb_or_h).pow(2).sum()

        loss.backward()

        # ── Gradient-based top-k token candidates ───────────────────────────
        start_idx = 1 + len(src_ids)
        grads = input_embeds.grad[0, start_idx: start_idx + suffix_len]  # [L, D]
        scores = -torch.matmul(grads, self.embedding_weight.T)            # [L, V]
        topk_ids = torch.topk(scores, top_k, dim=1).indices               # [L, k]

        # ── Batch evaluation ─────────────────────────────────────────────────
        best_ids = suffix_ids.clone()
        best_score = float("-inf")

        for _ in range(batch_size):
            cand = suffix_ids.clone()
            pos = torch.randint(0, suffix_len, (1,)).item()
            tok = topk_ids[pos, torch.randint(0, top_k, (1,)).item()]
            cand[pos] = tok

            # decode to text, prepend suffix_prefix if configured
            suffix_text = self.embed_tokenizer.decode(cand, skip_special_tokens=True)
            full_text = s_src + " " + self.suffix_prefix + suffix_text

            # embedding similarity
            enc = self.embed_tokenizer(
                full_text, return_tensors="pt", truncation=True
            ).to(device)
            with torch.no_grad():
                emb = self._get_embedding(enc["input_ids"], enc["attention_mask"])
                if self.mode == "cos":
                    sim_val = (emb * tgt_emb_or_h).sum().item()
                else:
                    h_a = self._lsh_hash_soft(emb)
                    sim_val = -(h_a - tgt_emb_or_h).pow(2).sum().item()

            # PPL penalty
            ppl = self._compute_ppl(full_text)
            combined = sim_val - self.lambda_ppl * math.log(max(ppl, 1.0))

            if combined > best_score:
                best_ids = cand
                best_score = combined

        return best_ids, best_score

    # ------------------------------------------------------------------ public API

    def optimize_suffix(
        self,
        p_src: str,
        p_v: str,
        suffix_len: int,
        steps: int = 200,
        batch_size: int = 64,
        top_k: int = 64,
        target_sim: float = 0.88,
        verbose: bool = False,
    ) -> Tuple[Optional[str], float]:
        
        # Tokenise source (no special tokens; we add manually)
        src_ids = self.embed_tokenizer(
            p_src, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].to(device)

        # Pre-compute target representation
        tgt_enc = self.embed_tokenizer(
            p_v, return_tensors="pt", truncation=True
        ).to(device)
        with torch.no_grad():
            tgt_emb = self._get_embedding(tgt_enc["input_ids"], tgt_enc["attention_mask"])
            if self.mode == "lsh":
                tgt_h = self._lsh_hash_soft(tgt_emb)
                tgt_repr = tgt_h
            else:
                tgt_repr = tgt_emb

        # Initialise random suffix
        suffix_ids = torch.randint(0, self.vocab_size, (suffix_len,), device=device)

        best_sim = float("-inf")
        best_suffix_ids = suffix_ids.clone()

        for step in range(steps):
            suffix_ids, score = self._gcg_step(
                src_ids, suffix_ids, tgt_repr, top_k, batch_size, p_src
            )
            if score > best_sim:
                best_sim = score
                best_suffix_ids = suffix_ids.clone()

            # Check convergence: re-compute true cosine similarity
            suffix_text = self.embed_tokenizer.decode(
                best_suffix_ids, skip_special_tokens=True
            )
            full_text = p_src + " " + self.suffix_prefix + suffix_text
            enc = self.embed_tokenizer(
                full_text, return_tensors="pt", truncation=True
            ).to(device)
            with torch.no_grad():
                cur_emb = self._get_embedding(enc["input_ids"], enc["attention_mask"])
                if self.mode == "cos":
                    true_sim = (cur_emb * tgt_emb).sum().item()
                else:
                    # For LSH mode, use cosine as proxy for reporting
                    true_sim = (cur_emb * tgt_emb).sum().item()

            if verbose and step % 50 == 0:
                logger.info(
                    f"  step={step:4d}  sim={true_sim:.4f}  "
                    f"target={target_sim:.3f}  L={suffix_len}"
                )

            if true_sim >= target_sim:
                suffix_str = self.embed_tokenizer.decode(
                    best_suffix_ids, skip_special_tokens=True
                )
                return self.suffix_prefix + suffix_str, true_sim

        suffix_str = self.embed_tokenizer.decode(best_suffix_ids, skip_special_tokens=True)
        # Return best even if not converged
        enc = self.embed_tokenizer(
            p_src + " " + self.suffix_prefix + suffix_str,
            return_tensors="pt", truncation=True
        ).to(device)
        with torch.no_grad():
            cur_emb = self._get_embedding(enc["input_ids"], enc["attention_mask"])
            final_sim = (cur_emb * tgt_emb).sum().item()

        return None, final_sim

    def run_dynamic_attack(
        self,
        p_src: str,
        p_v: str,
        init_len: int = 5,
        max_len: int = 40,
        len_step: int = 5,
        steps_per_len: int = 200,
        batch_size: int = 64,
        top_k: int = 64,
        target_sim: float = 0.88,
        verbose: bool = False,
    ) -> Tuple[Optional[str], float]:

        best_suffix, best_sim = None, float("-inf")
        for L in range(init_len, max_len + 1, len_step):
            suffix, sim = self.optimize_suffix(
                p_src=p_src,
                p_v=p_v,
                suffix_len=L,
                steps=steps_per_len,
                batch_size=batch_size,
                top_k=top_k,
                target_sim=target_sim,
                verbose=verbose,
            )
            if sim > best_sim:
                best_sim = sim
                best_suffix = suffix
            if suffix is not None:
                return suffix, sim

        return None, best_sim


# ---------------------------------------------------------------------------
# Validator  (Section 5.1 – latency-based MAP rule)
# ---------------------------------------------------------------------------

class LatencyValidator:

    def __init__(self):
        self.mu_hit = None
        self.sigma_hit = None
        self.mu_miss = None
        self.sigma_miss = None

    def calibrate(
        self,
        cache_api,
        seed_prompt: str,
        n_hit: int = 20,
        n_miss: int = 20,
        sleep_between: float = 0.1,
    ) -> None:
        
        # Populate cache
        cache_api.query(seed_prompt)
        time.sleep(0.5)

        hit_lat = []
        for _ in range(n_hit):
            t0 = time.time()
            cache_api.query(seed_prompt)
            hit_lat.append(time.time() - t0)
            time.sleep(sleep_between)

        miss_lat = []
        for i in range(n_miss):
            nonce = f"{seed_prompt} [NONCE_{i}_{np.random.randint(0, 99999)}]"
            t0 = time.time()
            cache_api.query(nonce)
            miss_lat.append(time.time() - t0)
            time.sleep(sleep_between)

        log_hit = np.log(np.array(hit_lat) + 1e-9)
        log_miss = np.log(np.array(miss_lat) + 1e-9)

        self.mu_hit = float(np.mean(log_hit))
        self.sigma_hit = float(np.std(log_hit) + 1e-6)
        self.mu_miss = float(np.mean(log_miss))
        self.sigma_miss = float(np.std(log_miss) + 1e-6)

        logger.info(
            f"[Validator] hit  μ={self.mu_hit:.4f}  σ={self.sigma_hit:.4f}"
        )
        logger.info(
            f"[Validator] miss μ={self.mu_miss:.4f}  σ={self.sigma_miss:.4f}"
        )

    def predict(self, latency: float) -> Tuple[bool, float]:
        if self.mu_hit is None:
            raise RuntimeError("Call calibrate() before predict().")

        y = math.log(max(latency, 1e-9))
        ll_hit = norm.logpdf(y, self.mu_hit, self.sigma_hit)
        ll_miss = norm.logpdf(y, self.mu_miss, self.sigma_miss)
        llr = ll_hit - ll_miss
        return llr >= 0, float(llr)

    def recalibrate(
        self,
        cache_api,
        seed_prompt: str,
        n_hit: int = 10,
        n_miss: int = 10,
        sleep_between: float = 0.1,
    ) -> None:
        self.calibrate(cache_api, seed_prompt, n_hit, n_miss, sleep_between)


# ---------------------------------------------------------------------------
# CacheAttack-1: direct validation on target model
# ---------------------------------------------------------------------------

class CacheAttack1:
    """
    CacheAttack-1.

    Generates suffix with the Generator, then validates directly on the
    black-box target cache.  A mandatory TTL wait is inserted between
    successive iterations to prevent inter-iteration cache interference.

    Limitations
    -----------
    * High temporal cost (one TTL per iteration).
    * Observable traffic pattern (detectable by traffic volume analysis).
    """

    def __init__(
        self,
        generator: CacheAttackGenerator,
        validator: LatencyValidator,
        cache_api,
        ttl_seconds: float = 60.0,
    ):
        self.generator = generator
        self.validator = validator
        self.cache_api = cache_api
        self.ttl = ttl_seconds

    def attack(
        self,
        p_src: str,
        p_v: str,
        max_iterations: int = 10,
        target_sim: float = 0.88,
        steps_per_len: int = 200,
        init_len: int = 5,
        max_len: int = 40,
        len_step: int = 5,
        batch_size: int = 64,
        top_k: int = 64,
        verbose: bool = False,
    ) -> Tuple[Optional[str], bool]:

        logger.info(
            f"[CacheAttack-1] p_src={p_src[:60]}  p_v={p_v[:60]}"
        )

        for iteration in range(max_iterations):
            logger.info(f"[CacheAttack-1] iteration {iteration + 1}/{max_iterations}")

            # 1. Generate candidate suffix
            suffix, sim = self.generator.run_dynamic_attack(
                p_src=p_src,
                p_v=p_v,
                init_len=init_len,
                max_len=max_len,
                len_step=len_step,
                steps_per_len=steps_per_len,
                batch_size=batch_size,
                top_k=top_k,
                target_sim=target_sim,
                verbose=verbose,
            )

            if suffix is None:
                logger.warning(
                    f"[CacheAttack-1] generator failed (best_sim={sim:.4f})"
                )
                continue

            p_adv = f"{p_src} {suffix}"
            logger.info(f"[CacheAttack-1] candidate: {p_adv[:100]}")
            logger.info(f"[CacheAttack-1] achieved sim={sim:.4f}")

            # 2. Plant adversarial prompt in cache
            self.cache_api.query(p_adv)

            # 3. Wait TTL to avoid inter-iteration interference
            logger.info(
                f"[CacheAttack-1] waiting TTL={self.ttl}s before validation…"
            )
            time.sleep(self.ttl)

            # 4. Validate: query with original victim prompt
            t0 = time.time()
            self.cache_api.query(p_v)
            latency = time.time() - t0

            is_hit, llr = self.validator.predict(latency)
            logger.info(
                f"[CacheAttack-1] latency={latency:.4f}s  "
                f"hit={is_hit}  LLR={llr:.3f}"
            )

            if is_hit:
                logger.info("[CacheAttack-1] ✓ Cache hit confirmed!")
                return p_adv, True

            # Wait another TTL before next attempt
            time.sleep(self.ttl)

        logger.warning("[CacheAttack-1] max_iterations reached without confirmed hit")
        return None, False


# ---------------------------------------------------------------------------
# CacheAttack-2: surrogate-assisted filtering
# ---------------------------------------------------------------------------

class CacheAttack2:
    """
    CacheAttack-2.

    Uses a surrogate semantic cache to evaluate generator candidates at high
    throughput.  The target system is only queried once per iteration
    (the final verification step), eliminating the TTL bottleneck.

    Algorithm per iteration
    -----------------------
    1. Run generator to produce a candidate p_adv.
    2. Evaluate p_adv on the surrogate cache:
       - If surrogate MISS → skip (resume generator loop).
       - If surrogate HIT  → proceed to target verification.
    3. Plant p_adv in the target cache and query p_v once.
    4. If target HIT confirmed → return.
    5. Else → resume optimization from next candidate.
    """

    def __init__(
        self,
        generator: CacheAttackGenerator,
        surrogate_cache,
        target_cache_api=None,
        target_validator: Optional[LatencyValidator] = None,
    ):
        self.generator = generator
        self.surrogate = surrogate_cache
        self.target = target_cache_api
        self.validator = target_validator

    def _surrogate_hit(self, p_adv: str, p_v: str, threshold: float) -> bool:
        """Check if p_adv would hit p_v on the surrogate cache."""
        # Try calling .check_hit if available
        if hasattr(self.surrogate, "check_hit"):
            return self.surrogate.check_hit(p_adv, p_v, threshold)

        # Fallback: compute cosine similarity directly
        enc1 = self.generator.embed_tokenizer(
            p_adv, return_tensors="pt", truncation=True
        ).to(device)
        enc2 = self.generator.embed_tokenizer(
            p_v, return_tensors="pt", truncation=True
        ).to(device)
        with torch.no_grad():
            emb1 = self.generator._get_embedding(enc1["input_ids"], enc1["attention_mask"])
            emb2 = self.generator._get_embedding(enc2["input_ids"], enc2["attention_mask"])
            sim = (emb1 * emb2).sum().item()
        return sim >= threshold

    def attack(
        self,
        p_src: str,
        p_v: str,
        max_iterations: int = 20,
        surrogate_threshold: float = 0.88,
        target_sim: float = 0.88,
        steps_per_len: int = 200,
        init_len: int = 5,
        max_len: int = 40,
        len_step: int = 5,
        batch_size: int = 64,
        top_k: int = 64,
        verbose: bool = False,
    ) -> Tuple[Optional[str], bool]:

        logger.info(
            f"[CacheAttack-2] p_src={p_src[:60]}  p_v={p_v[:60]}"
        )

        for iteration in range(max_iterations):
            logger.info(f"[CacheAttack-2] iteration {iteration + 1}/{max_iterations}")

            # 1. Generate candidate
            suffix, sim = self.generator.run_dynamic_attack(
                p_src=p_src,
                p_v=p_v,
                init_len=init_len,
                max_len=max_len,
                len_step=len_step,
                steps_per_len=steps_per_len,
                batch_size=batch_size,
                top_k=top_k,
                target_sim=target_sim,
                verbose=verbose,
            )

            if suffix is None:
                logger.warning(
                    f"[CacheAttack-2] generator failed (best_sim={sim:.4f}), skipping"
                )
                continue

            p_adv = f"{p_src} {suffix}"

            # 2. Surrogate evaluation (high-throughput proxy)
            surrogate_ok = self._surrogate_hit(p_adv, p_v, surrogate_threshold)
            logger.info(
                f"[CacheAttack-2] surrogate_hit={surrogate_ok}  sim={sim:.4f}"
            )

            if not surrogate_ok:
                logger.info("[CacheAttack-2] Surrogate miss – resuming optimization")
                continue

            # 3. Single query to target system
            if self.target is None:
                # No target available; return candidate as-is
                logger.info(
                    "[CacheAttack-2] No target API – returning surrogate-validated candidate"
                )
                return p_adv, True

            self.target.query(p_adv)  # plant in target cache

            t0 = time.time()
            self.target.query(p_v)    # verify with victim prompt
            latency = time.time() - t0

            if self.validator is not None:
                is_hit, llr = self.validator.predict(latency)
                logger.info(
                    f"[CacheAttack-2] target latency={latency:.4f}s  "
                    f"hit={is_hit}  LLR={llr:.3f}"
                )
            else:
                # Heuristic: if target responds very quickly, assume hit
                is_hit = latency < 0.5
                logger.info(
                    f"[CacheAttack-2] target latency={latency:.4f}s  "
                    f"hit={is_hit} (heuristic)"
                )

            if is_hit:
                logger.info("[CacheAttack-2] ✓ Target cache hit confirmed!")
                return p_adv, True

            logger.info(
                "[CacheAttack-2] Target miss – resuming optimization from next candidate"
            )

        logger.warning("[CacheAttack-2] max_iterations reached without confirmed hit")
        return None, False


class SurrogateCacheEmbeddingModel:
    def __init__(self, embed_model_name: str, threshold: float = 0.88):
        self.threshold = threshold
        self._tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self._model = AutoModel.from_pretrained(embed_model_name).to(device)
        self._model.eval()

    @torch.no_grad()
    def _embed(self, text: str) -> torch.Tensor:
        enc = self._tokenizer(text, return_tensors="pt", truncation=True).to(device)
        out = self._model(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, dim=1)

    def check_hit(self, p_adv: str, p_v: str, threshold: Optional[float] = None) -> bool:
        tau = threshold if threshold is not None else self.threshold
        e1 = self._embed(p_adv)
        e2 = self._embed(p_v)
        return float((e1 * e2).sum().item()) >= tau
