import numpy as np
from scipy.stats import norm
from typing import List, Tuple, Optional
import time
import json
from openai import OpenAI

ATTACKER_OPENAI_KEY = "sk-proj-......" 

class CacheThresholdEstimator:
    """
    Estimates the operational threshold for semantic cache hit/miss detection
    using latency-based Gaussian mixture model.
    """
    
    def __init__(self, cache_api, embedding_model):
        """
        Args:
            cache_api: The target cache API to query
            embedding_model: The surrogate embedding model
        """
        self.cache_api = cache_api
        self.embedding_model = embedding_model
        
        # Parameters for hit/miss distributions
        self.mu_hit = None  # Mean latency for cache hits
        self.sigma_hit = None  # Std dev for cache hits
        self.mu_miss = None  # Mean latency for cache misses
        self.sigma_miss = None  # Std dev for cache misses
        
        # Estimated threshold
        self.threshold = None
        self.client = OpenAI(api_key=ATTACKER_OPENAI_KEY)
        
    def calibrate_latency_distributions(
        self, 
        seed_prompt: str,
        n_hit_samples: int = 20,
        n_miss_samples: int = 20
    ) -> None:
        """
        Calibrate the latency distributions for cache hits and misses.
        
        Args:
            seed_prompt: A seed prompt to generate calibration queries
            n_hit_samples: Number of samples for estimating hit distribution
            n_miss_samples: Number of samples for estimating miss distribution
        """
        # Collect hit samples: repeated identical requests
        hit_latencies = []
        print(f"Collecting {n_hit_samples} hit samples...")
        
        # First query to populate cache
        _ = self.cache_api.query(seed_prompt)
        time.sleep(0.5)  # Short wait to ensure cache is populated
        
        for i in range(n_hit_samples):
            start_time = time.time()
            _ = self.cache_api.query(seed_prompt)
            latency = time.time() - start_time
            hit_latencies.append(latency)
            time.sleep(0.1)  # Small delay between queries
            
        # Collect miss samples: nonce-augmented queries (guaranteed misses)
        miss_latencies = []
        print(f"Collecting {n_miss_samples} miss samples...")
        
        for i in range(n_miss_samples):
            # Add unique nonce to ensure cache miss
            nonce_prompt = f"{seed_prompt} [NONCE_{i}_{np.random.randint(0, 10000)}]"
            start_time = time.time()
            _ = self.cache_api.query(nonce_prompt)
            latency = time.time() - start_time
            miss_latencies.append(latency)
            time.sleep(0.1)
            
        # Convert to log-latency space
        log_hit_latencies = np.log(hit_latencies)
        log_miss_latencies = np.log(miss_latencies)
        
        # Estimate parameters using MLE (maximum likelihood estimation)
        self.mu_hit = np.mean(log_hit_latencies)
        self.sigma_hit = np.std(log_hit_latencies)
        self.mu_miss = np.mean(log_miss_latencies)
        self.sigma_miss = np.std(log_miss_latencies)
        
        print(f"Hit distribution: μ={self.mu_hit:.4f}, σ={self.sigma_hit:.4f}")
        print(f"Miss distribution: μ={self.mu_miss:.4f}, σ={self.sigma_miss:.4f}")
        
    def predict_hit_or_miss(self, latency: float) -> Tuple[bool, float]:
        """
        Predict whether a query resulted in cache hit or miss using MAP rule.
        
        Args:
            latency: Observed query latency in seconds
            
        Returns:
            (is_hit, confidence): Boolean hit prediction and confidence score
        """
        if self.mu_hit is None or self.mu_miss is None:
            raise ValueError("Must calibrate distributions first!")
            
        # Convert to log-latency
        log_latency = np.log(latency)
        
        # Compute log-likelihood ratio
        # log P(Y|H=1) - log P(Y|H=0)
        ll_hit = norm.logpdf(log_latency, self.mu_hit, self.sigma_hit)
        ll_miss = norm.logpdf(log_latency, self.mu_miss, self.sigma_miss)
        
        log_likelihood_ratio = ll_hit - ll_miss
        
        # MAP rule: predict hit if log-likelihood ratio > 0
        is_hit = log_likelihood_ratio > 0
        
        # Confidence can be derived from likelihood ratio
        confidence = abs(log_likelihood_ratio)
        
        return is_hit, confidence
    
    def generate_probes(self, seed_prompt, n):
        print(f"[*] Generating {n} semantic probes using GPT-4...")
        with open("probe_prompt.txt", "r") as f:
            prompt = f.read().replace("{{INSERT_YOUR_QUERY_HERE}}", seed_prompt).replace("{{N}}", str(n))
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            variations = data.get("variations", [])
            if seed_prompt not in variations: variations.insert(0, seed_prompt)

            print(f"[*] Computing similarities for {len(variations)} probes...")
            seed_embedding = self.embedding_model.encode(seed_prompt)
            probes = []
            for variation in variations:
                var_embedding = self.embedding_model.encode(variation)
                similarity = self._cosine_similarity(seed_embedding, var_embedding)
                probes.append((variation, similarity))
            
            # Sort by similarity descending
            probes.sort(key=lambda x: x[1], reverse=True)
            return probes
        except Exception as e:
            print(f"[Error] {e}")
            return []
    
    def binary_search_threshold(
        self, 
        seed_prompt: str,
        n_probes: int = 50,
        max_queries: int = 10
    ) -> float:
        """
        Estimate the operational threshold using binary search over probe set.
        
        Args:
            seed_prompt: Seed prompt for generating probes
            n_probes: Number of probes to generate
            max_queries: Maximum number of queries (O(log N))
            
        Returns:
            Estimated threshold τ̂
        """
        # Generate probe set
        print(f"Generating {n_probes} probe variations...")
        probes = self.generate_probes(seed_prompt, n_probes)
        
        # First, insert seed_prompt into cache
        print("Populating cache with seed prompt...")
        _ = self.cache_api.query(seed_prompt)
        time.sleep(1.0)  # Wait for cache to stabilize
        
        # Binary search over probes
        left, right = 0, len(probes) - 1
        lowest_hit_idx = -1  # Lowest similarity that still hits
        highest_miss_idx = len(probes)  # Highest similarity that misses
        
        queries_used = 0
        print("Starting binary search for threshold...")
        
        while left <= right and queries_used < max_queries:
            mid = (left + right) // 2
            probe_text, probe_sim = probes[mid]
            
            # Query the cache
            start_time = time.time()
            response = self.cache_api.query(probe_text)
            latency = time.time() - start_time
            queries_used += 1
            
            # Predict hit or miss
            is_hit, confidence = self.predict_hit_or_miss(latency)
            
            print(f"Query {queries_used}: sim={probe_sim:.4f}, "
                  f"latency={latency:.4f}s, hit={is_hit}, conf={confidence:.4f}")
            
            if is_hit:
                lowest_hit_idx = mid
                left = mid + 1  # Search for lower similarities that still hit
            else:
                highest_miss_idx = mid
                right = mid - 1  # Search for higher similarities
                
            # Wait before next query to avoid rate limiting
            time.sleep(0.5)
        
        # Estimate threshold as midpoint
        if lowest_hit_idx == -1:
            # All probes missed
            self.threshold = 1.0
        elif highest_miss_idx == len(probes):
            # All probes hit
            self.threshold = 0.0
        else:
            # Midpoint between lowest hit and highest miss
            hit_sim = probes[lowest_hit_idx][1]
            miss_sim = probes[highest_miss_idx][1]
            self.threshold = (hit_sim + miss_sim) / 2.0
            
        print(f"\nEstimated threshold: τ̂ = {self.threshold:.4f}")
        print(f"Total queries used: {queries_used}")
        
        return self.threshold
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


