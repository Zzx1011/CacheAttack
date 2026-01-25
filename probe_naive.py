import json
import re
import time
import requests
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# ================= 攻击者配置 =================
ATTACKER_OPENAI_KEY = "sk-proj-......" # 攻击者自己的 Key (用于生成探针)
TARGET_URL = "http://127.0.0.1:5000/chat"
SEED_URL = "http://127.0.0.1:5000/admin/seed"

TARGET_QUERY = "What is the current capital requirement?"
LOCAL_MODEL = "all-MiniLM-L6-v2"
# ==============================================

class RealWorldProber:
    def __init__(self):
        self.client = OpenAI(api_key=ATTACKER_OPENAI_KEY)
        self.local_model = SentenceTransformer(LOCAL_MODEL)
        self.timing_threshold = 0.0 # 将通过校准自动设置

    def calibrate_network_latency(self):
        """
        【关键步骤】动态校准网络延迟。
        由于真实 LLM 响应时间波动，我们需要先测量基线。
        """
        print("[*] Calibrating network and LLM latency...")
        
        # 1. 确保目标已被缓存 (HIT Baseline)
        requests.post(SEED_URL)
        
        # 测量 HIT 延迟 (取 3 次平均)
        hit_times = []
        for _ in range(5):
            start = time.time()
            requests.post(TARGET_URL, json={"query": TARGET_QUERY})
            hit_times.append(time.time() - start)
        avg_hit = sum(hit_times) / len(hit_times)
        
        # 2. 测量 MISS 延迟 (使用一个随机乱码查询，确保穿透到 LLM)
        # 注意：这会消耗服务器少量的 token
        miss_times = []
        for i in range(5):
            random_q = f"Random query for calibration {time.time()} {i}"
            start = time.time()
            requests.post(TARGET_URL, json={"query": random_q})
            miss_times.append(time.time() - start)
        avg_miss = sum(miss_times) / len(miss_times)
        
        # 3. 设定阈值
        # 阈值设为 HIT 和 MISS 的几何平均或中间点
        self.timing_threshold = (avg_hit + avg_miss) / 2
        
        print(f"    Avg HIT Latency : {avg_hit:.4f}s")
        print(f"    Avg MISS Latency: {avg_miss:.4f}s (Real LLM Call)")
        print(f"    [!] Dynamic Threshold set to: {self.timing_threshold:.4f}s")
        
        if avg_miss < avg_hit * 2:
            print("[WARNING] Distinction between HIT and MISS is small. Attack may be unstable.")
            sys.exit(1)

    def generate_probes(self, n):
        print(f"[*] Generating {n} semantic probes using GPT-4...")
        with open("probe_prompt.txt", "r") as f:
            prompt = f.read().replace("{{INSERT_YOUR_QUERY_HERE}}", TARGET_QUERY).replace("{{N}}", str(n))
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            variations = data.get("variations", [])
            if TARGET_QUERY not in variations: variations.insert(0, TARGET_QUERY)
            return variations
        except Exception as e:
            print(f"[Error] {e}")
            return []

    def measure_latency(self, text):
        """Send a query and measure elapsed time."""
        start = time.time()
        requests.post(TARGET_URL, json={"query": text})
        return time.time() - start
    
    def em_infer_threshold(self, sims, latencies, max_iter=100):
        """
        EM to fit a 2-component Gaussian mixture for latency.
        One corresponds to HIT, one to MISS.
        The boundary is estimated in similarity space.
        """
        t = np.array(latencies)
        s = np.array(sims)
        
        # Init: k-means-like split based on timing threshold
        z = (t < self.timing_threshold).astype(float)

        mu_H = t[z == 1].mean()
        mu_M = t[z == 0].mean()
        sigma_H = t[z == 1].std() + 1e-4
        sigma_M = t[z == 0].std() + 1e-4

        tau = np.median(s)

        for _ in range(max_iter):
            # E-step: compute posterior HIT probability for each sample
            p_hit = (1.0 / (np.sqrt(2*np.pi)*sigma_H)) * np.exp(-0.5*((t - mu_H)/sigma_H)**2)
            p_miss = (1.0 / (np.sqrt(2*np.pi)*sigma_M)) * np.exp(-0.5*((t - mu_M)/sigma_M)**2)

            z = p_hit / (p_hit + p_miss + 1e-9)

            # M-step: update means/variances
            mu_H = np.sum(z * t) / np.sum(z)
            mu_M = np.sum((1-z) * t) / np.sum((1-z))

            sigma_H = np.sqrt(np.sum(z * (t-mu_H)**2) / np.sum(z))
            sigma_M = np.sqrt(np.sum((1-z) * (t-mu_M)**2) / np.sum((1-z)))

            # Update threshold: similarity where posterior HIT ≈ 0.5
            tau = s[np.argmin(np.abs(z - 0.5))]

        return tau

    # -------------------------------------------------------
    # Bootstrap confidence interval
    # -------------------------------------------------------
    def bootstrap_ci(self, sims, latencies, B=200):
        ests = []
        n = len(sims)
        for _ in range(B):
            idx = np.random.randint(0, n, n)
            s_b = np.array(sims)[idx]
            t_b = np.array(latencies)[idx]
            ests.append(self.em_infer_threshold(s_b, t_b, max_iter=50))
        return np.mean(ests), np.percentile(ests, 5), np.percentile(ests, 95)

    def attack(self):
        # Step 0: 植入目标 & 校准
        self.calibrate_network_latency()
        
        # Step 1: 生成探针
        probes = self.generate_probes(n=100)
        
        # Step 2: 本地排序
        print("[*] Ranking probes locally...")
        target_emb = self.local_model.encode([TARGET_QUERY])
        probe_embs = self.local_model.encode(probes)
        sims = cosine_similarity(target_emb, probe_embs)[0]
        
        ranked = [{"text": p, "score": float(s)} for p, s in zip(probes, sims)]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        # # Step 3: measure latency for every probe
        # print("\n[*] Measuring latency for all probes...")
        # # latencies = [self.measure_latency(p) for p in probes]
        # latencies = []
        # print(f"{'IDX':<4} | {'Sim':<8} | {'Latency':<10} | Status | Text")
        # print("-" * 80)

        # for i, p in enumerate(probes):
        #     latency = self.measure_latency(p)
        #     latencies.append(latency)

        #     sim = sims[i]
        #     status = "HIT" if latency < self.timing_threshold else "MISS"

        #     print(f"{i:<4} | {sim:.4f} | {latency:.4f}s | {status:<5} | {p[:60]}")

        # print("\n========== STEP 3: EM Threshold Inference ==========")
        # raw_tau = self.em_infer_threshold(sims, latencies)
        # print(f"Estimated threshold (EM): {raw_tau:.4f}")

        # print("\n========== STEP 4: Bootstrap CI ==========")
        # mean_tau, low_ci, high_ci = self.bootstrap_ci(sims, latencies)
        # print(f"Bootstrap Mean τ: {mean_tau:.4f}")
        # print(f"95% CI: [{low_ci:.4f}, {high_ci:.4f}]")

        # print("\n========== DONE ==========")
        
        # Step 3: 二分查找
        print("\n[*] Starting Binary Search Attack...")
        print(f"    {'IDX':<4} | {'Local Sim':<10} | {'Latency':<10} | {'Status'}")
        print("-" * 50)
        
        low = 0
        high = len(ranked) - 1
        boundary_idx = -1
        
        while low <= high:
            mid = (low + high) // 2
            probe = ranked[mid]
            
            # 测量时间
            start = time.time()
            requests.post(TARGET_URL, json={"query": probe['text']})
            latency = time.time() - start
            
            # 判定
            is_hit = latency < self.timing_threshold
            status = "HIT" if is_hit else "MISS"
            
            print(f"    {mid:<4} | {probe['score']:.4f}     | {latency:.4f}s   | {status}")
            
            if is_hit:
                boundary_idx = mid
                low = mid + 1 # 向更不相似的方向探索
            else:
                high = mid - 1 # 向更相似的方向回退
        
        # Step 4: 结果分析
        if boundary_idx != -1 and boundary_idx < len(ranked) - 1:
            hit_p = ranked[boundary_idx]
            miss_p = ranked[boundary_idx + 1]
            inferred_threshold = (hit_p['score'] + miss_p['score']) / 2
            
            print("\n" + "="*50)
            print(" [SUCCESS] Real-world Boundary Inferred")
            print("="*50)
            print(f" Server Threshold (Hidden) : 0.92")
            print(f" Inferred Threshold (Local): {inferred_threshold:.4f}")
            print(f" Accuracy Gap              : {abs(0.92 - inferred_threshold):.4f}")
            print("="*50)
        else:
            print("\n[Fail] Could not determine boundary.")
        
        return inferred_threshold

if __name__ == "__main__":
    attacker = RealWorldProber()
    inferred_threshold = attacker.attack()