# CacheAttack: Key Collision Attack on LLM Semantic Caching

<p align="center">
  <a href="https://arxiv.org/abs/2601.23088"><img src="https://img.shields.io/badge/arXiv-2601.23088-b31b1b.svg" alt="arXiv"></a>
  <a href="https://icml.cc/virtual/2026"><img src="https://img.shields.io/badge/ICML-2026-blue.svg" alt="ICML 2026"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> **From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching**
> ICML 2026

---

## Overview

Semantic caching accelerates LLM inference by returning cached responses to semantically similar queries, using embedding similarity rather than exact string matching. This repository provides the official implementation of **CacheAttack**, demonstrating that this core mechanism is itself an exploitable attack surface.

We show that a black-box attacker, given only API-level access to a semantic cache, can:

1. **Estimate the operational similarity threshold** without any white-box access, using a latency-based Gaussian mixture model and O(log N) queries (binary search over a semantically ordered probe set).
2. **Inject a poisoned cache entry** whose embedding is crafted to collide with the key space of a target query class, redirecting future victim queries to attacker-controlled responses.
3. **Exfiltrate private cached content** by exploiting cross-user cache sharing in cloud-hosted deployments (AWS Bedrock, Azure API Management).

Our experiments cover both self-hosted caches (GPTCache/FAISS, LangChain) and production cloud APIs, demonstrating high attack success rates under default configurations.

---

## Repository Structure

```
CacheAttack/
├── attack/                        # Core attack implementations
├── SemanticKVcache/src/           # Reference semantic cache server
├── evaluation/                    # Evaluation scripts and metrics
├── data/                          # Datasets and seed prompts
├── probe.py                       # Threshold estimator (GMM + binary search)
├── probe_naive.py                 # Naive sequential probing baseline
├── probe_prompt.txt               # LLM prompt template for probe generation
├── server_gpt_cache_modified.py   # GPTCache server with verbose logging
├── server_langchain_full.py       # LangChain-based semantic cache server
└── utils.py                       # Verbose similarity/FAISS wrappers for debugging
```

---

## Attack Pipeline

```
Step 1: Threshold Estimation
  └─ Calibrate hit/miss latency distributions (Gaussian mixture, log-latency space)
  └─ Generate N semantic probes via surrogate LLM + embedding model
  └─ Binary search over probe set → estimate τ̂ in O(log N) queries

Step 2: Key Collision Crafting
  └─ Construct adversarial query q_adv such that sim(embed(q_adv), embed(q_target)) ≥ τ̂
  └─ Inject q_adv with malicious response into cache

Step 3: Victim Redirection
  └─ Victim query q_victim hits cache → receives attacker response
```

---

## Setup

**Requirements:** Python 3.9+, an OpenAI API key (attacker side), and optionally AWS/Azure credentials for cloud experiments.

```bash
git clone https://github.com/Zzx1011/CacheAttack.git
cd CacheAttack

# Install dependencies
pip install openai gptcache faiss-cpu sentence-transformers langchain scipy numpy flask
```

Set your API key before running probing experiments:

```python
# In probe.py, replace the placeholder:
ATTACKER_OPENAI_KEY = "sk-proj-YOUR_KEY_HERE"
```

---

## License

MIT License. See `LICENSE` for details.
