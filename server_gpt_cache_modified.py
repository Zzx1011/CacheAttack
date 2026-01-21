import os
import shutil
import time
from flask import Flask, request, jsonify
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import sqlite3

# GPTCache 库
from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.embedding import Huggingface

app = Flask(__name__)

# ================= 真实服务器配置 =================
# 请在此填入服务器端的 API Key
SERVER_OPENAI_API_KEY = "sk-proj-......"
# 攻击者未知的、硬编码在系统中的阈值
# 任何相似度低于 0.92 的请求都会触发真实的扣费 API 调用
HIDDEN_THRESHOLD = 0.80 #0.92

# 初始化 OpenAI 客户端
client = OpenAI(api_key=SERVER_OPENAI_API_KEY)
# ==================================================

# 1. 环境清理 (保证实验可复现性)
if os.path.exists("gptcache_data"):
    shutil.rmtree("gptcache_data")
    print("[Server] Cleared old cache data.")

# 2. 初始化 GPTCache 组件 (工业级标准写法)
print("[Server] Initializing GPTCache (Onnx + Faiss)...")

# 使用 Onnx 运行轻量级嵌入模型 (避免 PyTorch 依赖过重)
# encoder = Onnx(model="GPTCache/paraphrase-albert-onnx") 
# encoder = Huggingface(model="BAAI/bge-small-en-v1.5")
encoder = Huggingface(model="intfloat/multilingual-e5-small")
print(f"    Embedding Dimension: {encoder.dimension}")

# 使用 SQLite 管理元数据，Faiss 管理向量索引
data_manager = manager_factory(
    "sqlite,faiss", 
    data_dir="gptcache_data", 
    vector_params={"dimension": encoder.dimension, 
                   "top_k": 1}
)

# 基于距离的评估器
evaluator = SearchDistanceEvaluation()

# 初始化缓存对象
cache = Cache()
cache.init(
    embedding_func=encoder.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=evaluator,
)
# cache.data_manager.report_cache()
print("Faiss index dimension =", cache.data_manager.v._index.d)

def real_llm_call(user_query):
    """
    真实的 LLM 调用。
    这将产生真实的延迟 (Network + Token Generation) 和真实的成本。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 使用便宜且快速的模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_query}
            ],
            max_completion_tokens=50, # 限制输出长度以保持延迟相对稳定
            # temperature=0.001
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Server Error] LLM Call failed: {e}")
        return "Service unavailable due to upstream error."

def normalized_embedding_func(text):
    def normalize(vec):
        # vec shape: (dim,) or (1, dim)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    emb = encoder.to_embeddings(text)
    # Ensure it's a numpy array and float32
    if not isinstance(emb, np.ndarray):
        emb = np.array(emb)
    # Normalize query/data before entering FAISS
    # GPTCache expects shape (1, dim) or (dim,)
    norm_emb = normalize(emb)
    return norm_emb

# curl -X POST http://127.0.0:5000/direct_call  -H "Content-Type: application/json"  -d '{"query": "What is the current capital requirement?"}'
@app.route('/direct_call', methods=['POST'])
def direct_call():
    data = request.json
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    start_time = time.time()
    
    answer = real_llm_call(query)

    latency = time.time() - start_time
    
    return jsonify({
        'response': answer,
        'server_latency': round(latency, 4),
        'debug_status': 'DIRECT_CALL',
        'similarity': None
    })

# curl -X POST http://127.0.0.1:5000/chat  -H "Content-Type: application/json"  -d '{"query": "What is the current capital requirement?"}'
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    start_time = time.time()
    
    # 生成查询向量
    query_emb = normalized_embedding_func(query).reshape(-1)

    # 向量检索（返回已按相似度排序的候选列表）
    search_res = cache.data_manager.search(query_emb, top_k=1)

    answer = None
    is_hit = False
    similarity_score = 0.0

    # ========== 关键修复点 ==========
    if search_res and len(search_res) > 0:
        best_match = search_res[0]
        distance, best_match_data = best_match  # 解包：距离 + 数据对象
        object_id = best_match_data
        print(f"Object ID: {object_id}")
        # FAISS returns L2 Distance (d).
        # Since vectors are normalized: Cosine Similarity = 1 - (d^2) / 2
        similarity_score = 1 - (distance ** 2) / 2.0

        if similarity_score >= HIDDEN_THRESHOLD:
            try:
                conn = sqlite3.connect("gptcache_data/sqlite.db")
                cursor = conn.cursor()
                cursor.execute("SELECT answer FROM gptcache_answer WHERE id = ?", (object_id,))
                row = cursor.fetchone()
                conn.close()

                if row and row[0] is not None:
                    answer = row[0]
                    is_hit = True
                    print(f"[Cache] HIT  | Sim: {similarity_score:.4f} | ID: {object_id} | Answer retrieved")
                else:
                    print(f"[Cache] HIT but answer is empty for ID: {object_id}")
            except Exception as e:
                print(f"[Cache] Failed to fetch answer for ID {object_id}: {e}")
    # =====================================

    if not is_hit:
        print(f"[Cache] MISS | Sim: {similarity_score:.4f} | Calling real LLM...")
        answer = real_llm_call(query)
        cache.data_manager.save(query, answer, query_emb)

    latency = time.time() - start_time
    
    return jsonify({
        'response': answer,
        'server_latency': round(latency, 4),
        'debug_status': 'HIT' if is_hit else 'MISS',
        'similarity': round(similarity_score, 4).item()  # 方便你观察阈值
    })

# curl -X POST http://127.0.0.1:5000/admin/seed
@app.route('/admin/seed', methods=['POST'])
def seed():
    """
    预热端点：为了实验，攻击者/管理员预先向缓存中写入一个“目标查询”。
    """
    target = "What is the current capital requirement?"
    # 预先调用一次 LLM 获取真实答案并存入
    print(f"[Server] Seeding target: {target}")
    # 模拟这一步是为了确保缓存里有数据
    emb = encoder.to_embeddings(target).reshape(-1)
    # print(emb)
    # print(type(emb), emb.shape)
    ans = "Regulatory Capital: Tier 1 capital ratio must be at least 6%."
    cache.data_manager.save(target, ans, emb)
    return jsonify({'status': 'seeded', 'target': target})

# curl http://127.0.0.1:5000/lookup
@app.route('/lookup', methods=['GET'])
def lookup():
    """
    管理端点：根据 GPTCache 实际的 SQLite schema 正确读取缓存内容。
    适配表结构：
        gptcache_question (id, question)
        gptcache_answer (id, answer)
    """
    db_path = "gptcache_data/sqlite.db"

    if not os.path.exists(db_path):
        return jsonify({'error': f'SQLite file not found at {db_path}'}), 500

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 关联 question + answer
        cursor.execute("""
            SELECT q.id, q.question, a.answer
            FROM gptcache_question AS q
            LEFT JOIN gptcache_answer AS a
            ON q.id = a.id
            ORDER BY q.id;
        """)

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'question': row[1],
                'answer': row[2]
            })

        return jsonify({
            'count': len(results),
            'items': results
        })

    except Exception as e:
        return jsonify({'error': f'sqlite query failed: {str(e)}'}), 500
    
# curl -X POST http://127.0.0.1:5000/flush_cache
@app.route("/flush_cache", methods=["POST"])
def flush_cache():
    try:
        db_path = "gptcache_data/sqlite.db"
        # 1. 清空 SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # GPTCache 默认表结构
        tables = [
            "gptcache_question",
            "gptcache_answer",
            "gptcache_question_dep",
            "gptcache_session",
            "gptcache_report"
        ]

        for t in tables:
            cursor.execute(f"DELETE FROM {t};")

        # cursor.execute("VACUUM;")

        conn.commit()
        conn.close()
        
        conn2 = sqlite3.connect(db_path)
        conn2.isolation_level = None  # 关键！使 VACUUM 不在事务内
        cursor2 = conn2.cursor()
        cursor2.execute("VACUUM;")
        conn2.close()
        
        return jsonify({
            'status': 'ok',
            'message': 'All cache cleared (SQLite + Faiss).'
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# curl http://127.0.0.1:5000/health
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

if __name__ == '__main__':
    print(f"[Server] Running on port 5000. Real LLM mode enabled.")
    app.run(host='0.0.0.0', port=5000, debug=False)