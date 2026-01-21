import re
import time
import os

# --- 1. LangChain Imports (Compatible with langchain 1.1.3) ---
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool

# --- 2. GPTCache Imports (Latest recommended way) ---
from gptcache.adapter.api import init_similar_cache
from gptcache import Config
from gptcache.embedding import Huggingface
from gptcache.manager import get_data_manager
from gptcache.manager import manager_factory, CacheBase, VectorBase
from langchain_core.globals import set_llm_cache
from langchain_community.cache import GPTCache
import hashlib

# --- Configuration ---
os.environ["OPENAI_API_KEY"] = "sk-proj-..."  # Replace with your real key

class SemanticCacheAgentSystem:
    """
    Semantic cache + ReAct Agent system compatible with LangChain 1.1.3.
    Uses official init_similar_cache for easy multilingual semantic caching.
    """

    def __init__(self, 
                 embedding_model, 
                 similarity_threshold: float = 0.93, 
                 llm_model_name: str = "gpt-4o-mini"):
        self.llm_model_name = llm_model_name
        self.agent = None
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        print(f"Initializing SemanticCacheAgentSystem with {llm_model_name}...")
        self._initialize_cache()
        self._initialize_llm_and_agent()
        print("Initialization complete.")

    def _initialize_cache(self):
        """Use official init_similar_cache for semantic caching (multilingual supported)."""
        def get_hashed_name(name: str) -> str:
            return hashlib.sha256(name.encode()).hexdigest()

        def init_gptcache(cache_obj, llm_str: str):
            hashed_llm = get_hashed_name(llm_str)
            config = Config(similarity_threshold=self.similarity_threshold)
            data_dir = f"gptcache_data_{hashed_llm}"

            # eviction_base=EvictionBase("redis",  maxmemory="0", policy="noeviction", ttl=1))
            data_manager = get_data_manager(
                cache_base=CacheBase('sqlite', path=os.path.join(data_dir, "sqlite.db")),
                vector_base=VectorBase('faiss', dimension=self.embedding_model.dimension, path=os.path.join(data_dir, "faiss.index")),
                data_path=data_dir,
            )
            
            # One-line setup: exact + semantic cache with FAISS + local multilingual embedding
            init_similar_cache(
                cache_obj=cache_obj,
                data_dir=data_dir,  # Avoid conflicts between different models
                config=config,
                embedding=self.embedding_model,  # Multilingual embedding model
                data_manager=data_manager,
            )

        # Set global LLM cache (caches all LLM calls, including those inside tools)
        set_llm_cache(GPTCache(init_gptcache))

    def _initialize_llm_and_agent(self):
        """Create LLM and ReAct agent."""
        llm = ChatOpenAI(model=self.llm_model_name, temperature=0)

        # --- Define Tools ---
        @tool
        def calculator(expression: str) -> str:
            """Useful for math calculations. Input is a valid Python expression."""
            try:
                return str(eval(expression, {"__builtins__": {}}))  # Safe eval
            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        def general_knowledge(question: str) -> str:
            """Useful for factual questions (history, people, science, etc.)."""
            # This direct LLM call will be cached semantically (including multilingual hits)
            result = llm.invoke(question)
            print(result)
            return result.content

        tools = [calculator, general_knowledge]

        # --- Manual ReAct Prompt (classic format) ---
        react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        # --- Create ReAct Agent ---
        self.agent = create_agent(
            llm,   ### THIS IS THE WEIRD POINT!!!! ###
            tools=tools,
            system_prompt=react_prompt_template,
        )
    
    def test(self):
        """Simple test function to verify setup."""
        response = self.agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Calculate 12 * 15 + 100"}
                ]
            }
        )
        print(response)

    def run_query(self, query_text: str, run_name: str):
        """Execute query and measure time."""
        print(f"\n--- {run_name} ---")
        print(f"Query: {query_text}")
        start_time = time.time()
        
        try:
            result = self.agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": query_text}
                    ]
                }
            )
            print(result)
            output = result["messages"][-1].content
            print(f"Result: {output}")
        except Exception as e:
            output = str(e)
            print(f"Error: {e}")

        duration = time.time() - start_time
        print(f"Time: {duration:.4f}s")
        return output


# --- Execution ---
if __name__ == "__main__":
    embedding = Huggingface(model="intfloat/multilingual-e5-small")
    # print("Embedding Dimension:", embedding.dimension)
    system = SemanticCacheAgentSystem(embedding_model=embedding, similarity_threshold=0.93)

    # answer = system.run_query("Calculate 50 * 15 + 200", "Run 1: Math (Cold)")
    answer = system.test()
    # answer = system.run_query("Calculate 50 times 15 plus 200", "Run 2: Math (Semantic Cache Hit Expected in Tool)")
    # print(answer)
    # system.run_query("Who is the founder of Microsoft?", "Run 3: Knowledge (Cold)")
    # system.run_query("微软的创始人是谁？", "Run 4: Knowledge (Multilingual Cache Hit Expected in Tool)")