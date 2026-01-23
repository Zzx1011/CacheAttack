import os
import time
import hashlib
from utils import VerboseSimilarityEvaluation, VerboseFaissVectorBase

# ========= LangChain =========
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_classic.prompts import PromptTemplate

# ========= GPTCache =========
from gptcache.adapter.api import init_similar_cache
from gptcache import Config
from gptcache.embedding import Huggingface
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from langchain_core.globals import set_llm_cache
from langchain_community.cache import GPTCache

# ========= ENV =========
os.environ["OPENAI_API_KEY"] = "sk-proj-..."  

class SemanticCacheAgentSystem:
    """
    ReAct Agent + GPTCache (semantic cache) system.
    Compatible with LangChain 1.1.x
    """

    def __init__(
        self,
        embedding_model,
        similarity_threshold: float = 0.93,
        llm_model_name: str = "gpt-4o-mini",
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.llm_model_name = llm_model_name

        print(f"[Init] LLM = {llm_model_name}")
        self._init_cache()
        self._init_agent()
        print("[Init] Done")

    # ------------------------------------------------------------------
    # 1. GPTCache initialization
    # ------------------------------------------------------------------
    def _init_cache(self):
        def _hash(name: str) -> str:
            return hashlib.sha256(name.encode()).hexdigest()

        def init_gptcache(cache_obj, llm_str: str):
            hashed = _hash(llm_str)
            print(f"[Init] GPTCache data dir hash: {hashed}")   
            data_dir = f"./gptcache_data_{hashed}"
            os.makedirs(data_dir, exist_ok=True)

            config = Config(similarity_threshold=self.similarity_threshold,
                            # similarity_evaluation=VerboseSimilarityEvaluation(threshold=self.similarity_threshold)
                            )

            data_manager = get_data_manager(
                cache_base=CacheBase(
                    "sqlite",
                    path=os.path.join(data_dir, "sqlite.db"),
                ),
                vector_base=VectorBase(
                    "faiss",
                    dimension=self.embedding_model.dimension,
                    path=os.path.join(data_dir, "faiss.index"),
                ),
                data_path=data_dir,
            )

            init_similar_cache(
                cache_obj=cache_obj,
                data_dir=data_dir,
                config=config,
                embedding=self.embedding_model,
                data_manager=data_manager,
            )

        # Set global cache (ALL LLM calls go through this)
        set_llm_cache(GPTCache(init_gptcache))

    # ------------------------------------------------------------------
    # 2. ReAct Agent initialization
    # ------------------------------------------------------------------
    def _init_agent(self):
        llm = ChatOpenAI(
            model=self.llm_model_name,
            temperature=0,
        )

        # ---------------- Tools ----------------
        @tool
        def calculator(expression: str) -> str:
            """Evaluate a Python math expression."""
            try:
                return str(eval(expression, {"__builtins__": {}}))
            except Exception as e:
                return f"Error: {e}"

        @tool
        def general_knowledge(question: str) -> str:
            """Answer general factual questions."""
            # IMPORTANT:
            # - This LLM call is cached by GPTCache
            # - Input/output are plain strings
            return llm.invoke(question).content

        tools = [calculator, general_knowledge]

        # ---------------- ReAct Prompt ----------------
        react_prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: reasoning about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
... (this Thought/Action/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
        )

        # ---------------- ReAct Agent ----------------
        react_agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=react_prompt,
        )

        self.agent = AgentExecutor(
            agent=react_agent,
            tools=tools,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # 3. Public API
    # ------------------------------------------------------------------
    def run(self, query: str):
        print("\n=== Query ===")
        print(query)
        start = time.time()

        result = self.agent.invoke({"input": query})

        print("\n=== Result ===")
        print(result["output"])
        print(f"\nTime: {time.time() - start:.3f}s")

        return result["output"]


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    embedding = Huggingface(model="intfloat/multilingual-e5-small")
    system = SemanticCacheAgentSystem(
        embedding_model=embedding,
        similarity_threshold=0.80,
    )

    system.run("Calculate 12 * 15 + 100")
    system.run("Calculate twelve times fifteen plus one hundred")