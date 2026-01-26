# financial_agent_react.py
"""
Financial Agent using ReAct (Reasoning + Acting) framework
With semantic cache integration
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests


class AgentState(Enum):
    """Agent execution state"""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"


@dataclass
class Thought:
    """Agent's thought/reasoning"""
    content: str
    
    def __str__(self):
        return f"Thought: {self.content}"


@dataclass
class Action:
    """Agent's action (tool call)"""
    tool: str
    arguments: Dict[str, Any]
    
    def __str__(self):
        return f"Action: {self.tool}({json.dumps(self.arguments)})"


@dataclass
class Observation:
    """Observation from action execution"""
    content: str
    success: bool
    
    def __str__(self):
        return f"Observation: {self.content}"


class Portfolio:
    """Portfolio management"""
    
    def __init__(self, initial_holdings: Dict[str, int] = None):
        self.holdings = initial_holdings or {}
        self.transaction_history = []
    
    def get_holding(self, stock: str) -> int:
        return self.holdings.get(stock, 0)
    
    def set_holding(self, stock: str, amount: int):
        self.holdings[stock] = amount
    
    def get_state(self) -> Dict[str, int]:
        return self.holdings.copy()
    
    def add_transaction(self, transaction: Dict):
        self.transaction_history.append(transaction)
    
    def get_history(self) -> List[Dict]:
        return self.transaction_history.copy()


class ToolExecutor:
    """Executes tools and returns observations"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def execute(self, action: Action) -> Observation:
        """Execute an action and return observation"""
        tool_name = action.tool
        args = action.arguments
        
        if tool_name == "set_order":
            return self._set_order(**args)
        elif tool_name == "get_portfolio":
            return self._get_portfolio()
        elif tool_name == "get_market_news":
            return self._get_market_news(**args)
        else:
            return Observation(
                content=f"Error: Unknown tool '{tool_name}'",
                success=False
            )
    
    def _set_order(self, stock: str, quantity: int, action: str) -> Observation:
        """Execute stock order"""
        print(f"\n{'='*60}")
        print(f"[TOOL EXECUTION] set_order")
        print(f"{'='*60}")
        print(f"  Stock: {stock}")
        print(f"  Quantity: {quantity}")
        print(f"  Action: {action}")
        
        if action == "SELL":
            current = self.portfolio.get_holding(stock)
            if current >= quantity:
                self.portfolio.set_holding(stock, current - quantity)
                msg = f"Successfully sold {quantity} shares of {stock}. Remaining: {self.portfolio.get_holding(stock)}"
                success = True
                print(f"  ✓ {msg}")
            else:
                msg = f"Failed: Insufficient shares. Have {current}, need {quantity}"
                success = False
                print(f"  ✗ {msg}")
        
        elif action == "BUY":
            current = self.portfolio.get_holding(stock)
            self.portfolio.set_holding(stock, current + quantity)
            msg = f"Successfully bought {quantity} shares of {stock}. Total: {self.portfolio.get_holding(stock)}"
            success = True
            print(f"  ✓ {msg}")
        
        else:
            msg = f"Invalid action: {action}"
            success = False
            print(f"  ✗ {msg}")
        
        # Record transaction
        self.portfolio.add_transaction({
            "tool": "set_order",
            "arguments": {"stock": stock, "quantity": quantity, "action": action},
            "result": msg,
            "success": success
        })
        
        print(f"{'='*60}\n")
        
        return Observation(content=msg, success=success)
    
    def _get_portfolio(self) -> Observation:
        """Get current portfolio"""
        print(f"\n[TOOL EXECUTION] get_portfolio")
        holdings = self.portfolio.get_state()
        msg = f"Current portfolio: {holdings}"
        print(f"  {msg}")
        return Observation(content=msg, success=True)
    
    def _get_market_news(self, stocks: List[str]) -> Observation:
        """Get market news"""
        print(f"\n[TOOL EXECUTION] get_market_news({stocks})")
        
        # Mock news
        news_db = {
            "STOCK_A": ["STOCK_A drops 5% on weak earnings", "Analysts downgrade STOCK_A"],
            "STOCK_B": ["STOCK_B rallies on strong guidance", "STOCK_B enters new market"],
        }
        
        news_summary = []
        for stock in stocks:
            stock_news = news_db.get(stock, [f"No news for {stock}"])
            news_summary.append(f"{stock}: {'; '.join(stock_news)}")
        
        msg = " | ".join(news_summary)
        print(f"  {msg}")
        
        return Observation(content=msg, success=True)


class SemanticCacheLLM:
    """LLM client with semantic cache"""
    
    def __init__(self, cache_server_url: str = "http://127.0.0.1:5000"):
        self.cache_server_url = cache_server_url.rstrip('/')
    
    def generate(self, prompt: str) -> Tuple[str, List[Dict], Dict]:
        """
        Generate response from LLM (via cache)
        
        Returns:
            (response_text, tool_calls, metadata)
        """
        try:
            response = requests.post(
                f"{self.cache_server_url}/chat",
                json={"query": prompt, "use_tools": True},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                metadata = {
                    'cache_status': data.get('debug_status'),
                    'similarity': data.get('similarity', 0.0),
                    'latency': data.get('server_latency', 0.0)
                }
                
                return (
                    data.get('response', ''),
                    data.get('tool_calls', []),
                    metadata
                )
            else:
                return (
                    f"LLM Error: HTTP {response.status_code}",
                    [],
                    {'cache_status': 'ERROR', 'similarity': 0.0}
                )
        
        except Exception as e:
            return (
                f"LLM Error: {str(e)}",
                [],
                {'cache_status': 'ERROR', 'similarity': 0.0}
            )


class ReActAgent:
    """
    ReAct Agent: Reasoning + Acting in interleaved manner
    
    ReAct Loop:
    1. Thought: Reason about what to do next
    2. Action: Execute a tool
    3. Observation: Observe the result
    4. Repeat until task is complete
    """
    
    def __init__(
        self,
        cache_server_url: str = "http://127.0.0.1:5000",
        initial_portfolio: Dict[str, int] = None,
        max_iterations: int = 5
    ):
        """
        Initialize ReAct Agent
        
        Args:
            cache_server_url: Semantic cache server URL
            initial_portfolio: Initial stock holdings
            max_iterations: Maximum ReAct iterations
        """
        self.llm = SemanticCacheLLM(cache_server_url)
        self.portfolio = Portfolio(initial_portfolio)
        self.tools = ToolExecutor(self.portfolio)
        self.max_iterations = max_iterations
        
        # Execution trace
        self.react_trace = []
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Run ReAct loop for a given task
        
        Args:
            task: User's task/query
            
        Returns:
            Dict with final answer and execution trace
        """
        print(f"\n{'='*80}")
        print(f"[ReAct Agent] Starting Task")
        print(f"{'='*80}")
        print(f"Task: {task}")
        print(f"Initial Portfolio: {self.portfolio.get_state()}")
        print(f"{'='*80}\n")
        
        # Initialize trace
        trace = {
            'task': task,
            'iterations': [],
            'final_answer': None,
            'cache_metadata': []
        }
        
        # ReAct loop
        for iteration in range(self.max_iterations):
            print(f"\n{'─'*80}")
            print(f"[Iteration {iteration + 1}/{self.max_iterations}]")
            print(f"{'─'*80}")
            
            # Build prompt with ReAct format
            prompt = self._build_react_prompt(task, trace['iterations'])
            
            # Get LLM response
            print(f"\n[State] THINKING...")
            response, tool_calls, metadata = self.llm.generate(prompt)
            
            trace['cache_metadata'].append(metadata)
            
            # Log cache status
            cache_status = metadata.get('cache_status', 'UNKNOWN')
            similarity = metadata.get('similarity', 0.0)
            
            if cache_status == 'HIT':
                print(f"[LLM Cache] HIT (similarity: {similarity:.4f})")
            else:
                print(f"[LLM Cache] MISS (similarity: {similarity:.4f})")
            
            # Parse response for thought
            thought = Thought(content=response)
            print(f"\n{thought}")
            
            iteration_data = {
                'iteration': iteration + 1,
                'thought': thought.content,
                'actions': [],
                'observations': [],
                'cache_status': cache_status,
                'similarity': similarity
            }
            
            # Check if task is complete (no tool calls)
            if not tool_calls:
                print(f"\n[State] FINISHED")
                print(f"Final Answer: {response}")
                trace['final_answer'] = response
                trace['iterations'].append(iteration_data)
                break
            
            # Execute actions
            print(f"\n[State] ACTING...")
            for i, tool_call in enumerate(tool_calls, 1):
                action = Action(
                    tool=tool_call.get('name'),
                    arguments=tool_call.get('arguments', {})
                )
                
                print(f"\n{action}")
                
                # Execute action
                print(f"[State] OBSERVING...")
                observation = self.tools.execute(action)
                
                print(f"{observation}")
                
                iteration_data['actions'].append({
                    'tool': action.tool,
                    'arguments': action.arguments
                })
                iteration_data['observations'].append({
                    'content': observation.content,
                    'success': observation.success
                })
            
            trace['iterations'].append(iteration_data)
        
        else:
            # Max iterations reached
            print(f"\n[Warning] Max iterations ({self.max_iterations}) reached")
            trace['final_answer'] = "Task incomplete: Maximum iterations reached"
        
        # Store trace
        self.react_trace.append(trace)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"[ReAct Agent] Task Complete")
        print(f"{'='*80}")
        print(f"Final Portfolio: {self.portfolio.get_state()}")
        print(f"Total Iterations: {len(trace['iterations'])}")
        print(f"Transactions: {len(self.portfolio.get_history())}")
        print(f"{'='*80}\n")
        
        return {
            'task': task,
            'final_answer': trace['final_answer'],
            'portfolio': self.portfolio.get_state(),
            'iterations': len(trace['iterations']),
            'transactions': self.portfolio.get_history(),
            'trace': trace
        }
    
    def _build_react_prompt(self, task: str, previous_iterations: List[Dict]) -> str:
        """
        Build ReAct-style prompt
        
        Format:
        Task: <user task>
        
        You have access to these tools:
        - set_order(stock, quantity, action): Execute stock trade
        - get_portfolio(): Get current holdings
        - get_market_news(stocks): Get news for stocks
        
        Use this format:
        Thought: <your reasoning>
        Action: <tool_name>(arguments)
        Observation: <result>
        ... (repeat Thought/Action/Observation as needed)
        Thought: <final reasoning>
        Final Answer: <answer to user>
        
        Previous iterations:
        <history>
        
        Continue:
        """
        
        prompt_parts = [
            f"Task: {task}",
            "",
            "You are a financial advisor. You have access to these tools:",
            "- set_order(stock, quantity, action): Execute stock trade (action: BUY or SELL)",
            "- get_portfolio(): Get current portfolio holdings",
            "- get_market_news(stocks): Get market news for given stocks",
            "",
            "Current Portfolio: " + json.dumps(self.portfolio.get_state()),
            "",
            "Use the ReAct format:",
            "Thought: <your reasoning about what to do>",
            "Action: <call a tool if needed>",
            "Observation: <result from tool>",
            "... (repeat Thought/Action/Observation as needed)",
            "Final Answer: <your final recommendation to the user>",
            "",
        ]
        
        # Add previous iterations
        if previous_iterations:
            prompt_parts.append("Previous steps:")
            for iter_data in previous_iterations:
                prompt_parts.append(f"Thought: {iter_data['thought']}")
                
                for j, action in enumerate(iter_data['actions']):
                    prompt_parts.append(f"Action: {action['tool']}({action['arguments']})")
                    obs = iter_data['observations'][j]
                    prompt_parts.append(f"Observation: {obs['content']}")
            
            prompt_parts.append("")
            prompt_parts.append("Continue with next Thought:")
        else:
            prompt_parts.append("Begin:")
        
        return "\n".join(prompt_parts)
    
    def get_portfolio(self) -> Dict[str, int]:
        """Get current portfolio"""
        return self.portfolio.get_state()
    
    def get_transaction_history(self) -> List[Dict]:
        """Get transaction history"""
        return self.portfolio.get_history()
    
    def get_react_trace(self) -> List[Dict]:
        """Get complete ReAct execution trace"""
        return self.react_trace


# ==================== Utility Functions ====================

def flush_cache(server_url: str = "http://127.0.0.1:5000"):
    """Flush semantic cache"""
    try:
        response = requests.post(f"{server_url}/flush_cache", timeout=10)
        if response.status_code == 200:
            print("[Setup] ✓ Cache flushed")
        else:
            print(f"[Setup] ✗ Cache flush failed: {response.status_code}")
    except Exception as e:
        print(f"[Setup] ✗ Cache flush error: {e}")


if __name__ == "__main__":
    # Simple test
    print("\n" + "="*80)
    print("ReAct Financial Agent - Unit Test")
    print("="*80)
    
    flush_cache()
    
    agent = ReActAgent(
        cache_server_url="http://127.0.0.1:5000",
        initial_portfolio={"STOCK_A": 10000, "STOCK_B": 5000},
        max_iterations=3
    )
    
    task = "Check the market news and advise me on my investments."
    
    result = agent.run(task)
    
    print("\n" + "="*80)
    print("RESULT SUMMARY")
    print("="*80)
    print(f"Task: {result['task']}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Final Portfolio: {result['portfolio']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Transactions: {len(result['transactions'])}")
    print("="*80)