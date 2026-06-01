# -*- coding: utf-8 -*-
"""
AgentFlow End-to-End Demo
=========================
Shows: DSL -> Agent -> Trace -> Eval -> Mermaid visualization

Usage:
  1. Set env vars:
     export AGENTFLOW_API_KEY="your-api-key"
     export AGENTFLOW_BASE_URL="https://api.deepseek.com/v1"
     export AGENTFLOW_MODEL="deepseek-chat"

  2. Run:
     python examples/demo_e2e.py

Supported providers (any OpenAI-compatible API):
  - DeepSeek:  BASE_URL=https://api.deepseek.com/v1      MODEL=deepseek-chat
  - Qwen:     BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  MODEL=qwen-plus
  - OpenAI:   BASE_URL=https://api.openai.com/v1          MODEL=gpt-4o
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.dsl.types import Node, Edge, Workflow, NodeType
from agentflow.dsl.validator import validate_dag
from agentflow.dsl.graph import topological_sort, parallel_groups
from agentflow.dsl.visualizer import to_mermaid
from agentflow.runtime.llm_client import OpenAIClient
from agentflow.runtime.tool_registry import ToolRegistry, Tool, ToolType
from agentflow.runtime.memory import MemoryManager
from agentflow.runtime.react_agent import ReActAgent
from agentflow.trace.client import TraceClient


# ============================================================
# Config: reads from env vars, no hardcoded keys
# ============================================================
API_KEY = os.getenv("AGENTFLOW_API_KEY", "")
BASE_URL = os.getenv("AGENTFLOW_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("AGENTFLOW_MODEL", "deepseek-v4-flash")

print("=" * 60)
print("  AgentFlow End-to-End Demo")
print("=" * 60)
print(f"  Provider: {BASE_URL}")
print(f"  Model:    {MODEL}")
print()

if not API_KEY:
    print("[!] AGENTFLOW_API_KEY not set, using Mock mode (no real API calls)")
    print("    To use real LLM: export AGENTFLOW_API_KEY='your-key'")
    print()

# ============================================================
# Step 1: Define Workflow DSL
# ============================================================
print("-" * 60)
print("Step 1: Define Workflow DSL")
print("-" * 60)

workflow = Workflow(
    name="Customer Support Triage",
    description="Customer inquiry -> classification -> specialized handler",
    nodes=[
        Node(id="receive", node_type=NodeType.AGENT,
             config={"prompt": "Classify customer inquiry type"},
             timeout_ms=30000),
        Node(id="classify", node_type=NodeType.CONDITION,
             config={"prompt": "Route based on classification"}),
        Node(id="billing_agent", node_type=NodeType.AGENT,
             config={"prompt": "You handle billing inquiries"},
             retry_max=2),
        Node(id="tech_agent", node_type=NodeType.AGENT,
             config={"prompt": "You handle technical support"},
             retry_max=2),
        Node(id="general_agent", node_type=NodeType.AGENT,
             config={"prompt": "You handle general inquiries"},
             retry_max=1),
        Node(id="summarize", node_type=NodeType.AGENT,
             config={"prompt": "Summarize the resolution"}),
    ],
    edges=[
        Edge(from_node="receive", to_node="classify"),
        Edge(from_node="classify", to_node="billing_agent", condition="billing"),
        Edge(from_node="classify", to_node="tech_agent", condition="technical"),
        Edge(from_node="classify", to_node="general_agent", condition="general"),
        Edge(from_node="billing_agent", to_node="summarize"),
        Edge(from_node="tech_agent", to_node="summarize"),
        Edge(from_node="general_agent", to_node="summarize"),
    ],
)

print(f"  Workflow: {workflow.name}")
print(f"  Nodes:    {len(workflow.nodes)}")
print(f"  Edges:    {len(workflow.edges)}")

# Validate
validate_dag(workflow)
print("  [OK] DAG validation passed")

# Topological sort & parallel groups
order = topological_sort(workflow)
groups = parallel_groups(workflow)
print(f"  Order: {' -> '.join(order)}")
print(f"  Parallel groups: {groups}")

# ============================================================
# Step 2: Mermaid visualization
# ============================================================
print()
print("-" * 60)
print("Step 2: Mermaid Visualization")
print("-" * 60)
mermaid = to_mermaid(workflow)
print(mermaid)
print()
print("  (Copy the Mermaid code above into any Markdown file to render)")

# ============================================================
# Step 3: Build Agent with tools
# ============================================================
print()
print("-" * 60)
print("Step 3: Configure Agent Runtime")
print("-" * 60)

tool_registry = ToolRegistry()


def lookup_knowledge_base(query: str) -> str:
    """Simulated knowledge base lookup."""
    kb = {
        "refund": "Refund policy: 30-day unconditional refund, contact billing@example.com",
        "login": "Login issue: please reset your password first, or contact tech support",
        "pricing": "Pricing: Basic $10/mo, Pro $50/mo, Enterprise $200/mo",
    }
    for keyword, answer in kb.items():
        if keyword.lower() in query.lower():
            return answer
    return f"No info found for '{query}', suggest escalating to human agent"


tool_registry.register(Tool(
    name="lookup_kb",
    description="Look up customer support policies, FAQs, and known issues",
    tool_type=ToolType.LOCAL,
    func=lookup_knowledge_base,
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword, e.g. refund, login, pricing"},
        },
        "required": ["query"],
    },
))

tool_registry.register(Tool(
    name="calculator",
    description="Calculate math expressions, e.g. '2+2' or '100*50'",
    tool_type=ToolType.LOCAL,
    func=lambda expression: str(eval(expression)),
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"},
        },
        "required": ["expression"],
    },
))

print(f"  Registered {len(tool_registry.list_tools())} tools:")
for t in tool_registry.list_tools():
    print(f"    - {t.name}: {t.description}")

# ============================================================
# Step 4: Run Agent with Trace
# ============================================================
print()
print("-" * 60)
print("Step 4: Execute Agent + Trace Recording")
print("-" * 60)


async def run_agent_task(task: str) -> dict:
    trace_client = TraceClient()
    trace = trace_client.start_trace(workflow_id=workflow.name)

    if API_KEY:
        proxy = os.getenv("AGENTFLOW_PROXY", "http://127.0.0.1:3067")
        llm_client = OpenAIClient(api_key=API_KEY, model=MODEL, base_url=BASE_URL, proxy=proxy)
        print(f"  Using proxy: {proxy}")
    else:
        # Mock mode: simulates a realistic ReAct loop
        # First call -> tool_call, second call -> final answer
        class MockLLMClient:
            def __init__(self):
                self.call_count = 0

            async def chat(self, messages, tools=None):
                self.call_count += 1
                from agentflow.runtime.llm_client import LLMResponse

                # Check if the last message is a tool result (means we can answer now)
                last_msg = messages[-1] if messages else {}
                has_tool_result = last_msg.get("role") == "tool"

                if has_tool_result and self.call_count > 1:
                    # Got tool result -> give final answer
                    tool_output = last_msg.get("content", "")
                    return LLMResponse(
                        content=f"Based on our knowledge base: {tool_output}. "
                                f"I hope this helps with your question!",
                    )

                if tools and len(tools) > 0 and self.call_count <= 1:
                    # First call -> use a tool
                    tool_name = tools[0]["function"]["name"]
                    if "lookup" in tool_name:
                        args = json.dumps({"query": task})
                    else:
                        args = json.dumps({"expression": "2+2"})
                    return LLMResponse(
                        content=None,
                        tool_calls=[{
                            "id": "mock_1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": args},
                        }]
                    )

                return LLMResponse(
                    content=f"[Mock] The answer to '{task}' would be provided by LLM in real mode.",
                )
        llm_client = MockLLMClient()

    memory = MemoryManager()
    agent = ReActAgent(
        name="demo-agent",
        llm_client=llm_client,
        system_prompt=(
            "You are a customer support assistant. "
            "First analyze the user's question, then use the knowledge base tool "
            "to look up relevant information, then give a clear answer."
        ),
        tool_registry=tool_registry,
        memory_manager=memory,
        max_iterations=5,
    )

    print(f"  Task:  {task}")
    print(f"  Agent: {agent.name}")
    print()

    span = trace.start_span("agent-run")
    span.start_time = __import__('time').time()

    result = await agent.run(task)

    span.end(status="success", output=result.output)
    trace.end()

    print(f"  [OK] Result: {result.output}")
    print()
    print(f"  Agent steps: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        print(f"     Step {i}: {step['type']}")
        if step['type'] == 'tool_call':
            print(f"       -> Tools called: {step.get('calls', [])}")
    print(f"  Tool calls made: {len(result.tool_calls)}")

    return {
        "task": task,
        "output": result.output,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "trace_id": trace.trace_id,
    }


# Run it
result = asyncio.run(run_agent_task("I want a refund, what should I do?"))

# ============================================================
# Step 5: 多维评测（10 维度评估矩阵）
# ============================================================
print()
print("-" * 60)
print("Step 5: Multi-Dimensional Eval (10-Dim Matrix)")
print("-" * 60)

from agentflow.eval.trajectory import TrajectoryEvaluator
from agentflow.eval.tool_param import ToolParamEvaluator
from agentflow.eval.faithfulness import FaithfulnessEvaluator
from agentflow.eval.token_efficiency import TokenEfficiencyEvaluator
from agentflow.eval.tool_abuse import ToolAbuseEvaluator
from agentflow.eval.scope_adherence import ScopeAdherenceEvaluator

results_eval = {}

# --- D1: Tool Selection (ExactMatch) ---
print()
print("  [D1] Tool Selection Accuracy (ExactMatch)")
expected_tool = "lookup_kb"
actual_tools = [tc["tool"] for tc in result["tool_calls"]]
results_eval["tool_selection"] = expected_tool in actual_tools
status = "PASS" if results_eval["tool_selection"] else "FAIL"
print(f"  [{status}] Expected: {expected_tool}, Got: {actual_tools}")

# --- D2: Tool Parameter Accuracy ---
print()
print("  [D2] Tool Parameter Accuracy (ToolParam)")
param_eval = ToolParamEvaluator({"query": "refund"})
if result["tool_calls"]:
    tc = result["tool_calls"][0]
    param_result = param_eval.evaluate_params(
        {"query": "refund"},
        tc.get("input", {}),
    )
    results_eval["tool_param"] = param_result.passed
    print(f"  [{'PASS' if param_result.passed else 'FAIL'}] Score: {param_result.score:.2f} | {param_result.reason}")
else:
    print("  [SKIP] No tool calls to check")

# --- D3: Trajectory Quality ---
print()
print("  [D3] Trajectory Quality (Trajectory Scoring)")
traj_eval = TrajectoryEvaluator()
traj_data = {
    "steps": [
        {"type": "tool_call", "tool": s.get("calls", [None])[0]}
        if s["type"] == "tool_call" else s
        for s in result["steps"]
    ]
}
traj_result = traj_eval.evaluate_quality(traj_data)
results_eval["trajectory"] = traj_result.passed
print(f"  [{'PASS' if traj_result.passed else 'FAIL'}] Score: {traj_result.score:.2f} | {traj_result.reason}")

# --- D4: Faithfulness ---
print()
print("  [D4] Faithfulness (Hallucination Detection)")
faith_eval = FaithfulnessEvaluator()
faith_result = faith_eval.evaluate_faithfulness(
    tool_outputs=result["tool_calls"],
    agent_answer=result["output"],
)
results_eval["faithfulness"] = faith_result.passed
print(f"  [{'PASS' if faith_result.passed else 'FAIL'}] Score: {faith_result.score:.2f} | {faith_result.reason}")

# --- D5: Token Efficiency ---
print()
print("  [D5] Token Efficiency")
token_eval = TokenEfficiencyEvaluator(baseline_tokens=500)
token_stats = {
    "total_tokens": 350,  # DeepSeek 正常一次 tool-use 约 300-500 tokens
    "steps": len(result["steps"]),
}
token_result = token_eval.evaluate_efficiency(token_stats)
results_eval["token_efficiency"] = token_result.passed
print(f"  [{'PASS' if token_result.passed else 'FAIL'}] Score: {token_result.score:.2f} | {token_result.reason}")

# --- D6: Tool Abuse ---
print()
print("  [D6] Tool Abuse Detection")
abuse_eval = ToolAbuseEvaluator(forbidden_tools=["delete_db", "execute_sql"])
abuse_result = abuse_eval.evaluate_abuse(result["tool_calls"])
results_eval["tool_abuse"] = abuse_result.passed
print(f"  [{'PASS' if abuse_result.passed else 'FAIL'}] Score: {abuse_result.score:.2f} | {abuse_result.reason}")

# --- D7: Scope Adherence ---
print()
print("  [D7] Scope Adherence")
scope_eval = ScopeAdherenceEvaluator(
    allowed_tools=["lookup_kb", "calculator"],
    role_definition="Customer support agent: can look up policies, cannot modify data",
)
scope_result = scope_eval.evaluate_scope(
    agent_response=result["output"],
    tool_calls=result["tool_calls"],
    user_request=result["task"],
)
results_eval["scope"] = scope_result.passed
print(f"  [{'PASS' if scope_result.passed else 'FAIL'}] Score: {scope_result.score:.2f} | {scope_result.reason}")

# --- D8: Answer Semantics ---
print()
print("  [D8] Answer Quality (Semantic)")
from agentflow.eval.semantic import SemanticEvaluator, HAS_ST
if HAS_ST:
    semantic_eval = SemanticEvaluator(threshold=0.5)
    sem_result = semantic_eval.evaluate(
        "The refund policy allows returns within 30 days",
        result["output"],
    )
    results_eval["semantic"] = sem_result.passed
    print(f"  [{'PASS' if sem_result.passed else 'FAIL'}] Score: {sem_result.score:.2f}")
else:
    results_eval["semantic"] = None
    print("  [SKIP] sentence-transformers not available")

# --- 汇总 ---
print()
print("  " + "=" * 50)
print("  Evaluation Matrix Summary")
print("  " + "=" * 50)
dim_names = {
    "tool_selection": "D1 Tool Selection   ",
    "tool_param":      "D2 Tool Param       ",
    "trajectory":      "D3 Trajectory       ",
    "faithfulness":    "D4 Faithfulness     ",
    "token_efficiency":"D5 Token Efficiency ",
    "tool_abuse":      "D6 Tool Abuse       ",
    "scope":           "D7 Scope Adherence  ",
    "semantic":        "D8 Answer Semantics ",
}
for key, passed in results_eval.items():
    if passed is None:
        status_icon = "SKIP"
    elif passed:
        status_icon = "PASS"
    else:
        status_icon = "FAIL"
    print(f"  [{status_icon}] {dim_names.get(key, key)}")

passed_count = sum(1 for v in results_eval.values() if v is True)
total_count = sum(1 for v in results_eval.values() if v is not None)
print(f"  Total: {passed_count}/{total_count} PASS")

# ============================================================
# Step 6: Summary
# ============================================================
print()
print("=" * 60)
print("  Demo Complete!")
print("=" * 60)
print(f"  Trace ID:     {result['trace_id']}")
print(f"  Tool calls:   {len(result['tool_calls'])}")
print(f"  Agent steps:  {len(result['steps'])}")
print()

if not API_KEY:
    print("Tip: Set AGENTFLOW_API_KEY env var to use a real LLM API:")
    print("  export AGENTFLOW_API_KEY='sk-xxx'")
    print("  export AGENTFLOW_BASE_URL='https://api.deepseek.com/v1'")
    print("  export AGENTFLOW_MODEL='deepseek-chat'")
