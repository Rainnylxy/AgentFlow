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
from agentflow.eval.exact_match import ExactMatchEvaluator


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
# Step 5: 多维评测
# ============================================================
print()
print("-" * 60)
print("Step 5: Multi-Dimensional Eval")
print("-" * 60)

# --- 维度 1: 工具调用准确性 (ExactMatch) ---
# 不是测试文本回答，而是测"Agent 有没有选对工具"
print()
print("  [Dimension 1] Tool-Use Accuracy (ExactMatch)")
tool_evaluator = ExactMatchEvaluator(case_sensitive=False)

# 检查 Agent 是否调了 lookup_kb 工具
expected_tool = "lookup_kb"
actual_tools = [tc["tool"] for tc in result["tool_calls"]]
tool_match = expected_tool in actual_tools

tool_score = 1.0 if tool_match else 0.0
status = "PASS" if tool_match else "FAIL"
print(f"  [{status}] Expected tool: {expected_tool}")
print(f"         Actual tools:  {actual_tools}")
print(f"         Score: {tool_score:.2f}")

# --- 维度 2: 推理轨迹质量 (Trajectory Scoring) ---
print()
print("  [Dimension 2] Trajectory Quality (Trajectory Scoring)")

from agentflow.eval.trajectory import TrajectoryEvaluator
traj_eval = TrajectoryEvaluator()

# 构建轨迹数据
trajectory_data = {
    "steps": [
        {"type": "tool_call", "tool": s.get("calls", [None])[0]}
        if s["type"] == "tool_call" else s
        for s in result["steps"]
    ]
}

traj_result = traj_eval.evaluate_quality(trajectory_data)
traj_status = "PASS" if traj_result.passed else "FAIL"
print(f"  [{traj_status}] Score: {traj_result.score:.2f}")
print(f"         Reason: {traj_result.reason}")
print(f"         Steps breakdown:")
for i, s in enumerate(result["steps"]):
    icon = {"tool_call": "|-- [TOOL]", "final": "|-- [ANSWER]"}.get(s["type"], "|-- [?]")
    detail = s.get("calls", []) if s["type"] == "tool_call" else s.get("output", "")[:60]
    print(f"           {icon} {detail}")

# --- 维度 3: 语义相似度 (Semantic) ---
print()
print("  [Dimension 3] Answer Quality (Semantic Similarity)")

from agentflow.eval.semantic import SemanticEvaluator, HAS_ST
if HAS_ST:
    semantic_eval = SemanticEvaluator(threshold=0.5)
    expected_answer = "The refund policy allows returns within 30 days"
    sem_result = semantic_eval.evaluate(expected_answer, result["output"])
    sem_status = "PASS" if sem_result.passed else "FAIL"
    print(f"  [{sem_status}] Score: {sem_result.score:.2f}")
    print(f"         Expected: {expected_answer}")
    print(f"         Got:      {result['output'][:80]}...")
    print(f"         Reason:   {sem_result.reason}")
else:
    print("  [SKIP] sentence-transformers not available on this Python version")

# --- 汇总 ---
print()
print("  Eval Summary:")
print(f"    Tool-Use:   {'PASS' if tool_match else 'FAIL'} (ExactMatch)")
print(f"    Trajectory: {traj_status} (Score: {traj_result.score:.2f})")
if HAS_ST:
    print(f"    Semantic:   {sem_status} (Score: {sem_result.score:.2f})")

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
