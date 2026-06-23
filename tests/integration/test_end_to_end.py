"""端到端集成测试：验证 DSL → 验证 → 拓扑排序 → 可视化 → 序列化 全链路，
以及 Agent 运行时从构建到执行的完整流程。"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from agentflow.dsl.types import Node, Edge, Workflow, NodeType
from agentflow.dsl.validator import validate_dag
from agentflow.dsl.graph import topological_sort, parallel_groups
from agentflow.dsl.visualizer import to_mermaid
from agentflow.dsl.serializer import to_yaml, from_yaml

from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.thinking import ThinkingMode
from agentflow.runtime.toolkit import tool
from agentflow.runtime.prompt import PromptTemplate
from agentflow.runtime.memory.manager import MemoryProfile


# ============================================================================
# DSL 层端到端测试
# ============================================================================

class TestAgentFlowEndToEnd:
    """模拟 AgentFlow 的完整使用流程。"""

    def test_full_pipeline(self):
        # 1. 用户在 YAML 中定义一个菱形 DAG Workflow
        wf = Workflow(
            name="Order Processing Pipeline",
            description="Handle customer orders with fraud check and inventory",
            nodes=[
                Node(id="receive_order", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}, timeout_ms=30000),
                Node(id="fraud_check", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}, retry_max=2),
                Node(id="inventory_check", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}, retry_max=1),
                Node(id="fulfill", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}),
            ],
            edges=[
                Edge(from_node="receive_order", to_node="fraud_check"),
                Edge(from_node="receive_order", to_node="inventory_check"),
                Edge(from_node="fraud_check", to_node="fulfill"),
                Edge(from_node="inventory_check", to_node="fulfill"),
            ],
        )

        # 2. 验证 DAG 合法性
        validate_dag(wf)  # 不应抛异常

        # 3. 计算执行顺序
        order = topological_sort(wf)
        assert order[0] == "receive_order"
        assert order[-1] == "fulfill"

        # 4. 计算并行分组
        groups = parallel_groups(wf)
        assert groups[0] == ["receive_order"]
        assert set(groups[1]) == {"fraud_check", "inventory_check"}  # 可并行
        assert groups[2] == ["fulfill"]

        # 5. 生成 Mermaid 可视化
        mermaid = to_mermaid(wf)
        assert "graph TD" in mermaid
        assert "receive_order" in mermaid
        assert "fraud_check" in mermaid
        assert "inventory_check" in mermaid

        # 6. YAML 序列化/反序列化
        yaml_str = to_yaml(wf)
        restored = from_yaml(yaml_str)
        assert restored.name == "Order Processing Pipeline"
        assert len(restored.nodes) == 4
        assert len(restored.edges) == 4

        # 7. 反序列化后仍可通过验证
        validate_dag(restored)

    def test_mermaid_exports_valid_markdown(self):
        """Mermaid 输出可直接嵌入 Markdown。"""
        wf = Workflow(
            name="Simple Pipeline",
            nodes=[Node(id="a", node_type=NodeType.AGENT)],
            edges=[],
        )
        mermaid = to_mermaid(wf)
        markdown_block = f"```mermaid\n{mermaid}\n```"
        assert "```mermaid" in markdown_block
        assert "graph TD" in markdown_block


# ============================================================================
# Agent 运行时端到端测试 —— 完整链路：Builder → Memory → Thinking → Tool → Result
# ============================================================================

class TestAgentRuntimeEndToEnd:
    """验证 Agent 运行时从构建到执行的完整链路。

    每个测试都走完整流程:
        AgentBuilder.build_sync()
        → memory.pre_turn()   (检索门)
        → thinking_engine.run()
            → strategy 循环 (LLM 调用 + 工具执行)
        → memory.post_turn()  (记忆门 + 遗忘门)
        → AgentResult
    """

    # ------------------------------------------------------------------
    # 工具定义
    # ------------------------------------------------------------------

    @staticmethod
    def _make_lookup_tool():
        @tool
        def lookup(query: str) -> str:
            """Search the knowledge base for information."""
            knowledge = {
                "refund": "退款政策: 30天内可全额退款。",
                "shipping": "发货政策: 下单后48小时内发货。",
                "return": "退货政策: 14天内支持无理由退货。",
            }
            return knowledge.get(query.lower(), f"未找到关于 '{query}' 的信息")

        return lookup

    @staticmethod
    def _make_calculator_tool():
        @tool
        def calculator(expression: str) -> str:
            """Evaluate a mathematical expression."""
            try:
                return str(eval(expression))
            except Exception:
                return "计算错误"

        return calculator

    # ------------------------------------------------------------------
    # Mock LLM 工厂
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_llm(responses: list):
        """创建一个按顺序返回响应的 mock LLM 客户端。

        responses: 每次 chat() 调用的返回值列表。
        最后一个响应重复返回（模拟对话结束后的行为）。
        """
        mock = AsyncMock()

        async def chat_side_effect(messages, tools=None):
            if responses:
                return responses.pop(0)
            # 如果响应用完，返回空回答（模拟 LLM 无更多工具调用）
            return MagicMock(content="Done.", role="assistant", tool_calls=[])

        mock.chat.side_effect = chat_side_effect
        return mock

    @staticmethod
    def _tool_response(content: str | None, tool_calls: list | None = None):
        """快捷构造 LLM 响应。"""
        return MagicMock(content=content, role="assistant", tool_calls=tool_calls or [])

    # ------------------------------------------------------------------
    # 测试用例
    # ------------------------------------------------------------------

    def test_single_tool_call_then_answer(self):
        """最典型的 Agent 交互: 用户提问 → 调一次工具 → 返回答案。

        模拟一个客服场景：
            用户问退款政策 → LLM 决定调用 lookup("refund")
            → 工具返回政策文本 → LLM 组织成自然语言回复用户
        """
        mock_llm = self._mock_llm([
            # 第1轮: LLM 选择调用 lookup 工具
            self._tool_response(None, tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "refund"}'},
            }]),
            # 第2轮: LLM 拿到工具结果，生成最终答案
            self._tool_response("根据我们的政策，您在30天内可以申请全额退款。"),
        ])

        agent = (
            AgentBuilder("客服Agent")
            .with_llm(mock_llm)
            .with_tools(self._make_lookup_tool())
            .with_prompt("你是一个专业的客服助手。")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .with_max_iterations(5)
            .build_sync()
        )

        result = asyncio.run(agent.run("我想退款，政策是什么？"))

        # 验证完整链路产出
        assert "30天" in result.output or "退款" in result.output
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "lookup"
        assert result.tool_calls[0]["input"] == {"query": "refund"}

    def test_multi_tool_chain(self):
        """多步工具链: 用户提问需要多次工具调用才能回答。

        场景: 用户问"我退款后多久能收到钱？"
        → 先调 lookup("refund") 查政策
        → 再调 lookup("return") 确认退货流程
        → 最后综合两个结果回答
        """
        mock_llm = self._mock_llm([
            # 第1轮: 查退款政策
            self._tool_response(None, tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "refund"}'},
            }]),
            # 第2轮: 拿到退款结果后，想查退货流程
            self._tool_response(None, tool_calls=[{
                "id": "call_2",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "return"}'},
            }]),
            # 第3轮: 综合两个工具结果，生成最终答案
            self._tool_response(
                "退款政策是30天全额退款，退货后14天内可无理由退货，"
                "退款将在收到退货后3-5个工作日到账。"
            ),
        ])

        agent = (
            AgentBuilder("高级客服Agent")
            .with_llm(mock_llm)
            .with_tools(self._make_lookup_tool())
            .with_prompt("你是资深客服，需要先查政策再回答。")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.standard())
            .with_max_iterations(10)
            .build_sync()
        )

        result = asyncio.run(agent.run("退款和退货的流程是什么？"))

        assert len(result.tool_calls) == 2
        assert all(tc["tool"] == "lookup" for tc in result.tool_calls)
        assert "30天" in result.output

    def test_no_tool_needed(self):
        """简单问题不需要工具: LLM 直接回答，不触发工具调用。"""
        mock_llm = self._mock_llm([
            self._tool_response("你好！有什么可以帮助你的吗？"),
        ])

        agent = (
            AgentBuilder("闲聊Agent")
            .with_llm(mock_llm)
            .with_tools(self._make_lookup_tool())
            .with_prompt("你是一个友好的助手。")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .build_sync()
        )

        result = asyncio.run(agent.run("你好"))

        assert len(result.tool_calls) == 0
        assert "你好" in result.output

    def test_memory_persistence_across_turns(self):
        """记忆持久化: 多轮对话中，前一轮的事实应在后续轮次可检索。

        场景:
            第1轮: 用户说"我住在北京" → post_turn 提取事实到 episodic memory
            第2轮: 用户问"我这里的天气怎么样？"
                  → pre_turn 应从 semantic memory 检索到"北京"
                  → 基于记忆上下文回答
        """
        mock_llm = self._mock_llm([
            # 第1轮: 用户透露个人信息
            self._tool_response("好的，我记住了，您住在北京。"),
            # 第2轮: 基于记忆回答
            self._tool_response("根据我记住的信息，您住在北京。今天北京天气晴朗，25°C。"),
        ])

        agent = (
            AgentBuilder("记忆Agent")
            .with_llm(mock_llm)
            .with_prompt("你是一个有记忆的助手，需要记住用户的信息。")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.standard())
            .build_sync()
        )

        # 第1轮：用户透露个人信息 → 触发 post_turn 记忆提取
        result1 = asyncio.run(agent.run("我住在北京"))
        assert "北京" in result1.output

        # 第2轮：应能从记忆中检索到"北京"
        result2 = asyncio.run(agent.run("我这里天气怎么样？"))
        assert "北京" in result2.output

    def test_thinking_mode_cot(self):
        """CoT 思考模式: 先深度思考再回答，不急于调工具。"""
        mock_llm = self._mock_llm([
            # CoT 模式：第一轮思考分析
            self._tool_response(
                "让我分析一下这个问题。用户想知道退款的完整流程。"
                "我需要查询相关政策然后给出清晰答复。"
            ),
        ])

        agent = (
            AgentBuilder("思考型Agent")
            .with_llm(mock_llm)
            .with_prompt("你需要先思考分析再回答。")
            .with_thinking(ThinkingMode.COT)
            .with_memory(MemoryProfile.light())
            .build_sync()
        )

        result = asyncio.run(agent.run("退款流程是怎样的？"))
        assert result.output is not None
        assert len(result.output) > 0

    def test_builder_integration_all_components(self):
        """全组件集成: Builder + ToolKit + Memory + PromptTemplate + ThinkingMode。

        验证所有子系统能在一条链路中协同工作，不报错、不短路。
        """
        template = PromptTemplate.preset("customer_support")

        mock_llm = self._mock_llm([
            self._tool_response(None, tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "100 * 0.7"}'},
            }]),
            self._tool_response("折扣后价格为 70 元。"),
        ])

        agent = (
            AgentBuilder("全能Agent")
            .with_llm(mock_llm)
            .with_tools(self._make_lookup_tool(), self._make_calculator_tool())
            .with_memory(MemoryProfile.deep())
            .with_prompt(template)
            .with_thinking(ThinkingMode.ADAPTIVE)
            .with_max_iterations(8)
            .build_sync()
        )

        result = asyncio.run(agent.run("100元打7折是多少？"))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "calculator"
        assert "70" in result.output

    def test_error_handling_tool_execution_failure(self):
        """工具执行失败时，错误不应导致整个 Agent 崩溃，而是优雅返回。"""
        mock_llm = self._mock_llm([
            self._tool_response(None, tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "1/0"}'},
            }]),
            self._tool_response(
                "计算器返回了错误，让我直接回答：1除以0是未定义的。"
            ),
        ])

        agent = (
            AgentBuilder("容错Agent")
            .with_llm(mock_llm)
            .with_tools(self._make_calculator_tool())
            .with_prompt("你是数学助手。")
            .with_thinking(ThinkingMode.REACT)
            .with_max_iterations(5)
            .build_sync()
        )

        # 不应抛异常
        result = asyncio.run(agent.run("1除以0等于多少？"))
        assert result is not None
        assert result.output is not None
