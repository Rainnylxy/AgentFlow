"""DSL 核心类型：Node, Edge, Workflow

每个 Node 就是一段可执行的工作单元，分为四种类型：
  Agent    — LLM Agent（思考 + 调工具）
  Tool     — 纯函数调用（零 LLM 开销）
  Human    — 人工确认（Human-in-the-Loop）
  Subgraph — 嵌套子 Workflow

控制流在边上：Edge.condition 决定是否走这条边。
并行由 DAG 结构自动推导（parallel_groups），无需手动声明。
循环配置在 Node.loop 里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Node executor type
# ---------------------------------------------------------------------------

class NodeKind(str, Enum):
    """节点的执行体类型——一个 Node 同时只有一个非空配置。

    四种类型覆盖 Agent 编排的所有真实场景：
      AGENT    — LLM 驱动（模型 + prompt + 工具 + 思考策略）
      TOOL     — 纯函数调用（零 token 开销，快速可靠）
      HUMAN    — 人工确认（Human-in-the-Loop）
      SUBGRAPH — 嵌套另一个 Workflow
    """
    AGENT = "agent"
    TOOL = "tool"
    HUMAN = "human"
    SUBGRAPH = "subgraph"


# ---------------------------------------------------------------------------
# 节点配置 dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Agent 节点的 LLM 和思考配置。"""
    model: str = "gpt-4o"
    prompt: str = ""                          # system prompt 文本或模板引用
    tools: list[str] = field(default_factory=list)   # 工具名列表
    thinking: str = "react"                   # react | cot | plan_execute | adaptive
    memory: str = "standard"                  # light | standard | deep
    memory_scope: str = "inherit"             # workflow | inherit | none


@dataclass
class ToolConfig:
    """Tool 节点——直接执行已注册的工具函数。"""
    name: str                                 # 工具名（ToolRegistry 中注册过的）
    inputs: dict = field(default_factory=dict)  # 参数映射


@dataclass
class HumanConfig:
    """Human 节点——暂停执行，等待人工输入。"""
    prompt: str = ""                          # 展示给审核者的说明
    timeout_sec: int = 3600                   # 等待超时（秒），超时后走 fallback
    default_response: str = ""                # 超时后的默认回复


@dataclass
class LoopConfig:
    """节点级循环配置。"""
    max_iterations: int = 10
    condition: str = ""                       # 停止条件表达式，如 'output.score > 0.8'
    break_on_tool: str = ""                   # 调用指定工具后退出循环


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class FallbackPolicy(str, Enum):
    """节点失败时的降级策略。"""
    SKIP = "skip"
    DEFAULT_VALUE = "default_value"
    FALLBACK_NODE = "fallback_node"
    RAISE = "raise"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Workflow 中的一个可执行节点。

    每个 Node 就是一个工作单元——Agent 思考、Tool 执行、
    Human 确认、或 Subgraph 子流程。

    四种执行体（agent / tool / human / subgraph）同时最多一个非空。
    """

    id: str
    kind: NodeKind = NodeKind.AGENT

    # 四种执行体——按 kind 选一个
    agent: Optional[AgentConfig] = None
    tool: Optional[ToolConfig] = None
    human: Optional[HumanConfig] = None
    subgraph: Optional[str] = None          # 子 Workflow 的 name（引用）

    # 通用配置
    label: str = ""                         # 显示名，默认用 id
    timeout_ms: int = 30_000
    retry_max: int = 0
    loop: Optional[LoopConfig] = None       # 非空则循环执行该节点
    fallback: FallbackPolicy = FallbackPolicy.RAISE
    default_value: Optional[str] = None
    fallback_node_id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            raise ValueError("id must be non-empty")
        # 自动填充默认 agent config
        if self.kind == NodeKind.AGENT and self.agent is None:
            self.agent = AgentConfig()


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """DAG 中的有向边——控制流在边上。

    condition 为空 → 无条件流转（前驱完成即触发）。
    condition 非空 → Jinja2 条件模板，如 '{{ output.score > 0.5 }}'。
    """

    from_node: str
    to_node: str
    condition: Optional[str] = None

    def __post_init__(self):
        if self.from_node == self.to_node:
            raise ValueError("from_node and to_node must differ")


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@dataclass
class Workflow:
    """多 Agent 工作流的完整定义。"""

    name: str
    nodes: list[Node]
    edges: list[Edge]
    description: str = ""
    global_timeout_ms: int = 300_000

    def __post_init__(self):
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            from collections import Counter
            dupes = [nid for nid, cnt in Counter(n.id for n in self.nodes).items() if cnt > 1]
            raise ValueError(f"Duplicate node id(s): {set(dupes)}")
        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(f"Edge from unknown node '{edge.from_node}'")
            if edge.to_node not in node_ids:
                raise ValueError(f"Edge to unknown node '{edge.to_node}'")
