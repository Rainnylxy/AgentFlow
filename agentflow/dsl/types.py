"""DSL 核心类型：Node, Edge, Workflow"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    """节点类型决定了编排引擎如何执行这个节点。"""
    AGENT = "agent"           # LLM Agent 节点
    CONDITION = "condition"   # 条件路由节点
    PARALLEL = "parallel"     # 并行分发节点
    LOOP = "loop"             # 循环节点


class FallbackPolicy(str, Enum):
    """节点失败时的降级策略。"""
    SKIP = "skip"                   # 跳过该节点，继续执行
    DEFAULT_VALUE = "default_value"  # 返回预设默认值
    FALLBACK_NODE = "fallback_node"  # 转到降级节点
    RAISE = "raise"                  # 直接抛异常（默认）


@dataclass
class Node:
    """Workflow 中一个可执行节点。

    每个 Node 代表 Workflow DAG 中的一个顶点，由编排引擎调度执行。
    """
    id: str
    node_type: NodeType
    config: dict = field(default_factory=dict)
    timeout_ms: int = 30000          # 单节点超时（毫秒）
    retry_max: int = 0               # 最大重试次数（0 = 不重试）
    fallback: FallbackPolicy = FallbackPolicy.RAISE
    default_value: Optional[str] = None       # fallback=DEFAULT_VALUE 时使用
    fallback_node_id: Optional[str] = None    # fallback=FALLBACK_NODE 时使用

    def __post_init__(self):
        if not self.id:
            raise ValueError("id must be non-empty")


@dataclass
class Edge:
    """DAG 中的有向边，连接两个节点。

    condition 是可选的 Jinja2 模板条件；
    没有 condition 的边表示无条件顺次执行。
    """
    from_node: str
    to_node: str
    condition: Optional[str] = None  # 如 '{{ output.score > 0.5 }}'

    def __post_init__(self):
        if self.from_node == self.to_node:
            raise ValueError("from_node and to_node must differ")


@dataclass
class Workflow:
    """多 Agent 工作流的完整定义。

    包含一组 Node（顶点）和 Edge（边），构成一个 DAG。
    Workflow 在创建时自动校验节点唯一性和边的引用完整性。
    """
    name: str
    nodes: list[Node]
    edges: list[Edge]
    description: str = ""
    max_iterations: int = 10          # Loop 节点的全局最大迭代次数
    global_timeout_ms: int = 300_000  # 整个 Workflow 的全局超时

    def __post_init__(self):
        node_ids = {n.id for n in self.nodes}

        # 检查节点 id 唯一性
        if len(node_ids) != len(self.nodes):
            from collections import Counter
            id_counts = Counter(n.id for n in self.nodes)
            dupes = [nid for nid, count in id_counts.items() if count > 1]
            raise ValueError(f"Duplicate node id(s): {set(dupes)}")

        # 检查边的引用完整性
        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(
                    f"Edge references unknown node '{edge.from_node}' "
                    f"(edge: {edge.from_node} -> {edge.to_node})"
                )
            if edge.to_node not in node_ids:
                raise ValueError(
                    f"Edge references unknown node '{edge.to_node}' "
                    f"(edge: {edge.from_node} -> {edge.to_node})"
                )
