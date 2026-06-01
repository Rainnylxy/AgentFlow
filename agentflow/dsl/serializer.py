"""YAML 序列化/反序列化：让用户用 YAML 文件定义 Workflow"""

import yaml
from agentflow.dsl.types import Node, Edge, Workflow, NodeType, FallbackPolicy


def to_dict(workflow: Workflow) -> dict:
    """将 Workflow 转为纯 dict，准备序列化。"""
    return {
        "name": workflow.name,
        "description": workflow.description,
        "max_iterations": workflow.max_iterations,
        "global_timeout_ms": workflow.global_timeout_ms,
        "nodes": [
            {
                "id": n.id,
                "type": n.node_type.value,
                "config": n.config,
                "timeout_ms": n.timeout_ms,
                "retry_max": n.retry_max,
                "fallback": n.fallback.value,
                "default_value": n.default_value,
                "fallback_node_id": n.fallback_node_id,
            }
            for n in workflow.nodes
        ],
        "edges": [
            {
                "from": e.from_node,
                "to": e.to_node,
                "condition": e.condition,
            }
            for e in workflow.edges
        ],
    }


def from_dict(data: dict) -> Workflow:
    """从 dict 恢复 Workflow。"""
    nodes = [
        Node(
            id=n["id"],
            node_type=NodeType(n["type"]),
            config=n.get("config", {}),
            timeout_ms=n.get("timeout_ms", 30000),
            retry_max=n.get("retry_max", 0),
            fallback=FallbackPolicy(n.get("fallback", "raise")),
            default_value=n.get("default_value"),
            fallback_node_id=n.get("fallback_node_id"),
        )
        for n in data["nodes"]
    ]

    edges = [
        Edge(
            from_node=e["from"],
            to_node=e["to"],
            condition=e.get("condition"),
        )
        for e in data.get("edges", [])
    ]

    return Workflow(
        name=data["name"],
        nodes=nodes,
        edges=edges,
        description=data.get("description", ""),
        max_iterations=data.get("max_iterations", 10),
        global_timeout_ms=data.get("global_timeout_ms", 300_000),
    )


def to_yaml(workflow: Workflow) -> str:
    """将 Workflow 序列化为 YAML 字符串。"""
    return yaml.dump(
        to_dict(workflow),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def from_yaml(source: str) -> Workflow:
    """从 YAML 字符串或 .yaml 文件路径反序列化。

    自动检测：如果 source 以 .yaml/.yml 结尾，视为文件路径；
    否则视为 YAML 字符串。
    """
    if source.endswith((".yaml", ".yml")):
        with open(source, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(source)

    if data is None:
        raise ValueError("Empty YAML document")

    return from_dict(data)
