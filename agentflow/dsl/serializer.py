"""YAML 序列化/反序列化：让用户用 YAML 文件定义 Workflow"""

from __future__ import annotations

import yaml
from agentflow.dsl.types import (
    Node, Edge, Workflow,
    NodeKind, FallbackPolicy,
    AgentConfig, ToolConfig, HumanConfig, LoopConfig,
)


def to_dict(workflow: Workflow) -> dict:
    """将 Workflow 转为纯 dict，准备序列化。"""
    return {
        "name": workflow.name,
        "description": workflow.description,
        "global_timeout_ms": workflow.global_timeout_ms,
        "nodes": [_node_to_dict(n) for n in workflow.nodes],
        "edges": [
            {
                "from": e.from_node,
                "to": e.to_node,
                "condition": e.condition,
            }
            for e in workflow.edges
        ],
    }


def _node_to_dict(n: Node) -> dict:
    d = {
        "id": n.id,
        "kind": n.kind.value,
        "label": n.label,
        "timeout_ms": n.timeout_ms,
        "retry_max": n.retry_max,
        "fallback": n.fallback.value,
        "default_value": n.default_value,
        "fallback_node_id": n.fallback_node_id,
    }
    # 执行体配置
    if n.kind == NodeKind.AGENT and n.agent:
        d["agent"] = {
            "model": n.agent.model,
            "prompt": n.agent.prompt,
            "tools": n.agent.tools,
            "thinking": n.agent.thinking,
            "memory": n.agent.memory,
        }
    elif n.kind == NodeKind.TOOL and n.tool:
        d["tool"] = {"name": n.tool.name, "inputs": n.tool.inputs}
    elif n.kind == NodeKind.HUMAN and n.human:
        d["human"] = {
            "prompt": n.human.prompt,
            "timeout_sec": n.human.timeout_sec,
            "default_response": n.human.default_response,
        }
    elif n.kind == NodeKind.SUBGRAPH and n.subgraph:
        d["subgraph"] = n.subgraph
    # 循环
    if n.loop:
        d["loop"] = {
            "max_iterations": n.loop.max_iterations,
            "condition": n.loop.condition,
            "break_on_tool": n.loop.break_on_tool,
        }
    return d


def from_dict(data: dict) -> Workflow:
    """从 dict 恢复 Workflow。"""
    return Workflow(
        name=data["name"],
        nodes=[_node_from_dict(n) for n in data["nodes"]],
        edges=[
            Edge(
                from_node=e["from"],
                to_node=e["to"],
                condition=e.get("condition"),
            )
            for e in data.get("edges", [])
        ],
        description=data.get("description", ""),
        global_timeout_ms=data.get("global_timeout_ms", 300_000),
    )


def _node_from_dict(n: dict) -> Node:
    kind = NodeKind(n.get("kind", "agent"))
    return Node(
        id=n["id"],
        kind=kind,
        label=n.get("label", ""),
        timeout_ms=n.get("timeout_ms", 30_000),
        retry_max=n.get("retry_max", 0),
        fallback=FallbackPolicy(n.get("fallback", "raise")),
        default_value=n.get("default_value"),
        fallback_node_id=n.get("fallback_node_id"),
        agent=_agent_from_dict(n.get("agent", {})) if kind == NodeKind.AGENT else None,
        tool=ToolConfig(**n["tool"]) if kind == NodeKind.TOOL and "tool" in n else None,
        human=HumanConfig(**n["human"]) if kind == NodeKind.HUMAN and "human" in n else None,
        subgraph=n.get("subgraph") if kind == NodeKind.SUBGRAPH else None,
        loop=LoopConfig(**n["loop"]) if "loop" in n else None,
    )


def _agent_from_dict(d: dict) -> AgentConfig:
    return AgentConfig(
        model=d.get("model", "gpt-4o"),
        prompt=d.get("prompt", ""),
        tools=d.get("tools", []),
        thinking=d.get("thinking", "react"),
        memory=d.get("memory", "standard"),
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
