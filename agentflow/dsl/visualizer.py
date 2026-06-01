"""Mermaid 图可视化：.to_mermaid() 将 Workflow 渲染为 Mermaid 字符串"""

from agentflow.dsl.types import Workflow, NodeType


def to_mermaid(workflow: Workflow) -> str:
    """将 Workflow 渲染为 Mermaid 流程图。

    可直接复制到 Markdown 的 ```mermaid 代码块中渲染。
    不同 NodeType 用不同形状表示：
      - AGENT: 矩形 [ ]
      - CONDITION: 菱形 { }
      - PARALLEL: 六边形 {{ }}
      - LOOP: 圆角矩形 ( )
    """
    lines = ["graph TD"]

    # 节点定义
    for node in workflow.nodes:
        shape = _shape_for(node)
        lines.append(f"    {node.id}{shape[0]}{node.id}{shape[1]};")

    # 边定义
    for edge in workflow.edges:
        line = f"    {edge.from_node} -->"
        if edge.condition:
            line += f"|{edge.condition}|"
        line += f" {edge.to_node}"
        lines.append(line)

    return "\n".join(lines)


def _shape_for(node) -> tuple[str, str]:
    """返回 Mermaid 形状的左右分隔符。"""
    shapes = {
        NodeType.AGENT: ("[", "]"),
        NodeType.CONDITION: ("{", "}"),
        NodeType.PARALLEL: ("{{", "}}"),
        NodeType.LOOP: ("(", ")"),
    }
    return shapes.get(node.node_type, ("[", "]"))
