"""Mermaid 图可视化：.to_mermaid() 将 Workflow 渲染为 Mermaid 字符串"""

from agentflow.dsl.types import Workflow, NodeKind


def to_mermaid(workflow: Workflow) -> str:
    """将 Workflow 渲染为 Mermaid 流程图。

    可直接复制到 Markdown 的 ```mermaid 代码块中渲染。
    不同 NodeKind 用不同形状表示：
      - AGENT:    矩形 [ ]
      - TOOL:     梯形 [/ \]
      - HUMAN:    斜四边形 [/ /]
      - SUBGRAPH: 子程序形状 [[ ]]
    """
    lines = ["graph TD"]

    # 节点定义
    for node in workflow.nodes:
        shape = _shape_for(node)
        label = node.label or node.id
        lines.append(f"    {node.id}{shape[0]}{label}{shape[1]};")

    # 边定义（含条件标注）
    for edge in workflow.edges:
        line = f"    {edge.from_node} -->"
        if edge.condition:
            line += f"|{edge.condition}|"
        line += f" {edge.to_node}"
        lines.append(line)

    return "\n".join(lines)


def _shape_for(node) -> tuple[str, str]:
    """返回 Mermaid 形状的左右分隔符。"""
    return {
        NodeKind.AGENT:    ("[", "]"),
        NodeKind.TOOL:     ("[/", "\\]"),
        NodeKind.HUMAN:    ("[/", "/]"),
        NodeKind.SUBGRAPH: ("[[", "]]"),
    }.get(node.kind, ("[", "]"))
