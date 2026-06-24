from agentflow.dsl.types import Node, Edge, Workflow, NodeKind
from agentflow.dsl.visualizer import to_mermaid


class TestToMermaid:
    def test_linear_workflow_generates_mermaid(self):
        """线性 Workflow → 包含所有节点和边的 Mermaid 图。"""
        wf = Workflow(
            name="Linear Pipeline",
            nodes=[
                Node(id="entry", kind=NodeKind.AGENT),
                Node(id="middle", kind=NodeKind.AGENT),
                Node(id="end", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="entry", to_node="middle"),
                Edge(from_node="middle", to_node="end"),
            ],
        )
        result = to_mermaid(wf)
        assert "graph TD" in result
        assert "entry --> middle" in result
        assert "middle --> end" in result

    def test_conditional_edge_shows_label(self):
        """条件边会显示条件表达式。"""
        wf = Workflow(
            name="Conditional",
            nodes=[
                Node(id="router", kind=NodeKind.AGENT),
                Node(id="a", kind=NodeKind.AGENT),
                Node(id="b", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="router", to_node="a", condition="{{ score > 0.5 }}"),
                Edge(from_node="router", to_node="b", condition="{{ score <= 0.5 }}"),
            ],
        )
        result = to_mermaid(wf)
        assert "score > 0.5" in result
        assert "score <= 0.5" in result

    def test_different_node_types_have_different_shapes(self):
        """不同 NodeKind 用不同形状区分。"""
        from agentflow.dsl.types import ToolConfig, HumanConfig
        wf = Workflow(
            name="Shapes",
            nodes=[
                Node(id="agent_node", kind=NodeKind.AGENT),
                Node(id="tool_node", kind=NodeKind.TOOL, tool=ToolConfig(name="echo")),
                Node(id="human_node", kind=NodeKind.HUMAN, human=HumanConfig(prompt="OK?")),
                Node(id="sub_node", kind=NodeKind.SUBGRAPH, subgraph="child"),
            ],
            edges=[],
        )
        result = to_mermaid(wf)
        # Agent 矩形 [ ]
        assert "agent_node[" in result
        # Tool 梯形 [/ \]
        assert "tool_node[/" in result
        # Human 斜边 [/ /]
        assert "human_node[/" in result
        # Subgraph 子程序 [[ ]]
        assert "sub_node[[" in result
