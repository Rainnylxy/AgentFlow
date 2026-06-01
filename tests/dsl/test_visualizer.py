from agentflow.dsl.types import Node, Edge, Workflow, NodeType
from agentflow.dsl.visualizer import to_mermaid


class TestToMermaid:
    def test_linear_workflow_generates_mermaid(self):
        """线性 Workflow → 包含所有节点和边的 Mermaid 图。"""
        wf = Workflow(
            name="Linear Pipeline",
            nodes=[
                Node(id="entry", node_type=NodeType.AGENT),
                Node(id="middle", node_type=NodeType.AGENT),
                Node(id="end", node_type=NodeType.AGENT),
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
                Node(id="router", node_type=NodeType.CONDITION),
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
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
        """不同 NodeType 用不同形状区分。"""
        wf = Workflow(
            name="Shapes",
            nodes=[
                Node(id="agent_node", node_type=NodeType.AGENT),
                Node(id="condition_node", node_type=NodeType.CONDITION),
            ],
            edges=[],
        )
        result = to_mermaid(wf)
        # Agent 是矩形 [ ]
        assert "agent_node[" in result
        # Condition 是菱形 { }
        assert "condition_node{" in result
