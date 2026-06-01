import pytest
from agentflow.dsl.types import Node, Edge, Workflow, NodeType
from agentflow.dsl.validator import validate_dag, DAGValidationError


class TestValidateDAG:
    def test_valid_linear_dag_passes(self):
        """最简单的合法 DAG：线性的 entry → a → b。"""
        wf = Workflow(
            name="valid-linear",
            nodes=[
                Node(id="entry", node_type=NodeType.AGENT),
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="entry", to_node="a"),
                Edge(from_node="a", to_node="b"),
            ],
        )
        validate_dag(wf)  # 不应抛异常

    def test_diamond_dag_passes(self):
        """菱形 DAG：entry 分叉到 left/right，汇合到 end。"""
        wf = Workflow(
            name="diamond",
            nodes=[
                Node(id="entry", node_type=NodeType.AGENT),
                Node(id="left", node_type=NodeType.AGENT),
                Node(id="right", node_type=NodeType.AGENT),
                Node(id="end", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="entry", to_node="left"),
                Edge(from_node="entry", to_node="right"),
                Edge(from_node="left", to_node="end"),
                Edge(from_node="right", to_node="end"),
            ],
        )
        validate_dag(wf)  # 不应抛异常

    def test_cycle_detected_raises(self):
        """a → b → a 构成环，必须检测到。"""
        wf = Workflow(
            name="cyclic",
            nodes=[
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="b", to_node="a"),
            ],
        )
        with pytest.raises(DAGValidationError, match="Cycle detected"):
            validate_dag(wf)

    def test_three_node_cycle_raises(self):
        """三节点环：a → b → c → a。"""
        wf = Workflow(
            name="triple-cycle",
            nodes=[
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
                Node(id="c", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="b", to_node="c"),
                Edge(from_node="c", to_node="a"),
            ],
        )
        with pytest.raises(DAGValidationError, match="Cycle detected"):
            validate_dag(wf)

    def test_disconnected_graph_raises(self):
        """不连通的图：两个节点没有边。"""
        wf = Workflow(
            name="disconnected",
            nodes=[
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
            ],
            edges=[],
        )
        with pytest.raises(DAGValidationError, match="Multiple entry nodes"):
            validate_dag(wf)

    def test_all_disconnected_nodes_raises(self):
        """三节点全不连通——每个都是"入口"。"""
        wf = Workflow(
            name="all-disconnected",
            nodes=[
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
                Node(id="c", node_type=NodeType.AGENT),
            ],
            edges=[],
        )
        with pytest.raises(DAGValidationError, match="Multiple entry nodes"):
            validate_dag(wf)
