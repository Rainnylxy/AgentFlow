import pytest
from agentflow.dsl.types import Node, Edge, Workflow, NodeType, FallbackPolicy


class TestNode:
    def test_create_basic_node(self):
        """最简单创建一个节点。"""
        node = Node(id="agent1", node_type=NodeType.AGENT)
        assert node.id == "agent1"
        assert node.node_type == NodeType.AGENT
        assert node.config == {}
        assert node.timeout_ms == 30000  # 默认30秒超时

    def test_create_node_with_full_config(self):
        """带全部配置的节点。"""
        node = Node(
            id="agent2",
            node_type=NodeType.AGENT,
            config={"model": "gpt-4o", "temperature": 0.7},
            timeout_ms=60000,
            retry_max=3,
            fallback=FallbackPolicy.SKIP,
        )
        assert node.config["model"] == "gpt-4o"
        assert node.timeout_ms == 60000
        assert node.retry_max == 3
        assert node.fallback == FallbackPolicy.SKIP

    def test_node_empty_id_raises(self):
        """空 id 应该在初始化时就报错——fail fast 原则。"""
        with pytest.raises(ValueError, match="id must be non-empty"):
            Node(id="", node_type=NodeType.AGENT)


class TestEdge:
    def test_create_basic_edge(self):
        """最基本的有向边。"""
        edge = Edge(from_node="entry", to_node="step2")
        assert edge.from_node == "entry"
        assert edge.to_node == "step2"
        assert edge.condition is None  # 无条件 = 总是执行

    def test_create_conditional_edge(self):
        """带条件的边——用于条件分支。"""
        edge = Edge(
            from_node="router",
            to_node="branch_a",
            condition='{{ output.score > 0.5 }}',
        )
        assert edge.condition == '{{ output.score > 0.5 }}'

    def test_self_loop_edge_raises(self):
        """不允许自环：from 和 to 不能是同一个节点。"""
        with pytest.raises(ValueError, match="from_node and to_node must differ"):
            Edge(from_node="self_loop", to_node="self_loop")


class TestWorkflow:
    def test_create_minimal_workflow(self):
        """最小 Workflow：一个节点，无边。"""
        wf = Workflow(
            name="hello-world",
            nodes=[Node(id="a", node_type=NodeType.AGENT)],
            edges=[],
        )
        assert wf.name == "hello-world"
        assert len(wf.nodes) == 1

    def test_create_linear_workflow(self):
        """线性工作流：entry → middle → end。"""
        wf = Workflow(
            name="linear-pipeline",
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
        assert len(wf.edges) == 2

    def test_duplicate_node_id_raises(self):
        """Workflow 内节点 id 不能重名。"""
        with pytest.raises(ValueError, match="Duplicate node id"):
            Workflow(
                name="dupes",
                nodes=[
                    Node(id="same", node_type=NodeType.AGENT),
                    Node(id="same", node_type=NodeType.AGENT),
                ],
                edges=[],
            )

    def test_edge_references_nonexistent_node_raises(self):
        """边引用的节点必须在 Workflow 内存在。"""
        with pytest.raises(ValueError, match="Edge references unknown node"):
            Workflow(
                name="missing-node",
                nodes=[Node(id="a", node_type=NodeType.AGENT)],
                edges=[Edge(from_node="a", to_node="ghost")],
            )
