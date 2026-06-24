import pytest
from agentflow.dsl.types import (
    Node, Edge, Workflow,
    NodeKind, FallbackPolicy,
    AgentConfig, ToolConfig, HumanConfig, LoopConfig,
)


class TestNode:
    def test_create_basic_agent_node(self):
        node = Node(id="agent1", kind=NodeKind.AGENT)
        assert node.kind == NodeKind.AGENT
        assert node.agent is not None  # auto-filled

    def test_create_node_with_agent_config(self):
        node = Node(
            id="agent2",
            kind=NodeKind.AGENT,
            agent=AgentConfig(model="gpt-4o", prompt="You are helpful."),
            timeout_ms=60_000,
            retry_max=3,
        )
        assert node.agent.model == "gpt-4o"
        assert node.timeout_ms == 60_000
        assert node.retry_max == 3

    def test_create_tool_node(self):
        node = Node(
            id="pdf_reader",
            kind=NodeKind.TOOL,
            tool=ToolConfig(name="read_pdf", inputs={"path": "/tmp/doc.pdf"}),
        )
        assert node.kind == NodeKind.TOOL
        assert node.tool.name == "read_pdf"

    def test_create_human_node(self):
        node = Node(
            id="approval",
            kind=NodeKind.HUMAN,
            human=HumanConfig(prompt="审核合同", timeout_sec=600),
        )
        assert node.kind == NodeKind.HUMAN
        assert node.human.timeout_sec == 600

    def test_create_subgraph_node(self):
        node = Node(
            id="risk_check",
            kind=NodeKind.SUBGRAPH,
            subgraph="risk_analysis_workflow",
        )
        assert node.kind == NodeKind.SUBGRAPH
        assert node.subgraph == "risk_analysis_workflow"

    def test_create_node_with_loop(self):
        node = Node(
            id="retry_step",
            kind=NodeKind.AGENT,
            loop=LoopConfig(max_iterations=5, condition="score > 0.8"),
        )
        assert node.loop.max_iterations == 5

    def test_node_empty_id_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Node(id="")


class TestEdge:
    def test_create_basic_edge(self):
        e = Edge(from_node="a", to_node="b")
        assert e.from_node == "a"
        assert e.condition is None

    def test_create_conditional_edge(self):
        e = Edge(from_node="a", to_node="b", condition="score > 0.5")
        assert e.condition == "score > 0.5"

    def test_self_loop_edge_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            Edge(from_node="x", to_node="x")


class TestWorkflow:
    def test_create_minimal_workflow(self):
        wf = Workflow(name="test", nodes=[Node(id="a")], edges=[])
        assert len(wf.nodes) == 1

    def test_create_linear_workflow(self):
        wf = Workflow(
            name="linear",
            nodes=[Node(id="entry"), Node(id="middle"), Node(id="end")],
            edges=[
                Edge(from_node="entry", to_node="middle"),
                Edge(from_node="middle", to_node="end"),
            ],
        )
        assert len(wf.nodes) == 3
        assert len(wf.edges) == 2

    def test_duplicate_node_id_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Workflow(name="bad", nodes=[Node(id="same"), Node(id="same")], edges=[])

    def test_edge_references_nonexistent_node_raises(self):
        with pytest.raises(ValueError, match="unknown node"):
            Workflow(name="bad", nodes=[Node(id="a")], edges=[Edge(from_node="a", to_node="ghost")])
