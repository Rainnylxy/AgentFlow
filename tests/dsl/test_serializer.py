import tempfile
import os
from agentflow.dsl.types import (
    Node, Edge, Workflow,
    NodeKind, FallbackPolicy,
    AgentConfig, LoopConfig,
)
from agentflow.dsl.serializer import to_yaml, from_yaml, to_dict, from_dict


class TestToDictFromDict:
    def test_roundtrip_preserves_all_data(self):
        """序列化到 dict 再恢复，内容应完全一致。"""
        wf = Workflow(
            name="TestWF",
            description="A test workflow",
            global_timeout_ms=600_000,
            nodes=[
                Node(
                    id="entry",
                    kind=NodeKind.AGENT,
                    agent=AgentConfig(model="gpt-4o", prompt="Helpful"),
                    timeout_ms=60000,
                    retry_max=3,
                    fallback=FallbackPolicy.SKIP,
                ),
                Node(id="end", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="entry", to_node="end"),
            ],
        )

        d = to_dict(wf)
        restored = from_dict(d)

        assert restored.name == "TestWF"
        assert len(restored.nodes) == 2
        assert restored.nodes[0].agent.model == "gpt-4o"
        assert restored.nodes[0].timeout_ms == 60000
        assert restored.nodes[0].retry_max == 3

    def test_conditional_edge_roundtrip(self):
        """边上的 condition 字段应正确持久化。"""
        wf = Workflow(
            name="Cond",
            nodes=[
                Node(id="r", kind=NodeKind.AGENT),
                Node(id="a", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="r", to_node="a", condition="{{ output > 0.5 }}"),
            ],
        )
        d = to_dict(wf)
        restored = from_dict(d)
        assert restored.edges[0].condition == "{{ output > 0.5 }}"


class TestYAMLSerialization:
    def test_yaml_roundtrip(self):
        wf = Workflow(
            name="YAMLTest",
            nodes=[
                Node(id="entry", kind=NodeKind.AGENT, agent=AgentConfig(model="gpt-4o")),
                Node(id="exit", kind=NodeKind.AGENT),
            ],
            edges=[Edge(from_node="entry", to_node="exit")],
        )
        yaml_str = to_yaml(wf)
        restored = from_yaml(yaml_str)

        assert restored.name == "YAMLTest"
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1

    def test_from_yaml_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("name: FileTest\nnodes:\n- id: n1\n  kind: agent\nedges: []\n")
            path = f.name
        try:
            wf = from_yaml(path)
            assert wf.name == "FileTest"
            assert len(wf.nodes) == 1
        finally:
            os.unlink(path)
