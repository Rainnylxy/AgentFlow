import tempfile
import os
from agentflow.dsl.types import Node, Edge, Workflow, NodeType, FallbackPolicy
from agentflow.dsl.serializer import to_yaml, from_yaml, to_dict, from_dict


class TestToDictFromDict:
    def test_roundtrip_preserves_all_data(self):
        """序列化到 dict 再恢复，数据应完全一致。"""
        wf = Workflow(
            name="TestWF",
            description="A test workflow",
            max_iterations=5,
            global_timeout_ms=600_000,
            nodes=[
                Node(
                    id="entry",
                    node_type=NodeType.AGENT,
                    config={"model": "gpt-4o"},
                    timeout_ms=60000,
                    retry_max=3,
                    fallback=FallbackPolicy.SKIP,
                ),
                Node(id="end", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="entry", to_node="end"),
            ],
        )
        d = to_dict(wf)
        restored = from_dict(d)

        assert restored.name == "TestWF"
        assert restored.description == "A test workflow"
        assert restored.max_iterations == 5
        assert len(restored.nodes) == 2
        assert restored.nodes[0].config["model"] == "gpt-4o"
        assert restored.nodes[0].fallback == FallbackPolicy.SKIP
        assert restored.edges[0].from_node == "entry"
        assert restored.edges[0].to_node == "end"

    def test_conditional_edge_roundtrip(self):
        """条件边的 condition 字段应正确保留。"""
        wf = Workflow(
            name="Cond",
            nodes=[
                Node(id="r", node_type=NodeType.CONDITION),
                Node(id="a", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="r", to_node="a", condition="{{ output > 0.5 }}"),
            ],
        )
        restored = from_dict(to_dict(wf))
        assert restored.edges[0].condition == "{{ output > 0.5 }}"


class TestYAMLSerialization:
    def test_yaml_roundtrip(self):
        """YAML 字符串 → Workflow → 再导出 YAML 应一致。"""
        wf = Workflow(
            name="YAML-Test",
            nodes=[
                Node(id="entry", node_type=NodeType.AGENT),
                Node(id="exit", node_type=NodeType.AGENT),
            ],
            edges=[Edge(from_node="entry", to_node="exit")],
        )
        yaml_str = to_yaml(wf)
        assert "YAML-Test" in yaml_str
        assert "entry" in yaml_str

        restored = from_yaml(yaml_str)
        assert restored.name == "YAML-Test"
        assert len(restored.nodes) == 2

    def test_from_yaml_file(self):
        """支持从 .yaml 文件直接加载。"""
        wf = Workflow(
            name="FileTest",
            nodes=[Node(id="n1", node_type=NodeType.AGENT)],
            edges=[],
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(to_yaml(wf))
            path = f.name

        try:
            restored = from_yaml(path)
            assert restored.name == "FileTest"
        finally:
            os.unlink(path)
