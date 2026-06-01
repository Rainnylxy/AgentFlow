"""端到端集成测试：验证 DSL → 验证 → 拓扑排序 → 可视化 → 序列化 全链路"""

from agentflow.dsl.types import Node, Edge, Workflow, NodeType
from agentflow.dsl.validator import validate_dag
from agentflow.dsl.graph import topological_sort, parallel_groups
from agentflow.dsl.visualizer import to_mermaid
from agentflow.dsl.serializer import to_yaml, from_yaml


class TestAgentFlowEndToEnd:
    """模拟 AgentFlow 的完整使用流程。"""

    def test_full_pipeline(self):
        # 1. 用户在 YAML 中定义一个菱形 DAG Workflow
        wf = Workflow(
            name="Order Processing Pipeline",
            description="Handle customer orders with fraud check and inventory",
            nodes=[
                Node(id="receive_order", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}, timeout_ms=30000),
                Node(id="fraud_check", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}, retry_max=2),
                Node(id="inventory_check", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}, retry_max=1),
                Node(id="fulfill", node_type=NodeType.AGENT,
                     config={"model": "gpt-4o"}),
            ],
            edges=[
                Edge(from_node="receive_order", to_node="fraud_check"),
                Edge(from_node="receive_order", to_node="inventory_check"),
                Edge(from_node="fraud_check", to_node="fulfill"),
                Edge(from_node="inventory_check", to_node="fulfill"),
            ],
        )

        # 2. 验证 DAG 合法性
        validate_dag(wf)  # 不应抛异常

        # 3. 计算执行顺序
        order = topological_sort(wf)
        assert order[0] == "receive_order"
        assert order[-1] == "fulfill"

        # 4. 计算并行分组
        groups = parallel_groups(wf)
        assert groups[0] == ["receive_order"]
        assert set(groups[1]) == {"fraud_check", "inventory_check"}  # 可并行
        assert groups[2] == ["fulfill"]

        # 5. 生成 Mermaid 可视化
        mermaid = to_mermaid(wf)
        assert "graph TD" in mermaid
        assert "receive_order" in mermaid
        assert "fraud_check" in mermaid
        assert "inventory_check" in mermaid

        # 6. YAML 序列化/反序列化
        yaml_str = to_yaml(wf)
        restored = from_yaml(yaml_str)
        assert restored.name == "Order Processing Pipeline"
        assert len(restored.nodes) == 4
        assert len(restored.edges) == 4

        # 7. 反序列化后仍可通过验证
        validate_dag(restored)

    def test_mermaid_exports_valid_markdown(self):
        """Mermaid 输出可直接嵌入 Markdown。"""
        wf = Workflow(
            name="Simple Pipeline",
            nodes=[Node(id="a", node_type=NodeType.AGENT)],
            edges=[],
        )
        mermaid = to_mermaid(wf)
        markdown_block = f"```mermaid\n{mermaid}\n```"
        assert "```mermaid" in markdown_block
        assert "graph TD" in markdown_block
