from agentflow.dsl.types import Node, Edge, Workflow, NodeType
from agentflow.dsl.graph import topological_sort, parallel_groups


class TestTopologicalSort:
    def test_linear_chain_keeps_order(self):
        """线性链 a→b→c 输出必须保持顺序。"""
        wf = Workflow(
            name="linear",
            nodes=[
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
                Node(id="c", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="b", to_node="c"),
            ],
        )
        order = topological_sort(wf)
        assert order == ["a", "b", "c"]

    def test_diamond_entry_first_end_last(self):
        """菱形 DAG：entry 永远第一，end 永远最后。"""
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
        order = topological_sort(wf)
        assert order[0] == "entry"
        assert order[-1] == "end"
        # left 和 right 在 entry 之后、end 之前（不关心它们之间的顺序）
        assert set(order[1:3]) == {"left", "right"}

    def test_single_node_workflow(self):
        """单节点 Workflow：只有一个元素。"""
        wf = Workflow(
            name="singleton",
            nodes=[Node(id="a", node_type=NodeType.AGENT)],
            edges=[],
        )
        order = topological_sort(wf)
        assert order == ["a"]


class TestParallelGroups:
    def test_linear_yields_singleton_groups(self):
        """线性链：每个阶段只有一个节点可执行。"""
        wf = Workflow(
            name="linear",
            nodes=[
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
            ],
            edges=[Edge(from_node="a", to_node="b")],
        )
        groups = parallel_groups(wf)
        assert groups == [["a"], ["b"]]

    def test_diamond_yields_parallel_middle(self):
        """菱形 DAG：中间层的 left/right 可并行。"""
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
        groups = parallel_groups(wf)
        assert groups[0] == ["entry"]              # 第1层：入口
        assert set(groups[1]) == {"left", "right"} # 第2层：可并行
        assert groups[2] == ["end"]                # 第3层：终点

    def test_three_parallel_branches(self):
        """entry 分叉出三个并行分支。"""
        wf = Workflow(
            name="fan-out",
            nodes=[
                Node(id="entry", node_type=NodeType.AGENT),
                Node(id="a", node_type=NodeType.AGENT),
                Node(id="b", node_type=NodeType.AGENT),
                Node(id="c", node_type=NodeType.AGENT),
                Node(id="end", node_type=NodeType.AGENT),
            ],
            edges=[
                Edge(from_node="entry", to_node="a"),
                Edge(from_node="entry", to_node="b"),
                Edge(from_node="entry", to_node="c"),
                Edge(from_node="a", to_node="end"),
                Edge(from_node="b", to_node="end"),
                Edge(from_node="c", to_node="end"),
            ],
        )
        groups = parallel_groups(wf)
        assert groups[0] == ["entry"]
        assert set(groups[1]) == {"a", "b", "c"}   # 三者可并行
        assert groups[2] == ["end"]
