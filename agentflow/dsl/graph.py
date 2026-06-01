"""DAG 图算法：拓扑排序 + 并行分組"""

from collections import deque
from agentflow.dsl.types import Workflow


def _build_adj_and_in_degree(workflow: Workflow):
    """构建邻接表和入度表——两个算法共享的预处理步骤。"""
    nodes = {n.id for n in workflow.nodes}
    in_degree = {nid: 0 for nid in nodes}
    adj = {nid: [] for nid in nodes}

    for edge in workflow.edges:
        adj[edge.from_node].append(edge.to_node)
        in_degree[edge.to_node] += 1

    return adj, in_degree


def topological_sort(workflow: Workflow) -> list[str]:
    """返回 Workflow DAG 的拓扑排序。

    Kahn's algorithm 的应用：
    1. 找到所有入度为 0 的节点（入口）
    2. 逐个输出，将其下游节点的入度减 1
    3. 如有新节点入度变为 0，加入队列
    4. 重复直到所有节点都被输出

    时间复杂度：O(V + E)
    """
    adj, in_degree = _build_adj_and_in_degree(workflow)

    queue = deque([
        nid for nid, deg in in_degree.items() if deg == 0
    ])
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)

        for neighbor in adj[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


def parallel_groups(workflow: Workflow) -> list[list[str]]:
    """返回按并行分层后的节点组。

    每一组包含所有"上游依赖已全部完成"的节点，组内节点可以并发执行。
    这直接用于 Go 编排引擎的并行调度。

    例如：
        entry → left  → end
        entry → right → end
    输出：[[entry], [left, right], [end]]

    时间复杂度：O(V + E)
    """
    adj, in_degree = _build_adj_and_in_degree(workflow)

    # 初始队列 = 第一组（入口节点）
    current_level = sorted(
        nid for nid, deg in in_degree.items() if deg == 0
    )
    groups = []

    while current_level:
        groups.append(current_level)
        next_level = []

        for current in current_level:
            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_level.append(neighbor)

        current_level = sorted(next_level)

    return groups
