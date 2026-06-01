"""DAG 验证器：环检测（Kahn's algorithm）+ 入口节点校验"""

from agentflow.dsl.types import Workflow


class DAGValidationError(Exception):
    """DAG 结构不合法时抛出。"""
    pass


def validate_dag(workflow: Workflow) -> None:
    """验证 Workflow 是一个合法的 DAG。

    规则：
    1. 必须恰好有一个入口节点（入度为 0 的节点）
    2. 无环（用 Kahn's algorithm 检测）

    时间复杂度：O(V + E)，V = 节点数，E = 边数
    """
    if not workflow.nodes:
        return  # 空图暂时放行

    nodes = {n.id for n in workflow.nodes}

    # 计算每个节点的入度
    in_degree = {nid: 0 for nid in nodes}
    adj = {nid: [] for nid in nodes}  # 邻接表

    for edge in workflow.edges:
        adj[edge.from_node].append(edge.to_node)
        in_degree[edge.to_node] += 1

    # ===== 规则 1：入口节点数量 =====
    entry_nodes = [nid for nid, deg in in_degree.items() if deg == 0]

    if len(entry_nodes) == 0:
        raise DAGValidationError(
            "Cycle detected: every node has at least one incoming edge, "
            "meaning the graph contains a closed loop with no entry point"
        )
    if len(entry_nodes) > 1:
        raise DAGValidationError(
            f"Multiple entry nodes found: {entry_nodes}. DAG must have exactly one entry point."
        )

    # ===== 规则 2：环检测 (Kahn's algorithm) =====
    # 核心思想：不断移除入度为 0 的节点，最后如果还有剩余节点 = 有环
    in_degree_copy = dict(in_degree)
    queue = list(entry_nodes)
    visited = 0

    while queue:
        current = queue.pop(0)
        visited += 1
        for neighbor in adj[current]:
            in_degree_copy[neighbor] -= 1
            if in_degree_copy[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(nodes):
        raise DAGValidationError(
            f"Cycle detected: only {visited}/{len(nodes)} nodes reachable "
            f"in topological order. The remaining {len(nodes) - visited} node(s) form a cycle."
        )
