package dag

import (
	"sync"
	"time"
)

// Status 表示 DAG 或单个节点的执行状态。
type Status string

const (
	StatusSuccess Status = "success"
	StatusFailed  Status = "failed"
	StatusTimeout Status = "timeout"
)

// NodeHandler 是每个节点的可执行函数。
// 接收 NodeContext，返回 NodeResult 或错误。
type NodeHandler func(ctx *NodeContext) (*NodeResult, error)

// Node 是 DAG 中的一个可执行节点。
type Node struct {
	ID      string
	Handler NodeHandler
	Timeout time.Duration
}

// NodeContext 在执行时传递给节点处理器，包含上游节点的输出。
type NodeContext struct {
	WorkflowID string
	NodeID     string
	Inputs     map[string]string // key = 上游节点 ID, value = 输出
}

// NodeResult 是单个节点的执行结果。
type NodeResult struct {
	Output   string
	Error    string
	Duration time.Duration
}

// Edge 是 DAG 中的有向边。
type Edge struct {
	From string
	To   string
}

// DAG 是一个有向无环图，包含节点和边。
type DAG struct {
	Name  string
	Nodes []*Node
	Edges []*Edge
}

// NewDAG 创建一个新的 DAG。
func NewDAG(name string, nodes []*Node, edges []*Edge) *DAG {
	return &DAG{Name: name, Nodes: nodes, Edges: edges}
}

// DAGResult 是整个 DAG 执行完成后返回的结果。
type DAGResult struct {
	Status      Status
	NodeResults map[string]*NodeResult
	TotalMs     int64
}

// Executor 负责按拓扑排序 + 并行分组执行 DAG。
type Executor struct{}

// NewExecutor 创建一个新的 Executor。
func NewExecutor() *Executor {
	return &Executor{}
}

// Execute 执行 DAG，按并行组调度节点，收集每个节点的结果。
//
// 算法：
//  1. 构建邻接表和入度表
//  2. 入度为 0 的节点 = 第一层（可并行执行）
//  3. 每层所有节点并发执行（goroutine）
//  4. 节点完成后，下游节点的入度减 1
//  5. 入度减到 0 的节点进入下一层
//  6. 重复直到所有节点完成
func (e *Executor) Execute(dag *DAG) (*DAGResult, error) {
	start := time.Now()

	// 构建图结构
	adj := make(map[string][]string)          // 邻接表
	inDegree := make(map[string]int)          // 入度
	nodeMap := make(map[string]*Node)         // ID → Node
	nodeInputs := make(map[string]map[string]string) // nodeID → {fromID: output}

	for _, n := range dag.Nodes {
		nodeMap[n.ID] = n
		inDegree[n.ID] = 0
		adj[n.ID] = []string{}
		nodeInputs[n.ID] = make(map[string]string)
	}
	for _, e := range dag.Edges {
		adj[e.From] = append(adj[e.From], e.To)
		inDegree[e.To]++
	}

	// 共享状态（需要 Mutex 保护）
	results := make(map[string]*NodeResult)
	var mu sync.Mutex
	var hasError bool
	var firstErr error
	errCh := make(chan error, len(dag.Nodes))

	// 按层迭代执行
	for {
		// 找出当前层（入度为 0 的节点）
		var level []string
		for nid, deg := range inDegree {
			if deg == 0 {
				level = append(level, nid)
			}
		}
		if len(level) == 0 {
			break
		}

		// 标记当前层节点为"已处理"（入度设为 -1）
		for _, nid := range level {
			inDegree[nid] = -1
		}

		// 并发执行当前层的所有节点
		var wg sync.WaitGroup
		for _, nid := range level {
			wg.Add(1)
			go func(nodeID string) {
				defer wg.Done()

				node := nodeMap[nodeID]
				ctx := &NodeContext{
					WorkflowID: dag.Name,
					NodeID:     nodeID,
					Inputs:     nodeInputs[nodeID],
				}

				// 执行节点处理器
				nodeStart := time.Now()
				result, err := node.Handler(ctx)
				if result == nil {
					result = &NodeResult{}
				}
				result.Duration = time.Since(nodeStart)

				if err != nil {
					result.Error = err.Error()
					errCh <- err
				}

				// 写入结果（加锁）
				mu.Lock()
				results[nodeID] = result
				if result.Error != "" {
					hasError = true
					if firstErr == nil {
						firstErr = err
					}
				}
				mu.Unlock()

				// 更新下游节点的入度和输入（加锁）
				mu.Lock()
				for _, neighbor := range adj[nodeID] {
					// 传递当前节点的输出给下游
					nodeInputs[neighbor][nodeID] = result.Output
					inDegree[neighbor]--
				}
				mu.Unlock()
			}(nid)
		}
		wg.Wait()
	}

	// 确定总体状态
	status := StatusSuccess
	if hasError {
		status = StatusFailed
	}

	return &DAGResult{
		Status:      status,
		NodeResults: results,
		TotalMs:     time.Since(start).Milliseconds(),
	}, firstErr
}
