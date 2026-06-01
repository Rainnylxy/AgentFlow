package dag

import (
	"sync"
	"testing"
	"time"
)

// recordHandler 返回一个 NodeHandler，它会记录执行顺序和模拟延迟。
// 用于测试拓扑顺序和并行度。
func recordHandler(id string, order *[]string, mu *sync.Mutex, delay time.Duration) NodeHandler {
	return func(ctx *NodeContext) (*NodeResult, error) {
		time.Sleep(delay)
		mu.Lock()
		*order = append(*order, id)
		mu.Unlock()
		return &NodeResult{Output: id}, nil
	}
}

func TestExecutorLinearWorkflow(t *testing.T) {
	// 最简单的线性 DAG：a → b
	var order []string
	var mu sync.Mutex

	nodes := []*Node{
		{ID: "a", Handler: recordHandler("a", &order, &mu, 10*time.Millisecond)},
		{ID: "b", Handler: recordHandler("b", &order, &mu, 10*time.Millisecond)},
	}
	edges := []*Edge{
		{From: "a", To: "b"},
	}
	dag := NewDAG("linear", nodes, edges)

	result, err := NewExecutor().Execute(dag)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if result.Status != StatusSuccess {
		t.Fatalf("expected success, got %v", result.Status)
	}
	if len(result.NodeResults) != 2 {
		t.Fatalf("expected 2 node results, got %d", len(result.NodeResults))
	}
}

func TestExecutorTopologicalOrder(t *testing.T) {
	// 验证 a 在 b 和 c 之前执行（拓扑顺序）
	var order []string
	var mu sync.Mutex

	nodes := []*Node{
		{ID: "a", Handler: recordHandler("a", &order, &mu, 50*time.Millisecond)},
		{ID: "b", Handler: recordHandler("b", &order, &mu, 10*time.Millisecond)},
		{ID: "c", Handler: recordHandler("c", &order, &mu, 10*time.Millisecond)},
	}
	edges := []*Edge{
		{From: "a", To: "b"},
		{From: "a", To: "c"},
	}
	dag := NewDAG("order-test", nodes, edges)

	_, err := NewExecutor().Execute(dag)
	if err != nil {
		t.Fatal(err)
	}
	// a 必须在 b 和 c 之前
	if order[0] != "a" {
		t.Fatalf("expected 'a' first, got %v", order)
	}
}

func TestExecutorParallelExecution(t *testing.T) {
	// entry → left, right → end
	// left 和 right 应该并行执行（总时间 < 串行时间）
	var order []string
	var mu sync.Mutex

	nodes := []*Node{
		{ID: "entry", Handler: recordHandler("entry", &order, &mu, 30*time.Millisecond)},
		{ID: "left", Handler: recordHandler("left", &order, &mu, 200*time.Millisecond)},
		{ID: "right", Handler: recordHandler("right", &order, &mu, 200*time.Millisecond)},
		{ID: "end", Handler: func(ctx *NodeContext) (*NodeResult, error) {
			mu.Lock()
			order = append(order, "end")
			mu.Unlock()
			return &NodeResult{Output: "end"}, nil
		}},
	}
	edges := []*Edge{
		{From: "entry", To: "left"},
		{From: "entry", To: "right"},
		{From: "left", To: "end"},
		{From: "right", To: "end"},
	}
	dag := NewDAG("parallel-test", nodes, edges)

	start := time.Now()
	_, err := NewExecutor().Execute(dag)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatal(err)
	}
	// 串行需要 >= 430ms (30+200+200)，并行约 230ms
	if elapsed >= 350*time.Millisecond {
		t.Fatalf("expected parallel execution (<350ms), got %v", elapsed)
	}
}

func TestExecutorNodeFailure(t *testing.T) {
	// 节点失败时，执行结果应标记为 failed
	nodes := []*Node{
		{ID: "a", Handler: func(ctx *NodeContext) (*NodeResult, error) {
			return &NodeResult{Output: "", Error: "boom"}, nil
		}},
	}
	dag := NewDAG("fail-test", nodes, nil)

	result, _ := NewExecutor().Execute(dag)
	nodeResult := result.NodeResults["a"]
	if nodeResult.Error != "boom" {
		t.Fatalf("expected error 'boom', got '%s'", nodeResult.Error)
	}
}
