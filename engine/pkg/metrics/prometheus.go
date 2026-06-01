package metrics

import (
	"sort"
	"sync"
	"sync/atomic"
)

// Metrics 收集编排引擎的执行指标。
// 使用 atomic 计数器（无锁）+ Mutex 保护延迟切片（写入频率低）。
type Metrics struct {
	namespace    string
	totalCalls   atomic.Int64
	successCount atomic.Int64
	failureCount atomic.Int64
	latencies    []int64
	mu           sync.Mutex
}

// NewMetrics 创建新的 Metrics 实例。
func NewMetrics(namespace string) *Metrics {
	return &Metrics{namespace: namespace}
}

// RecordExecution 记录一次节点执行的指标。
// status: "success" 或 "failed"
// latencyMs: 执行耗时（毫秒）
func (m *Metrics) RecordExecution(workflowID, nodeID, status string, latencyMs int64) {
	m.totalCalls.Add(1)
	if status == "success" {
		m.successCount.Add(1)
	} else {
		m.failureCount.Add(1)
	}
	m.mu.Lock()
	m.latencies = append(m.latencies, latencyMs)
	m.mu.Unlock()
}

// ExecSummary 是执行摘要。
type ExecSummary struct {
	TotalExecutions int64
	SuccessCount    int64
	FailureCount    int64
}

// Summary 返回执行计数摘要（无锁读取）。
func (m *Metrics) Summary() ExecSummary {
	return ExecSummary{
		TotalExecutions: m.totalCalls.Load(),
		SuccessCount:    m.successCount.Load(),
		FailureCount:    m.failureCount.Load(),
	}
}

// LatencyStats 是延迟分布统计。
type LatencyStats struct {
	AvgMs int64
	P50Ms int64
	P99Ms int64
	MaxMs int64
}

// LatencyStats 计算延迟的百分位分布。
// 返回 avg / p50 / p99 / max。
func (m *Metrics) LatencyStats() LatencyStats {
	m.mu.Lock()
	sorted := make([]int64, len(m.latencies))
	copy(sorted, m.latencies)
	m.mu.Unlock()

	if len(sorted) == 0 {
		return LatencyStats{}
	}

	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	n := int64(len(sorted))
	var sum int64
	for _, v := range sorted {
		sum += v
	}

	return LatencyStats{
		AvgMs: sum / n,
		P50Ms: sorted[n/2],
		P99Ms: sorted[(n*99)/100],
		MaxMs: sorted[n-1],
	}
}
