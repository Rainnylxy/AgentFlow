package metrics

import (
	"testing"
)

func TestMetricsRecording(t *testing.T) {
	m := NewMetrics("agentflow")

	m.RecordExecution("wf-1", "node-a", "success", 150)
	m.RecordExecution("wf-1", "node-b", "failed", 42)
	m.RecordExecution("wf-2", "node-a", "success", 200)

	summary := m.Summary()
	if summary.TotalExecutions != 3 {
		t.Fatalf("expected 3 total, got %d", summary.TotalExecutions)
	}
	if summary.SuccessCount != 2 {
		t.Fatalf("expected 2 successes, got %d", summary.SuccessCount)
	}
	if summary.FailureCount != 1 {
		t.Fatalf("expected 1 failure, got %d", summary.FailureCount)
	}
}

func TestMetricsLatencyStats(t *testing.T) {
	m := NewMetrics("test")

	m.RecordExecution("wf", "a", "success", 100)
	m.RecordExecution("wf", "b", "success", 200)
	m.RecordExecution("wf", "c", "success", 300)

	stats := m.LatencyStats()
	// avg = (100+200+300)/3 = 200
	if stats.AvgMs != 200 {
		t.Fatalf("expected avg 200ms, got %d", stats.AvgMs)
	}
	// p50 = median = 200
	if stats.P50Ms != 200 {
		t.Fatalf("expected p50 200ms, got %d", stats.P50Ms)
	}
}

func TestMetricsEmptyStats(t *testing.T) {
	m := NewMetrics("empty")
	stats := m.LatencyStats()
	if stats.AvgMs != 0 || stats.P50Ms != 0 || stats.P99Ms != 0 {
		t.Fatal("empty metrics should return zeros")
	}
}
