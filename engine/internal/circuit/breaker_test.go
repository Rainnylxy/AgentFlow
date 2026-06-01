package circuit

import (
	"errors"
	"testing"
	"time"
)

func TestBreakerClosedAllowsCalls(t *testing.T) {
	// 正常状态下，所有请求都应通过
	cb := NewBreaker(Config{FailureThreshold: 3, RecoveryTimeout: time.Second})
	for i := 0; i < 5; i++ {
		err := cb.Execute(func() error { return nil })
		if err != nil {
			t.Fatalf("call %d: expected no error, got %v", i, err)
		}
	}
	if cb.State() != StateClosed {
		t.Fatalf("expected closed state, got %v", cb.State())
	}
}

func TestBreakerOpensAfterFailures(t *testing.T) {
	// 连续失败达到阈值 → 断路器跳闸
	cb := NewBreaker(Config{FailureThreshold: 2, RecoveryTimeout: 100 * time.Millisecond})
	failErr := errors.New("boom")

	cb.Execute(func() error { return failErr })
	cb.Execute(func() error { return failErr })

	// 第三次调用应该被拒绝（断路器打开）
	err := cb.Execute(func() error { return nil })
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen, got %v", err)
	}
	if cb.State() != StateOpen {
		t.Fatalf("expected open state, got %v", cb.State())
	}
}

func TestBreakerHalfOpenAllowsProbe(t *testing.T) {
	// 断路器打开后，过了 RecoveryTimeout 应该进入半开状态允许探测
	cb := NewBreaker(Config{FailureThreshold: 1, RecoveryTimeout: 50 * time.Millisecond})

	// 触发熔断
	cb.Execute(func() error { return errors.New("fail") })
	if cb.State() != StateOpen {
		t.Fatal("expected open state after failure")
	}

	// 等待恢复时间
	time.Sleep(60 * time.Millisecond)

	// 探测调用应该成功
	err := cb.Execute(func() error { return nil })
	if err != nil {
		t.Fatalf("probe call should succeed, got %v", err)
	}
	if cb.State() != StateClosed {
		t.Fatalf("expected closed after successful probe, got %v", cb.State())
	}
}

func TestBreakerHalfOpenFailsGoesBackToOpen(t *testing.T) {
	// 半开状态下探测失败 → 回到 Open 状态
	cb := NewBreaker(Config{FailureThreshold: 1, RecoveryTimeout: 30 * time.Millisecond})

	cb.Execute(func() error { return errors.New("fail") })
	time.Sleep(40 * time.Millisecond)

	// 探测失败
	cb.Execute(func() error { return errors.New("fail again") })

	// 应该回到 Open
	if cb.State() != StateOpen {
		t.Fatalf("expected open after failed probe, got %v", cb.State())
	}

	// 后续调用应被拒绝
	err := cb.Execute(func() error { return nil })
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen after failed probe, got %v", err)
	}
}
