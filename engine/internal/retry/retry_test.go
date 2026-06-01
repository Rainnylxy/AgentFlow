package retry

import (
	"errors"
	"testing"
	"time"
)

func TestRetrySucceedsOnFirstTry(t *testing.T) {
	// 成功时不应重试
	calls := 0
	err := Do(func() error {
		calls++
		return nil
	}, WithMaxRetries(3), WithBackoff(10*time.Millisecond, time.Second))

	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 call, got %d", calls)
	}
}

func TestRetrySucceedsAfterFailures(t *testing.T) {
	// 前 N 次失败后成功
	calls := 0
	err := Do(func() error {
		calls++
		if calls < 3 {
			return errors.New("transient error")
		}
		return nil
	}, WithMaxRetries(5), WithBackoff(5*time.Millisecond, time.Second))

	if err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
	if calls != 3 {
		t.Fatalf("expected 3 calls (2 failures + 1 success), got %d", calls)
	}
}

func TestRetryExhaustedReturnsLastError(t *testing.T) {
	// 重试耗尽后返回最后一次的错误
	expected := errors.New("persistent error")
	err := Do(func() error {
		return expected
	}, WithMaxRetries(2), WithBackoff(1*time.Millisecond, time.Second))

	if !errors.Is(err, expected) {
		t.Fatalf("expected %v, got %v", expected, err)
	}
}

func TestExponentialBackoffIncreases(t *testing.T) {
	// 退避延迟应该递增
	delays := computeDelays(5, 10*time.Millisecond, 10*time.Second)
	for i := 1; i < len(delays); i++ {
		if delays[i] < delays[i-1] {
			t.Fatalf("delays should increase: delays[%d]=%v < delays[%d]=%v",
				i, delays[i], i-1, delays[i-1])
		}
	}
	// 第 0 次延迟 < 第 4 次延迟（至少翻倍了 4 次）
	if delays[4] <= delays[0] {
		t.Fatalf("last delay %v should be > first delay %v", delays[4], delays[0])
	}
}
