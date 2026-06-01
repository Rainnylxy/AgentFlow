package retry

import (
	"math"
	"time"
)

// Config 是重试策略的配置。
type Config struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
}

// Option 是函数式选项，用于配置重试行为。
// Go 惯用法：替代 Python 的 **kwargs，编译期类型安全。
type Option func(*Config)

// WithMaxRetries 设置最大重试次数。
func WithMaxRetries(n int) Option {
	return func(c *Config) { c.MaxRetries = n }
}

// WithBackoff 设置退避的起始延迟和最大延迟。
func WithBackoff(base, max time.Duration) Option {
	return func(c *Config) {
		c.BaseDelay = base
		c.MaxDelay = max
	}
}

func defaultConfig() *Config {
	return &Config{
		MaxRetries: 3,
		BaseDelay:  100 * time.Millisecond,
		MaxDelay:   30 * time.Second,
	}
}

// Do 执行给定的函数，失败时按指数退避策略重试。
//
// 重试次数 = MaxRetries + 1（第一次调用不算重试）
// 退避延迟 = BaseDelay * 2^attempt（封顶 MaxDelay）
//
// 使用示例：
//   err := retry.Do(func() error { ... },
//       retry.WithMaxRetries(5),
//       retry.WithBackoff(100*time.Millisecond, 5*time.Second))
func Do(fn func() error, opts ...Option) error {
	cfg := defaultConfig()
	for _, o := range opts {
		o(cfg)
	}

	var lastErr error
	for i := 0; i <= cfg.MaxRetries; i++ {
		if i > 0 {
			delay := computeDelay(i-1, cfg.BaseDelay, cfg.MaxDelay)
			time.Sleep(delay)
		}
		lastErr = fn()
		if lastErr == nil {
			return nil
		}
	}
	return lastErr
}

// computeDelay 计算第 attempt 次重试的等待时间。
// 公式：min(BaseDelay * 2^attempt, MaxDelay)
func computeDelay(attempt int, base, max time.Duration) time.Duration {
	d := time.Duration(float64(base) * math.Pow(2, float64(attempt)))
	if d > max {
		d = max
	}
	return d
}

// computeDelays 批量计算前 n 次重试的等待时间，用于测试。
func computeDelays(n int, base, max time.Duration) []time.Duration {
	delays := make([]time.Duration, n)
	for i := 0; i < n; i++ {
		delays[i] = computeDelay(i, base, max)
	}
	return delays
}
