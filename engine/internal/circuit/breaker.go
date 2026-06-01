package circuit

import (
	"errors"
	"sync"
	"time"
)

// ErrCircuitOpen 在断路器打开（熔断中）时返回，表示请求被拒绝。
var ErrCircuitOpen = errors.New("circuit breaker is open")

// State 表示断路器的三态。
type State int

const (
	StateClosed   State = iota // 正常通行
	StateOpen                  // 熔断中，拒绝请求
	StateHalfOpen              // 探测中，允许一个请求通过
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// Config 是断路器的配置。
type Config struct {
	FailureThreshold int           // 连续失败多少次后跳闸
	RecoveryTimeout  time.Duration // 跳闸多久后进入半开探测
}

// Breaker 实现三态熔断器模式。
//
// 状态转换：
//   Closed ── failures >= threshold ──▶ Open
//   Open   ── recovery timeout ──────▶ HalfOpen
//   HalfOpen ── 成功 ────────────────▶ Closed
//   HalfOpen ── 失败 ────────────────▶ Open
type Breaker struct {
	config       Config
	state        State
	mu           sync.Mutex
	failures     int
	lastFailTime time.Time
}

// NewBreaker 创建一个新的断路器。
func NewBreaker(cfg Config) *Breaker {
	return &Breaker{
		config: cfg,
		state:  StateClosed,
	}
}

// State 返回当前状态（线程安全）。
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.maybeTransition()
	return b.state
}

// Execute 执行给定的函数，根据断路器状态决定是否放行。
//
//   - Closed: 正常执行；执行失败则累加失败计数
//   - Open: 直接返回 ErrCircuitOpen（fail fast）
//   - HalfOpen: 允许执行；成功→Closed，失败→Open
func (b *Breaker) Execute(fn func() error) error {
	b.mu.Lock()
	b.maybeTransition()

	if b.state == StateOpen {
		b.mu.Unlock()
		return ErrCircuitOpen
	}
	// HalfOpen 或 Closed → 允许调用
	b.mu.Unlock()

	err := fn()

	b.mu.Lock()
	defer b.mu.Unlock()

	if err != nil {
		b.failures++
		b.lastFailTime = time.Now()
		if b.failures >= b.config.FailureThreshold {
			b.state = StateOpen
		}
	} else {
		// 成功 → 重置
		b.failures = 0
		b.state = StateClosed
	}

	return err
}

// maybeTransition 检查是否需要从 Open 进入 HalfOpen。
// 必须在持有锁的情况下调用。
func (b *Breaker) maybeTransition() {
	if b.state == StateOpen && time.Since(b.lastFailTime) >= b.config.RecoveryTimeout {
		b.state = StateHalfOpen
	}
}
