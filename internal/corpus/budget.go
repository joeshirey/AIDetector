package corpus

import (
	"errors"
	"fmt"
	"sync"
)

// ErrBudget means the run's estimated spend reached its limit. Generation
// stops; the caller should ask before raising the limit and resuming.
var ErrBudget = errors.New("estimated spend reached the budget")

// Price is USD per million tokens.
type Price struct{ In, Out float64 }

// Budget estimates spend from the token usage each API response reports
// (output includes hidden reasoning tokens) and stops at Limit. Prices are
// deliberately conservative; models without a listed price are charged at
// the flagship rate.
type Budget struct {
	mu     sync.Mutex
	Limit  float64
	Spent  float64
	Calls  int
	Prices map[string]Price
}

var unknownModelPrice = Price{In: 10, Out: 50}

// Spend tracks paid API usage for the whole process. cmd/corpus sets Limit
// from -budget.
var Spend = &Budget{Limit: 20, Prices: map[string]Price{
	"gpt-6-astra": {10, 50},
	"gpt-6.1-sol": {2, 10},
	"gpt-6-luna":  {0.10, 0.50},
	"gpt-5.6-sol": {5, 30}, // upper end of the published range
	"gpt-5-mini":  {0.25, 2},
	"gpt-5-nano":  {0.05, 0.40},
}}

func (b *Budget) price(model string) Price {
	if p, ok := b.Prices[model]; ok {
		return p
	}
	return unknownModelPrice
}

// Check returns ErrBudget once the limit has been reached.
func (b *Budget) Check() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Limit > 0 && b.Spent >= b.Limit {
		return fmt.Errorf("%w ($%.2f of $%.2f)", ErrBudget, b.Spent, b.Limit)
	}
	return nil
}

// Add records one call's usage.
func (b *Budget) Add(model string, in, out int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	p := b.price(model)
	b.Spent += (float64(in)*p.In + float64(out)*p.Out) / 1e6
	b.Calls++
}

func (b *Budget) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return fmt.Sprintf("estimated spend $%.2f of $%.2f budget (%d paid calls)", b.Spent, b.Limit, b.Calls)
}
