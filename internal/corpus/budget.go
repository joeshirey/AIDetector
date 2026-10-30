package corpus

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// ErrBudget means the night's estimated spend (or one model's share of it)
// reached its cap. Paid generation stops; nothing raises a cap on its own.
var ErrBudget = errors.New("estimated spend reached the budget")

// Usage is what one call consumed. Thinking tokens are billed as output.
// Claude reports thinking inside Output; Gemini reports it separately.
type Usage struct {
	Input     int     `json:"input"`
	Output    int     `json:"output"`
	Thinking  int     `json:"thinking,omitempty"`
	CacheRead int     `json:"cache_read,omitempty"`
	Seconds   float64 `json:"seconds,omitempty"`
}

// Mode is how a call was made, which decides its price.
const (
	ModeOnline = "online"
	ModeBatch  = "batch"
	ModeLocal  = "local" // free; logged for throughput
)

// Entry is one ledger line.
type Entry struct {
	Time      string `json:"time"`
	Night     string `json:"night"`
	Provider  string `json:"provider"`
	Model     string `json:"model"`
	Mode      string `json:"mode"`
	Purpose   string `json:"purpose,omitempty"`
	JobID     string `json:"job_id,omitempty"`
	Estimated bool   `json:"estimated,omitempty"` // batch cost reserved at submission
	N         int    `json:"n,omitempty"`         // requests the entry covers (batch corrections), if more than one
	Usage
	Cost float64 `json:"cost"`
}

// Budget prices every call from the dated price table, appends it to the
// ledger, and refuses new paid work once the night's total or a model's cap
// is reached. Batch jobs are charged (as an estimate) when submitted and
// corrected when collected, both against the night they were submitted.
type Budget struct {
	mu        sync.Mutex
	Limit     float64            // total cap for the night, USD
	ModelCaps map[string]float64 // optional per-model caps, USD
	Prices    *PriceTable
	Night     time.Time
	Purpose   string // default purpose label for ledger lines
	ledger    *os.File

	Spent   float64
	ByModel map[string]float64
	Calls   int
	local   map[string]int // output tokens from local models, for throughput
}

// Spend is the process-wide budget. Commands set its limit, caps, night,
// and ledger file.
var Spend = NewBudget(20)

func NewBudget(limit float64) *Budget {
	return &Budget{Limit: limit, Prices: DefaultPriceTable(), Night: NightOf(time.Now()), ByModel: map[string]float64{}}
}

// NightOf maps a time to the night it belongs to: before noon counts as
// the previous evening's night.
func NightOf(t time.Time) time.Time {
	if t.Hour() < 12 {
		t = t.AddDate(0, 0, -1)
	}
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
}

// OpenLedger appends to path and restores this night's spend from it, so
// caps hold across restarts.
func (b *Budget) OpenLedger(path string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if f, err := os.Open(path); err == nil {
		night := b.Night.Format("2006-01-02")
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var e Entry
			if json.Unmarshal(sc.Bytes(), &e) == nil && e.Night == night {
				b.Spent += e.Cost
				b.ByModel[e.Model] += e.Cost
				if e.Mode != ModeLocal && !strings.HasSuffix(e.Purpose, "correction") {
					b.Calls++
				}
			}
		}
		f.Close()
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	b.ledger = f
	return nil
}

// Cost prices a call without recording it.
func (b *Budget) Cost(model, mode string, u Usage) float64 {
	if mode == ModeLocal {
		return 0
	}
	p := b.Prices.At(model, b.Night)
	cacheIn := p.CacheIn
	if cacheIn == 0 {
		cacheIn = p.In
	}
	fresh := max(u.Input-u.CacheRead, 0)
	c := (float64(fresh)*p.In + float64(u.CacheRead)*cacheIn + float64(u.Output+u.Thinking)*p.Out) / 1e6
	if mode == ModeBatch {
		c *= b.Prices.Batch(model)
	}
	return c
}

// Check returns ErrBudget once the night's total, or model's cap, is reached.
func (b *Budget) Check(model string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Limit > 0 && b.Spent >= b.Limit {
		return fmt.Errorf("%w ($%.2f of $%.2f)", ErrBudget, b.Spent, b.Limit)
	}
	if c, ok := b.ModelCaps[model]; ok && b.ByModel[model] >= c {
		return fmt.Errorf("%w for %s ($%.2f of its $%.2f cap)", ErrBudget, model, b.ByModel[model], c)
	}
	return nil
}

// CheckCost is Check for a known upcoming cost: it fails if spending est
// more would cross the night's total or the model's cap.
func (b *Budget) CheckCost(model string, est float64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Limit > 0 && b.Spent+est > b.Limit {
		return fmt.Errorf("%w ($%.2f spent + $%.2f would pass $%.2f)", ErrBudget, b.Spent, est, b.Limit)
	}
	if c, ok := b.ModelCaps[model]; ok && b.ByModel[model]+est > c {
		return fmt.Errorf("%w for %s ($%.2f + $%.2f would pass its $%.2f cap)", ErrBudget, model, b.ByModel[model], est, c)
	}
	return nil
}

// Headroom is how much more can be spent on model tonight.
func (b *Budget) Headroom(model string) float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	h := b.Limit - b.Spent
	if c, ok := b.ModelCaps[model]; ok {
		h = min(h, c-b.ByModel[model])
	}
	return max(h, 0)
}

// Record prices a call, adds it to the night's spend, and writes it to the
// ledger. It returns the cost.
func (b *Budget) Record(e Entry) float64 {
	// Corrections carry their cost explicitly (their usage is informational).
	if e.Cost == 0 && !strings.HasSuffix(e.Purpose, "correction") {
		e.Cost = b.Cost(e.Model, e.Mode, e.Usage)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if e.Night == "" {
		e.Night = b.Night.Format("2006-01-02")
	}
	if e.Purpose == "" {
		e.Purpose = b.Purpose
	}
	e.Time = time.Now().UTC().Format(time.RFC3339)
	if e.Night == b.Night.Format("2006-01-02") {
		b.Spent += e.Cost
		b.ByModel[e.Model] += e.Cost
	}
	if e.Mode != ModeLocal && !strings.HasSuffix(e.Purpose, "correction") {
		b.Calls++
	}
	if e.Mode == ModeLocal {
		if b.local == nil {
			b.local = map[string]int{}
		}
		b.local[e.Model] += e.Output + e.Thinking
	}
	if b.ledger != nil {
		line, _ := json.Marshal(e)
		b.ledger.Write(append(line, '\n'))
	}
	return e.Cost
}

// LocalOutput is the output tokens recorded for a local model so far.
func (b *Budget) LocalOutput(model string) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.local[model]
}

func (b *Budget) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return fmt.Sprintf("estimated spend $%.2f of $%.2f budget (%d paid calls)", b.Spent, b.Limit, b.Calls)
}

// Breakdown returns spend by model, largest first.
func (b *Budget) Breakdown() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []string
	for m, c := range b.ByModel {
		if c > 0 {
			out = append(out, fmt.Sprintf("%s $%.2f", m, c))
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
	return out
}
