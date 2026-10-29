package corpus

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testBudget(limit float64) *Budget {
	b := NewBudget(limit)
	b.Prices = &PriceTable{BatchMultiplier: 0.55, Unknown: PricePoint{In: 10, Out: 50},
		Models: map[string][]PricePoint{"m": {{From: "2026-01-01", In: 5, Out: 30}}}}
	b.Night = time.Date(2026, 10, 28, 0, 0, 0, 0, time.UTC)
	return b
}

// A fake OpenAI-compatible server reports usage; the budget stops the second
// call without sending it.
func TestBudgetStopsGeneration(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Write([]byte(`{"choices":[{"message":{"content":"Fix the parser"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":1000}}`))
	}))
	defer srv.Close()
	b := testBudget(0.01)
	g := &chatGen{baseURL: srv.URL, model: "m", budget: b}
	if _, err := g.Generate(context.Background(), "x"); err != nil {
		t.Fatal(err)
	}
	// 1000*5/1e6 + 1000*30/1e6 = $0.035, over the $0.01 limit.
	if _, err := g.Generate(context.Background(), "x"); !errors.Is(err, ErrBudget) {
		t.Fatalf("got %v, want ErrBudget", err)
	}
	if calls != 1 {
		t.Errorf("server saw %d calls, want 1", calls)
	}
}

func TestCostRules(t *testing.T) {
	b := testBudget(100)
	if c := b.Cost("unlisted", ModeOnline, Usage{Input: 1_000_000}); c != 10 {
		t.Errorf("unknown model: got %v, want flagship 10", c)
	}
	if c := b.Cost("m", ModeBatch, Usage{Output: 1_000_000}); c != 30*0.55 {
		t.Errorf("batch: got %v", c)
	}
	if c := b.Cost("m", ModeOnline, Usage{Output: 500_000, Thinking: 500_000}); c != 30 {
		t.Errorf("thinking billed as output: got %v", c)
	}
	if c := b.Cost("m", ModeLocal, Usage{Input: 1e6, Output: 1e6}); c != 0 {
		t.Errorf("local: got %v", c)
	}
}

func TestModelCapAndLedgerRestore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	b := testBudget(100)
	b.ModelCaps = map[string]float64{"m": 1}
	if err := b.OpenLedger(path); err != nil {
		t.Fatal(err)
	}
	b.Record(Entry{Provider: "p", Model: "m", Mode: ModeOnline, Usage: Usage{Output: 40_000}}) // $1.20
	if err := b.Check("m"); !errors.Is(err, ErrBudget) {
		t.Fatalf("model cap not enforced: %v", err)
	}
	if b.Check("other") != nil {
		t.Error("cap on m shouldn't block other models")
	}
	// A restart in the same night restores the spend from the ledger.
	b2 := testBudget(100)
	b2.ModelCaps = map[string]float64{"m": 1}
	if err := b2.OpenLedger(path); err != nil {
		t.Fatal(err)
	}
	if b2.Check("m") == nil {
		t.Error("restored budget forgot tonight's spend")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestPriceTableByDate(t *testing.T) {
	p := DefaultPriceTable()
	before := p.At("gemini-4-argon", time.Date(2026, 11, 30, 0, 0, 0, 0, time.UTC))
	after := p.At("gemini-4-argon", time.Date(2026, 12, 1, 0, 0, 0, 0, time.UTC))
	if before.In != 2 || after.In != 4 {
		t.Errorf("Argon intro price should end Nov 30: %v then %v", before.In, after.In)
	}
}
