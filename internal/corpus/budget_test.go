package corpus

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

// A fake OpenAI-compatible server reports usage; the budget stops the second
// call without sending it.
func TestBudgetStopsGeneration(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Write([]byte(`{"choices":[{"message":{"content":"Fix the parser"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":1000}}`))
	}))
	defer srv.Close()

	b := &Budget{Limit: 0.01, Prices: map[string]Price{"m": {In: 5, Out: 30}}}
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

func TestUnknownModelChargedAtFlagshipRate(t *testing.T) {
	b := &Budget{Limit: 100, Prices: map[string]Price{}}
	b.Add("some-new-model", 1_000_000, 0)
	if b.Spent != unknownModelPrice.In {
		t.Errorf("spent %v, want %v", b.Spent, unknownModelPrice.In)
	}
}
