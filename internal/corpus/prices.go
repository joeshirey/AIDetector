package corpus

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// PriceTable holds dated per-model prices (USD per million tokens at the
// online, global-endpoint rate) plus the batch multiplier. The default is
// embedded; -prices loads an override file in the same format.
type PriceTable struct {
	Updated string `json:"updated"`
	// BatchMultiplier applies to Vertex batch prediction: 50% off the
	// regional rate, and batch jobs run in regions that cost ~10% more
	// than the global endpoint, so 0.55 of the global online price.
	BatchMultiplier float64 `json:"batch_multiplier"`
	// BatchByPrefix overrides BatchMultiplier by model-name prefix. Claude
	// batches run regionally (us-east5, +10%: 0.55); Gemini batches run on
	// the global endpoint (0.50).
	BatchByPrefix map[string]float64      `json:"batch_by_prefix,omitempty"`
	Models        map[string][]PricePoint `json:"models"`
	Unknown       PricePoint              `json:"unknown"` // charged for models not listed: deliberately high
}

type PricePoint struct {
	From    string  `json:"from"` // first night (YYYY-MM-DD) this price applies
	In      float64 `json:"in"`
	Out     float64 `json:"out"` // also charged for thinking tokens
	CacheIn float64 `json:"cache_in,omitempty"`
	Note    string  `json:"note,omitempty"`
}

//go:embed prices.json
var defaultPrices []byte

func DefaultPriceTable() *PriceTable {
	var t PriceTable
	if err := json.Unmarshal(defaultPrices, &t); err != nil {
		panic(fmt.Sprintf("embedded prices.json: %v", err))
	}
	return &t
}

func LoadPriceTable(path string) (*PriceTable, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var t PriceTable
	return &t, json.Unmarshal(b, &t)
}

// At returns the price for model on the given night.
func (t *PriceTable) At(model string, night time.Time) PricePoint {
	pts := t.Models[model]
	sort.Slice(pts, func(i, j int) bool { return pts[i].From < pts[j].From })
	day := night.Format("2006-01-02")
	var best *PricePoint
	for i := range pts {
		if pts[i].From <= day {
			best = &pts[i]
		}
	}
	if best == nil {
		return t.Unknown
	}
	return *best
}

// Batch is the batch multiplier for a model.
func (t *PriceTable) Batch(model string) float64 {
	for p, m := range t.BatchByPrefix {
		if strings.HasPrefix(model, p) {
			return m
		}
	}
	return t.BatchMultiplier
}
