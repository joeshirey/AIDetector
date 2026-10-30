// Package calibrate fits and applies a logistic regression over detector
// signal values, replacing hand-set weights with ones learned from a corpus.
package calibrate

import (
	"math"
	"slices"
)

// Row is one labeled training example.
type Row struct {
	Feats  map[string]float64 // signal name -> raw value
	Words  int
	IsAI   bool
	Weight float64 // sample weight; 0 means 1
}

func totalWeight(rows []Row) float64 {
	n := 0.0
	for _, r := range rows {
		if r.Weight > 0 {
			n += r.Weight
		} else {
			n++
		}
	}
	return n
}

// Model is a standardized logistic regression. Signals absent from a text
// contribute their Missing weight instead.
type Model struct {
	Kind     string             `json:"kind"`
	Bias     float64            `json:"bias"`
	Features []Feature          `json:"features"`
	Missing  map[string]float64 `json:"missing"`
	Trained  int                `json:"trained_on"`
}

type Feature struct {
	Name   string  `json:"name"`
	Mean   float64 `json:"mean"`
	Std    float64 `json:"std"`
	Weight float64 `json:"weight"`
	// Typical values in the training corpus, shown to users as reference.
	HumanMedian float64 `json:"human_median"`
	AIMedian    float64 `json:"ai_median"`
}

// Contribution is one feature's share of a score, in log-odds.
type Contribution struct {
	Feature
	Value   float64
	Present bool
	Logit   float64
}

// Contributions breaks Prob's log-odds down by feature.
func (m *Model) Contributions(feats map[string]float64, words int) []Contribution {
	out := make([]Contribution, 0, len(m.Features))
	for _, f := range m.Features {
		c := Contribution{Feature: f}
		c.Value, c.Present = feats[f.Name]
		if f.Name == wordsFeature {
			c.Value, c.Present = float64(words), true
			c.Logit = f.Weight * (math.Log1p(float64(words)) - f.Mean) / f.Std
		} else if c.Present {
			c.Logit = f.Weight * (c.Value - f.Mean) / f.Std
		} else {
			c.Logit = m.Missing[f.Name]
		}
		out = append(out, c)
	}
	return out
}

const wordsFeature = "log_words"

// Prob returns P(AI) for a text's signal values.
func (m *Model) Prob(feats map[string]float64, words int) float64 {
	z := m.Bias
	for _, f := range m.Features {
		v, ok := feats[f.Name]
		if f.Name == wordsFeature {
			v, ok = math.Log1p(float64(words)), true
		}
		if ok {
			z += f.Weight * (v - f.Mean) / f.Std
		} else {
			z += m.Missing[f.Name]
		}
	}
	return 1 / (1 + math.Exp(-z))
}

// Fit trains a model with L2-regularized gradient descent.
func Fit(kind string, rows []Row) *Model {
	names := map[string]bool{wordsFeature: true}
	for _, r := range rows {
		for n := range r.Feats {
			names[n] = true
		}
	}
	m := &Model{Kind: kind, Missing: map[string]float64{}, Trained: len(rows)}
	for n := range names {
		m.Features = append(m.Features, Feature{Name: n})
	}
	slices.SortFunc(m.Features, func(a, b Feature) int {
		if a.Name < b.Name {
			return -1
		}
		return 1
	})

	// Design matrix: one standardized column per feature, plus a
	// "missing" indicator column per feature.
	k := len(m.Features)
	x := make([][]float64, len(rows))
	for j := range m.Features {
		f := &m.Features[j]
		var sum, sq float64
		n := 0
		for _, r := range rows {
			if v, ok := value(r, f.Name); ok {
				sum += v
				sq += v * v
				n++
			}
		}
		if n > 0 {
			f.Mean = sum / float64(n)
			f.Std = math.Sqrt(max(sq/float64(n)-f.Mean*f.Mean, 0))
		}
		if f.Std < 1e-9 {
			f.Std = 1
		}
		var human, ai []float64
		for _, r := range rows {
			if v, ok := r.Feats[f.Name]; ok || f.Name == wordsFeature {
				if f.Name == wordsFeature {
					v = float64(r.Words)
				}
				if r.IsAI {
					ai = append(ai, v)
				} else {
					human = append(human, v)
				}
			}
		}
		f.HumanMedian, f.AIMedian = median(human), median(ai)
	}
	for i, r := range rows {
		x[i] = make([]float64, 2*k)
		for j, f := range m.Features {
			if v, ok := value(r, f.Name); ok {
				x[i][j] = (v - f.Mean) / f.Std
			} else {
				x[i][k+j] = 1
			}
		}
	}

	w := make([]float64, 2*k)
	var b float64
	const lr, l2, iters = 0.1, 0.01, 3000
	grad := make([]float64, 2*k)
	for range iters {
		clear(grad)
		var gb float64
		for i, r := range rows {
			z := b
			for j, xv := range x[i] {
				z += w[j] * xv
			}
			err := 1/(1+math.Exp(-z)) - label(r)
			if r.Weight > 0 {
				err *= r.Weight
			}
			for j, xv := range x[i] {
				grad[j] += err * xv
			}
			gb += err
		}
		n := totalWeight(rows)
		for j := range w {
			w[j] -= lr * (grad[j]/n + l2*w[j])
		}
		b -= lr * gb / n
	}
	m.Bias = b
	for j := range m.Features {
		m.Features[j].Weight = w[j]
		m.Missing[m.Features[j].Name] = w[k+j]
	}
	return m
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := slices.Clone(xs)
	slices.Sort(s)
	return s[len(s)/2]
}

func value(r Row, name string) (float64, bool) {
	if name == wordsFeature {
		return math.Log1p(float64(r.Words)), true
	}
	v, ok := r.Feats[name]
	return v, ok
}

func label(r Row) float64 {
	if r.IsAI {
		return 1
	}
	return 0
}
