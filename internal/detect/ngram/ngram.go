// Package ngram is a word-pattern classifier: logistic regression over word
// and punctuation unigrams and bigrams, learned from the corpus. It finds the
// phrasing differences between human and AI text that no hand-written list
// anticipates.
package ngram

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"regexp"
	"slices"
	"strings"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/text"
)

// Model maps n-grams to weights. Documents are featurized as binary
// presence vectors scaled to unit length.
type Model struct {
	Kind    string             `json:"kind"`
	Bias    float64            `json:"bias"`
	Weights map[string]float64 `json:"weights"`
}

var (
	tokenRE = regexp.MustCompile(`[\p{L}\p{N}]+(?:'[\p{L}]+)*|[.,;:!?()—–-]`)
	quotes  = strings.NewReplacer("’", "'", "‘", "'", "“", `"`, "”", `"`)
)

// Terms returns the distinct unigrams and bigrams in s. Curly quotes are
// straightened first so the model can't key on typography that depends on
// where a text was scraped from.
func Terms(s string) []string {
	toks := tokenRE.FindAllString(strings.ToLower(quotes.Replace(s)), -1)
	seen := make(map[string]bool, 2*len(toks))
	var out []string
	add := func(t string) {
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	for i, t := range toks {
		add(t)
		if i > 0 {
			add(toks[i-1] + " " + t)
		}
	}
	return out
}

// Logit returns the model's log-odds that s is AI-written, plus each
// present term's contribution.
func (m *Model) Logit(s string) (float64, map[string]float64) {
	terms := Terms(text.Normalize(m.Kind, s))
	contrib := map[string]float64{}
	var known []string
	for _, t := range terms {
		if _, ok := m.Weights[t]; ok {
			known = append(known, t)
		}
	}
	z := m.Bias
	if len(known) == 0 {
		return z, contrib
	}
	scale := 1 / math.Sqrt(float64(len(terms)))
	for _, t := range known {
		c := m.Weights[t] * scale
		contrib[t] = c
		z += c
	}
	return z, contrib
}

type TrainOptions struct {
	MinDocs  int // drop terms seen in fewer documents
	MaxTerms int // keep the most common terms
	Epochs   int
	L2       float64
	Seed     uint64
}

func DefaultTrainOptions() TrainOptions {
	return TrainOptions{MinDocs: 3, MaxTerms: 40000, Epochs: 40, L2: 1e-4, Seed: 1}
}

// Train fits a model with SGD.
func Train(kind string, docs []string, isAI []bool, o TrainOptions) *Model {
	docTerms := make([][]string, len(docs))
	df := map[string]int{}
	for i, d := range docs {
		docTerms[i] = Terms(text.Normalize(kind, d))
		for _, t := range docTerms[i] {
			df[t]++
		}
	}
	var vocab []string
	for t, n := range df {
		if n >= o.MinDocs {
			vocab = append(vocab, t)
		}
	}
	slices.SortFunc(vocab, func(a, b string) int {
		if c := cmp.Compare(df[b], df[a]); c != 0 {
			return c
		}
		return strings.Compare(a, b)
	})
	if len(vocab) > o.MaxTerms {
		vocab = vocab[:o.MaxTerms]
	}
	index := make(map[string]int, len(vocab))
	for i, t := range vocab {
		index[t] = i
	}

	// Sparse rows: feature indices plus the per-row scale. The scale uses
	// all of a document's terms, matching Logit.
	type row struct {
		idx   []int
		scale float64
		y     float64
	}
	rows := make([]row, len(docs))
	for i, terms := range docTerms {
		r := row{scale: 1 / math.Sqrt(float64(max(1, len(terms))))}
		for _, t := range terms {
			if j, ok := index[t]; ok {
				r.idx = append(r.idx, j)
			}
		}
		if isAI[i] {
			r.y = 1
		}
		rows[i] = r
	}

	w := make([]float64, len(vocab))
	var b float64
	rng := rand.New(rand.NewPCG(o.Seed, o.Seed))
	order := make([]int, len(rows))
	for i := range order {
		order[i] = i
	}
	for epoch := range o.Epochs {
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		lr := 0.5 / math.Sqrt(float64(epoch+1))
		for _, i := range order {
			r := rows[i]
			z := b
			for _, j := range r.idx {
				z += w[j] * r.scale
			}
			g := 1/(1+math.Exp(-z)) - r.y
			for _, j := range r.idx {
				w[j] -= lr * (g*r.scale + o.L2*w[j])
			}
			b -= lr * g
		}
	}

	m := &Model{Kind: kind, Bias: b, Weights: map[string]float64{}}
	for j, t := range vocab {
		if math.Abs(w[j]) >= 1e-3 {
			m.Weights[t] = math.Round(w[j]*1e4) / 1e4
		}
	}
	return m
}

// Detector wraps per-kind models as a signal source.
type Detector struct {
	models map[detect.Kind]*Model
}

func NewDetector(models map[detect.Kind]*Model) *Detector { return &Detector{models: models} }

func (*Detector) Name() string { return "ngram" }

func (d *Detector) Detect(_ context.Context, in detect.Input) ([]detect.Signal, error) {
	m := d.models[in.Kind]
	if m == nil {
		return nil, nil
	}
	z, contrib := m.Logit(in.Text)
	return []detect.Signal{{
		Name: "word_patterns", Detector: "ngram", Value: z, AIProb: 1 / (1 + math.Exp(-z)), Weight: 1.5,
		Detail: explain(contrib),
	}}, nil
}

func explain(contrib map[string]float64) string {
	type tc struct {
		t string
		c float64
	}
	var all []tc
	for t, c := range contrib {
		all = append(all, tc{t, c})
	}
	slices.SortFunc(all, func(a, b tc) int { return cmp.Compare(b.c, a.c) })
	pick := func(xs []tc, n int, sign float64) []string {
		var out []string
		for _, x := range xs {
			if len(out) == n {
				break
			}
			if x.c*sign > 0 {
				out = append(out, fmt.Sprintf("%q", x.t))
			}
		}
		return out
	}
	ai := pick(all, 5, 1)
	slices.Reverse(all)
	human := pick(all, 5, -1)
	return fmt.Sprintf("learned phrasing; AI-leaning: %s; human-leaning: %s",
		orNone(strings.Join(ai, ", ")), orNone(strings.Join(human, ", ")))
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}
