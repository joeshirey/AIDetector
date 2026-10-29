package overnight

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
)

// frozenGenre is a frozen genre's prompt genre and its sources (in corpus_frozen).
func (d Data) frozenSources(genre string) ([]corpus.Sample, Genre, error) {
	g, _ := GenreByName(genre)
	srcs, err := (&corpus.Store{Root: d.Path("corpus_frozen")}).Sources(genre)
	sort.Slice(srcs, func(i, j int) bool { return srcs[i].ID < srcs[j].ID })
	return srcs, g, err
}

func (d Data) frozenOutput(genre, gen, id string) string {
	return d.Path("corpus_frozen", genre, "ai", gen, id+".txt")
}

// FrozenRequests builds plain-style requests for a frozen generator set:
// every frozen source, from briefs cached under briefs/frozen (never used
// for training). Sources already generated for gen are skipped.
func FrozenRequests(ctx context.Context, d Data, br *Briefer, gen corpus.Generator, rng *rand.Rand) ([]PlannedRequest, float64, error) {
	var out []PlannedRequest
	total := 0.0
	for _, genre := range sortedKeys(FrozenGenres) {
		srcs, g, err := d.frozenSources(genre)
		if err != nil {
			return nil, 0, err
		}
		for _, sm := range srcs {
			if _, err := os.Stat(d.frozenOutput(genre, gen.Name(), sm.ID)); err == nil {
				continue
			}
			brief := ""
			if needsBrief(g, StylePlain) {
				if brief, err = br.Brief(ctx, g, sm); err != nil {
					return out, total, fmt.Errorf("brief %s: %w", sm.ID, err)
				}
			}
			prompt, words, err := BuildPrompt(g, StylePlain, sm, brief, rng)
			if err != nil {
				return nil, 0, err
			}
			mode := corpus.ModeBatch
			if p := gen.Spec().Provider; p == "lmstudio" || p == "mlx" {
				mode = corpus.ModeLocal
			}
			cost := corpus.Spend.Cost(gen.Spec().Model, mode, EstimateUsage(gen.Spec(), prompt, words))
			out = append(out, PlannedRequest{Meta: ReqMeta{Genre: genre, ID: sm.ID, Style: StylePlain, Band: BandOf(g, words), Track: "frozen"},
				Prompt: prompt, Title: sm.Context["title"], Cost: cost})
			total += cost
		}
	}
	return out, total, nil
}

// RunFrozenLocal generates a frozen set with a local model (Gemma).
func RunFrozenLocal(ctx context.Context, d Data, gen corpus.Generator, reqs []PlannedRequest, workers int, log io.Writer) LocalStats {
	st := LocalStats{Gen: gen.Name()}
	var mu sync.Mutex
	ch := make(chan PlannedRequest)
	var wg sync.WaitGroup
	for range max(workers, 1) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := range ch {
				before := corpus.Spend.LocalOutput(gen.Spec().Model)
				out, err := gen.Generate(ctx, r.Prompt)
				mu.Lock()
				st.OutputTokens += corpus.Spend.LocalOutput(gen.Spec().Model) - before
				if err != nil {
					st.Failed++
					fmt.Fprintf(log, "%s %s: %v\n", gen.Name(), r.Meta.ID, err)
					mu.Unlock()
					continue
				}
				g, _ := GenreByName(r.Meta.Genre)
				_, ok, err := d.WriteOutput(g, gen.Name(), r.Meta, out, r.Title, d.Path("corpus_frozen"))
				if err != nil || !ok {
					fmt.Fprintf(log, "%s %s: not kept (%d words after cleanup, err %v)\n", gen.Name(), r.Meta.ID, len(strings.Fields(out)), err)
					st.Failed++
				} else {
					st.Written++
				}
				if (st.Written+st.Failed)%25 == 0 {
					fmt.Fprintf(log, "%s frozen: %d written, %d failed of %d\n", gen.Name(), st.Written, st.Failed, len(reqs))
				}
				mu.Unlock()
			}
		}()
	}
	for _, r := range reqs {
		if ctx.Err() != nil {
			break
		}
		ch <- r
	}
	close(ch)
	wg.Wait()
	return st
}

// Effort experiment: the same sources generated at default and low effort
// per paid model. Outputs go to runs/effort (never the training corpus).

var EffortModels = []string{"claude:claude-haiku-5-5", "gemini:gemini-3.8-flash", "claude:claude-sonnet-5-5", "gemini:gemini-4-argon"}

// EffortSources picks n train-split sources spread across genres, the same
// for every model and effort.
func EffortSources(d Data, n int) ([]Source, error) {
	var out []Source
	per := (n + len(Genres) - 1) / len(Genres)
	for _, g := range Genres {
		bands, err := d.Sources(g)
		if err != nil {
			return nil, err
		}
		var all []Source
		for _, b := range sortedInts(bands) {
			all = append(all, bands[b]...)
		}
		sort.Slice(all, func(i, j int) bool { return stableSeed(all[i].Sample.ID) < stableSeed(all[j].Sample.ID) })
		out = append(out, all[:min(per, len(all))]...)
	}
	return out[:min(n, len(out))], nil
}

func sortedInts[V any](m map[int]V) []int {
	var out []int
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

func EffortRequests(ctx context.Context, srcs []Source, br *Briefer, spec corpus.Spec) ([]PlannedRequest, float64, error) {
	rng := rand.New(rand.NewPCG(11, 11)) // same personas/prompts for both efforts
	var out []PlannedRequest
	total := 0.0
	for _, s := range srcs {
		brief := ""
		if needsBrief(s.Genre, StylePlain) {
			b, err := br.Brief(ctx, s.Genre, s.Sample)
			if err != nil {
				return nil, 0, err
			}
			brief = b
		}
		prompt, words, err := BuildPrompt(s.Genre, StylePlain, s.Sample, brief, rng)
		if err != nil {
			return nil, 0, err
		}
		cost := corpus.Spend.Cost(spec.Model, corpus.ModeBatch, EstimateUsage(spec, prompt, words))
		out = append(out, PlannedRequest{Meta: ReqMeta{Genre: s.Genre.Name, ID: s.Sample.ID, Style: StylePlain, Band: s.Band, Track: "effort"},
			Prompt: prompt, Title: s.Sample.Context["title"], Cost: cost})
		total += cost
	}
	return out, total, nil
}

// EffortResult compares default and low effort for one model.
type EffortResult struct {
	Model                string
	NDefault, NLow       int
	MeanDefault, MeanLow float64
	KS                   float64 // two-sample Kolmogorov-Smirnov statistic
	KSCritical           float64 // at alpha 0.05
	TokensDefault        corpus.Usage
	TokensLow            corpus.Usage // per sample
	CostDefault, CostLow float64      // per sample, batch
	Match                bool
}

// AnalyzeEffort scores runs/effort outputs with production and decides,
// per model, whether low effort matches default: the score distributions
// must not differ (KS below the 5% critical value) and mean scores must be
// within 3 points.
func AnalyzeEffort(ctx context.Context, d Data, jobs *Jobs) ([]EffortResult, error) {
	e := engine.Default()
	root := d.Path("runs", "effort")
	var out []EffortResult
	for _, spec := range EffortModels {
		s, _ := corpus.ParseSpec(spec)
		r := EffortResult{Model: s.Model}
		scores := map[string][]float64{}
		for _, eff := range []string{"", "low"} {
			name := s.Model
			if eff != "" {
				name += "@" + eff
			}
			for _, g := range Genres {
				files, _ := filepath.Glob(filepath.Join(root, g.Name, "ai", name, "*.txt"))
				for _, f := range files {
					res, err := e.Analyze(ctx, detect.Input{Text: readText(f), Kind: g.Det})
					if err == nil && res.Label != engine.TooShort {
						scores[eff] = append(scores[eff], float64(res.Score))
					}
				}
			}
			var u corpus.Usage
			n, cost := 0, 0.0
			for _, t := range jobs.List {
				if t.Collected && t.Job.Spec.Name() == name && t.Purpose == "effort" {
					u.Input += t.Usage.Input
					u.Output += t.Usage.Output
					u.Thinking += t.Usage.Thinking
					n += t.Written + t.Failed
					cost += t.Actual
				}
			}
			if n > 0 {
				u = corpus.Usage{Input: u.Input / n, Output: u.Output / n, Thinking: u.Thinking / n}
				cost /= float64(n)
			}
			if eff == "" {
				r.TokensDefault, r.CostDefault = u, cost
			} else {
				r.TokensLow, r.CostLow = u, cost
			}
		}
		r.NDefault, r.NLow = len(scores[""]), len(scores["low"])
		r.MeanDefault, r.MeanLow = mean(scores[""]), mean(scores["low"])
		if r.NDefault > 0 && r.NLow > 0 {
			r.KS = ksStat(scores[""], scores["low"])
			nd, nl := float64(r.NDefault), float64(r.NLow)
			r.KSCritical = 1.358 * math.Sqrt((nd+nl)/(nd*nl))
			r.Match = r.KS < r.KSCritical && math.Abs(r.MeanDefault-r.MeanLow) <= 3
		}
		out = append(out, r)
	}
	return out, nil
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func ksStat(a, b []float64) float64 {
	a, b = append([]float64(nil), a...), append([]float64(nil), b...)
	sort.Float64s(a)
	sort.Float64s(b)
	var i, j int
	d := 0.0
	for i < len(a) && j < len(b) {
		x := math.Min(a[i], b[j])
		for i < len(a) && a[i] <= x {
			i++
		}
		for j < len(b) && b[j] <= x {
			j++
		}
		d = math.Max(d, math.Abs(float64(i)/float64(len(a))-float64(j)/float64(len(b))))
	}
	return d
}
