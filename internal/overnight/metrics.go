package overnight

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/eval"
	"github.com/joeshirey/AIDetector/internal/fixtures"
	"github.com/joeshirey/AIDetector/internal/split"
	"github.com/joeshirey/AIDetector/internal/train"
)

// AR is AUROC plus AI recall at <=1% false positives.
type AR struct {
	AUROC   float64 `json:"auroc"`
	Recall1 float64 `json:"recall1"`
	Human   int     `json:"human"`
	AI      int     `json:"ai"`
}

// Metrics is everything the gate compares.
type Metrics struct {
	Test     map[string]AR         `json:"test"`     // kind -> test split
	PoolDoc  map[string][2]float64 `json:"pool_doc"` // pool -> share of human docs at 60+, 80+
	PoolSeg  map[string]float64    `json:"pool_seg"` // pool -> share of human segments at 60+
	PoolN    map[string][2]int     `json:"pool_n"`   // pool -> {documents, segments} scored
	Fixtures fixtures.CheckResult  `json:"fixtures"`
	Frozen   map[string]AR         `json:"frozen"` // "<gen>/<kind>" -> frozen set
	Slices   map[string]AR         `json:"slices"` // held-out generator slices
}

func reportAR(r *eval.Report) AR {
	if r == nil {
		return AR{}
	}
	return AR{AUROC: r.AUROC, Recall1: r.TPRAtThresh, Human: r.Human, AI: r.AI}
}

// frozenDirs returns corpus_frozen genre dirs for a detector kind.
func (d Data) frozenDirs(k detect.Kind) []string {
	var out []string
	for genre := range FrozenGenres {
		g, _ := GenreByName(genre)
		if g.Det == k {
			out = append(out, d.Path("corpus_frozen", genre))
		}
	}
	sort.Strings(out)
	return out
}

// onlyGens keeps human files and AI files from generator dirs matching match.
func onlyGens(match func(gen string) bool) eval.Keep {
	sep := string(filepath.Separator)
	return func(path, id string) bool {
		if strings.Contains(path, sep+"human"+sep) {
			return true
		}
		parts := strings.Split(path, sep)
		for i := range parts {
			if parts[i] == "ai" && i+1 < len(parts) {
				return match(parts[i+1])
			}
		}
		return false
	}
}

// Measure computes the gate metrics for an engine. It never reads the
// final holdout.
func Measure(ctx context.Context, e *engine.Engine, d Data, fixList *fixtures.List, fixDir string) (*Metrics, error) {
	m := &Metrics{Test: map[string]AR{}, PoolDoc: map[string][2]float64{}, PoolSeg: map[string]float64{}, PoolN: map[string][2]int{},
		Frozen: map[string]AR{}, Slices: map[string]AR{}}
	for _, k := range []detect.Kind{detect.KindProse, detect.KindPR} {
		r, err := eval.RunFiltered(ctx, e, d.TrainDirs(k), k, "test", nil)
		if err != nil {
			return nil, fmt.Errorf("test split %s: %w", k, err)
		}
		m.Test[string(k)] = reportAR(r)
		// Held-out generator slices inside the test split.
		for _, gen := range []string{"gemini-3.8-flash", "qwen3.6-35b-a3b"} {
			if sub := r.Subset(onlyGens(func(g string) bool { return g == gen })); sub.AI > 0 {
				m.Slices[gen+"/"+string(k)] = reportAR(sub)
			}
		}
		if sub := r.Subset(onlyGens(func(g string) bool { return strings.HasPrefix(g, "gpt-") })); sub.AI > 0 {
			m.Frozen["gpt/"+string(k)] = reportAR(sub)
		}
		// Frozen generator sets (corpus_frozen), each against the same human sources.
		fr, err := eval.RunFiltered(ctx, e, d.frozenDirs(k), k, "", nil)
		if err != nil {
			continue // not generated yet
		}
		for _, spec := range append(slices.Clone(FrozenGenerators), "lmstudio:"+GemmaModel) {
			gen := strings.SplitN(spec, ":", 2)[1]
			sub := fr.Subset(onlyGens(func(g string) bool { return g == gen }))
			if sub.AI == 0 {
				continue
			}
			key := gen + "/" + string(k)
			if strings.HasPrefix(gen, "gemma") {
				m.Slices[key] = reportAR(sub)
			} else {
				m.Frozen[key] = reportAR(sub)
			}
		}
	}
	noHoldout := func(path, id string) bool { return !FinalHoldout(id) }
	for name, dir := range d.FreshPools() {
		k := detect.KindProse
		if name == "fresh-pr" {
			k = detect.KindPR
		}
		all, err := train.ScoreFiles(ctx, e, []string{dir}, k, noHoldout)
		if err != nil {
			return nil, err
		}
		var n60, n80, segN, seg60 int
		for _, s := range all {
			if s.Score >= 60 {
				n60++
			}
			if s.Score >= 80 {
				n80++
			}
			for _, v := range s.Segs {
				segN++
				if v >= 60 {
					seg60++
				}
			}
		}
		if len(all) > 0 {
			m.PoolDoc[name] = [2]float64{float64(n60) / float64(len(all)), float64(n80) / float64(len(all))}
		}
		if segN > 0 {
			m.PoolSeg[name] = float64(seg60) / float64(segN)
		}
		m.PoolN[name] = [2]int{len(all), segN}
	}
	if fixList != nil {
		res, err := fixtures.Check(fixList, fixDir, func(text string) (int, int, string, error) {
			r, err := e.Analyze(ctx, detect.Input{Text: text, Kind: detect.KindProse})
			if err != nil {
				return 0, 0, "", err
			}
			maxSeg := -1
			for _, s := range r.Segments {
				maxSeg = max(maxSeg, s.Score)
			}
			return r.Score, maxSeg, r.Label, nil
		})
		if err != nil {
			return nil, err
		}
		m.Fixtures = res
	}
	return m, nil
}

// Check is one gate comparison.
type Check struct {
	Name         string  `json:"name"`
	Before       float64 `json:"before"`
	After        float64 `json:"after"`
	Tolerance    float64 `json:"tolerance"`
	HigherBetter bool    `json:"higher_better"`
	Pass         bool    `json:"pass"`
}

// Gate compares a candidate with production. Every check must pass.
func Gate(prod, cand *Metrics) ([]Check, bool) {
	var checks []Check
	add := func(name string, before, after, tol float64, higherBetter bool) {
		pass := after >= before-tol
		if !higherBetter {
			pass = after <= before+tol
		}
		checks = append(checks, Check{Name: name, Before: before, After: after, Tolerance: tol, HigherBetter: higherBetter, Pass: pass})
	}
	for _, k := range sortedKeys(prod.Test) {
		add("test AUROC "+k, prod.Test[k].AUROC, cand.Test[k].AUROC, 0.002, true)
		add("test recall@1%FP "+k, prod.Test[k].Recall1, cand.Test[k].Recall1, 0.01, true)
	}
	// False-positive rates may not rise by more than one text's worth
	// (Joe, 2026-10-28): a refit on the same data moves a document or a
	// segment or two across a cutoff by chance.
	oneOf := func(n int) float64 {
		if n == 0 {
			return 0
		}
		return 1/float64(n) + 1e-9
	}
	for _, p := range sortedKeys(prod.PoolDoc) {
		tol := oneOf(prod.PoolN[p][0])
		add("doc FPR 60+ "+p, prod.PoolDoc[p][0], cand.PoolDoc[p][0], tol, false)
		add("doc FPR 80+ "+p, prod.PoolDoc[p][1], cand.PoolDoc[p][1], tol, false)
	}
	for _, p := range sortedKeys(prod.PoolSeg) {
		add("segment FPR 60+ "+p, prod.PoolSeg[p], cand.PoolSeg[p], oneOf(prod.PoolN[p][1]), false)
	}
	fx := 0.0
	if cand.Fixtures.Pass() {
		fx = 1
	}
	checks = append(checks, Check{Name: "regression fixtures pass", Before: 1, After: fx, HigherBetter: true, Pass: cand.Fixtures.Pass()})
	for _, k := range sortedKeys(prod.Frozen) {
		add("frozen AUROC "+k, prod.Frozen[k].AUROC, cand.Frozen[k].AUROC, 0.01, true)
		add("frozen recall@1%FP "+k, prod.Frozen[k].Recall1, cand.Frozen[k].Recall1, 0.01, true)
	}
	for _, k := range sortedKeys(prod.Slices) {
		add("slice AUROC "+k, prod.Slices[k].AUROC, cand.Slices[k].AUROC, 0.005, true)
	}
	ok := true
	for _, c := range checks {
		ok = ok && c.Pass
	}
	return checks, ok
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// RetrainCandidate fits prose and PR models into dir (excluding frozen and
// test-split samples, which Fit's train split and the data layout already
// guarantee) and calibrates cutoffs the same way as a manual refit:
// document cutoffs on the original fresh pools, segment cutoffs on the
// wider pool set, both excluding the final holdout.
// With keepCutoffs, the candidate keeps production's label cutoffs (a
// diagnostic: it separates model changes from cutoff changes).
func RetrainCandidate(ctx context.Context, d Data, dir string, exclude eval.Keep, keepCutoffs bool) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// Start from production cutoffs so untouched keys carry over.
	prodThresholds, err := engineFile("thresholds.json")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "thresholds.json"), prodThresholds, 0o644); err != nil {
		return err
	}
	trainable := func(path, id string) bool { return split.Of(id) == "train" && (exclude == nil || !exclude(path, id)) }
	for _, k := range []detect.Kind{detect.KindProse, detect.KindPR} {
		if _, err := train.Fit(ctx, k, d.TrainDirs(k), dir, false, trainable); err != nil {
			return fmt.Errorf("fit %s: %w", k, err)
		}
	}
	if keepCutoffs {
		return nil
	}
	cand, err := engine.FromDir(dir)
	if err != nil {
		return err
	}
	noHoldout := func(path, id string) bool { return !FinalHoldout(id) }
	out := filepath.Join(dir, "thresholds.json")
	o := train.DefaultThresholdOptions()
	o.Only = "documents"
	if _, err := train.Thresholds(ctx, cand, detect.KindPR, []string{d.Path("corpus_fresh", "pr", "pool")}, out, o, noHoldout); err != nil {
		return err
	}
	docPools := []string{d.Path("corpus_fresh", "blog", "pool"), d.Path("corpus_fresh", "memo", "pool"), d.Path("corpus_fresh", "essay", "pool")}
	if _, err := train.Thresholds(ctx, cand, detect.KindProse, docPools, out, o, noHoldout); err != nil {
		return err
	}
	// Segment cutoffs are matched to the new document cutoffs, so reload.
	if cand, err = engine.FromDir(dir); err != nil {
		return err
	}
	o.Only = "segments"
	segPools := append(docPools, d.Path("corpus_tech", "blog", "pool"), d.Path("corpus_nontech", "blog", "pool"))
	_, err = train.Thresholds(ctx, cand, detect.KindProse, segPools, out, o, noHoldout)
	return err
}

func engineFile(name string) ([]byte, error) { return engine.EmbeddedFile(name) }

// GemmaModel names the Gemma generator (its corpus directory). The 31B ran
// under 15 tok/s alongside Qwen, so the loop uses the 26B A4B (8-bit MLX),
// served by mlx_lm.server.
var GemmaModel = "gemma-4-26b-a4b-it-8bit"
