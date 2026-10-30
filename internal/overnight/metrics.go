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
	// S holds the per-sample scores behind the numbers, for the gate's
	// paired bootstrap.
	S []RankSample `json:"s,omitempty"`
}

// RankSample is one scored sample: Key identifies the file (so baseline
// and candidate pair up), Group its source (resampling unit).
type RankSample struct {
	Key   string  `json:"k"`
	Group string  `json:"g"`
	AI    bool    `json:"ai"`
	Raw   float64 `json:"r"`
}

// PoolDoc is one checking-half human document and its scores.
type PoolDoc struct {
	ID    string `json:"id"`
	Score int    `json:"s"`
	Segs  []int  `json:"segs,omitempty"`
}

// Metrics is everything the gate compares.
type Metrics struct {
	Test     map[string]AR         `json:"test"`     // kind -> test split
	PoolDoc  map[string][2]float64 `json:"pool_doc"` // pool -> share of human docs at 60+, 80+
	PoolSeg  map[string]float64    `json:"pool_seg"` // pool -> share of human segments at 60+
	PoolN    map[string][2]int     `json:"pool_n"`   // pool -> {documents, segments} scored
	PoolDocs map[string][]PoolDoc  `json:"pool_docs"`
	Fixtures fixtures.CheckResult  `json:"fixtures"`
	Frozen   map[string]AR         `json:"frozen"` // "<gen>/<kind>" -> frozen set
	Slices   map[string]AR         `json:"slices"` // held-out generator slices
}

func reportAR(r *eval.Report) AR {
	if r == nil {
		return AR{}
	}
	ar := AR{AUROC: r.AUROC, Recall1: r.Recall1, Human: r.Human, AI: r.AI}
	for _, s := range r.Samples {
		ar.S = append(ar.S, RankSample{Key: s.Path, Group: split.Base(s.ID), AI: s.IsAI, Raw: s.Raw})
	}
	return ar
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
		for i := len(parts) - 2; i >= 0; i-- { // the last "ai" component, so parent folders can't fool it
			if parts[i] == "ai" {
				return match(parts[i+1])
			}
		}
		return false
	}
}

// Measure computes the gate metrics for an engine. It never reads the
// final holdout.
func Measure(ctx context.Context, e *engine.Engine, d Data, fixList *fixtures.List, fixDir string) (*Metrics, error) {
	m := &Metrics{Test: map[string]AR{}, PoolDoc: map[string][2]float64{}, PoolSeg: map[string]float64{}, PoolN: map[string][2]int{}, PoolDocs: map[string][]PoolDoc{},
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
	// False positives are measured on the checking half only: the
	// calibration half set the cutoffs, so it would be in-sample.
	checkHalf := func(path, id string) bool { return !FinalHoldout(id) && !split.Calib(id) }
	for name, dir := range d.FreshPools() {
		k := PoolKind(name)
		all, err := train.ScoreFiles(ctx, e, []string{dir}, k, checkHalf)
		if err != nil {
			return nil, err
		}
		var n60, n80, segN, seg60 int
		for _, s := range all {
			m.PoolDocs[name] = append(m.PoolDocs[name], PoolDoc{ID: s.ID, Score: s.Score, Segs: s.Segs})
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

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// FitOptions controls a candidate fit.
type FitOptions struct {
	Exclude     eval.Keep // training files to leave out
	KeepCutoffs bool      // keep production's cutoffs (diagnostic)
	Weighting   string    // "" (every file counts once) or "source" (see train.FitOptions)
	DenseL2     bool      // experimental word-pattern regularization (see train.FitOptions)
}

// RetrainCandidate fits prose and PR models into dir from train-split
// files and recalibrates the cutoffs with CalibrateCutoffs.
func RetrainCandidate(ctx context.Context, d Data, dir string, o FitOptions) error {
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
	trainable := func(path, id string) bool {
		return split.Of(id) == "train" && (o.Exclude == nil || !o.Exclude(path, id))
	}
	for _, k := range []detect.Kind{detect.KindProse, detect.KindPR} {
		if _, err := train.FitWith(ctx, k, d.TrainDirs(k), dir, train.FitOptions{Keep: trainable, Weighting: o.Weighting, DenseL2: o.DenseL2}); err != nil {
			return fmt.Errorf("fit %s: %w", k, err)
		}
	}
	if o.KeepCutoffs {
		return nil
	}
	return CalibrateCutoffs(ctx, d, dir)
}

// ProductionBaseline copies the production models into dir and refits
// their cutoffs with CalibrateCutoffs, so candidates are compared with
// production under the same calibration data and procedure. (Production's
// shipped cutoffs were fit on every pool text, including what is now the
// checking half and the final holdout.)
func ProductionBaseline(ctx context.Context, d Data, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	names, err := engine.EmbeddedNames()
	if err != nil {
		return err
	}
	for _, n := range names {
		b, err := engineFile(n)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, n), b, 0o644); err != nil {
			return err
		}
	}
	return CalibrateCutoffs(ctx, d, dir)
}

// CalibrateCutoffs refits the label cutoffs of the models in dir the same
// way as a manual refit: document cutoffs on the original fresh pools,
// segment cutoffs on the wider pool set. Only the calibration half of each
// pool is used to fit (train.Thresholds splits with split.Calib), and the
// final holdout is left out entirely.
func CalibrateCutoffs(ctx context.Context, d Data, dir string) error {
	cand, err := engine.FromDir(dir)
	if err != nil {
		return err
	}
	noHoldout := func(path, id string) bool { return !FinalHoldout(id) }
	out := filepath.Join(dir, "thresholds.json")
	o := train.DefaultThresholdOptions()
	o.Only = "documents"
	prPools := []string{d.Path("corpus_fresh", "pr", "pool"), d.Path("corpus_prpool", "pr", "pool")}
	if _, err := train.Thresholds(ctx, cand, detect.KindPR, prPools, out, o, noHoldout); err != nil {
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
