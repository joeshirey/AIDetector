// Package eval measures detector accuracy on a labeled corpus.
//
// Corpus layout (one or more dirs):
//
//	<dir>/human/**/*.{txt,md}
//	<dir>/ai/**/*.{txt,md}
//
// A file's basename (minus extension) is its sample id, which decides its
// train/test split.
package eval

import (
	"cmp"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/split"
)

type Sample struct {
	ID    string  `json:"id"`
	Path  string  `json:"path"`
	IsAI  bool    `json:"is_ai"`
	Score int     `json:"score"`
	Raw   float64 `json:"raw"` // unrounded model probability (0-100), for ranking metrics
	Words int     `json:"words"`
	Kind  string  `json:"kind"`
	Feats Signals `json:"signals"` // raw signals, for fitting weights later
}

type Signals map[string]float64

type Report struct {
	Human, AI   int
	AUROC       float64
	TPRAt50     float64 // AI texts scoring >= 50
	FPRAt50     float64 // human texts scoring >= 50
	Threshold1  int     // lowest threshold with FPR <= 1%
	TPRAtThresh float64
	// Recall1 is AI recall at <=1% false positives on the unrounded raw
	// score, which (unlike the integer score) has no ties to cliff on.
	Recall1 float64
	Samples []Sample
}

// Run scores every sample in dirs. splitName is "train", "test", or "" for all.
func Run(ctx context.Context, e *engine.Engine, dirs []string, kind detect.Kind, splitName string) (*Report, error) {
	return RunFiltered(ctx, e, dirs, kind, splitName, nil)
}

// Keep decides whether a sample file takes part. nil keeps everything.
type Keep func(path, id string) bool

// RunFiltered is Run restricted to the samples keep accepts.
func RunFiltered(ctx context.Context, e *engine.Engine, dirs []string, kind detect.Kind, splitName string, keep Keep) (*Report, error) {
	r := &Report{}
	for _, dir := range dirs {
		if err := r.walk(ctx, e, dir, kind, splitName, keep); err != nil {
			return nil, err
		}
	}
	r.compute()
	if r.Human == 0 || r.AI == 0 {
		return r, fmt.Errorf("need both human and ai samples (found %d human, %d ai)", r.Human, r.AI)
	}
	return r, nil
}

func (r *Report) walk(ctx context.Context, e *engine.Engine, dir string, kind detect.Kind, splitName string, keep Keep) error {
	for _, label := range []string{"human", "ai"} {
		root := filepath.Join(dir, label)
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return err
			}
			ext := strings.ToLower(filepath.Ext(path))
			if ext != ".txt" && ext != ".md" {
				return nil
			}
			id := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
			if splitName != "" && split.Of(id) != splitName {
				return nil
			}
			if keep != nil && !keep(path, id) {
				return nil
			}
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			res, err := e.Analyze(ctx, detect.Input{Text: string(b), Kind: kind})
			if err != nil {
				return fmt.Errorf("%s: %w", path, err)
			}
			feats := Signals{}
			for _, s := range res.Signals {
				if s.Detector != "sanitizer" && s.Detector != "prepare" {
					feats[s.Name] = s.Value
				}
			}
			r.Samples = append(r.Samples, Sample{ID: id, Path: path, IsAI: label == "ai", Score: res.Score, Raw: res.Raw,
				Words: res.WordCount, Kind: string(res.Kind), Feats: feats})
			return nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Report) compute() {
	var human, ai []int
	var humanRaw, aiRaw []float64
	for _, s := range r.Samples {
		if s.IsAI {
			ai, aiRaw = append(ai, s.Score), append(aiRaw, s.Raw)
		} else {
			human, humanRaw = append(human, s.Score), append(humanRaw, s.Raw)
		}
	}
	r.Human, r.AI = len(human), len(ai)
	if r.Human == 0 || r.AI == 0 {
		return
	}
	// AUROC: probability a random AI sample outscores a random human one,
	// on the unrounded raw score.
	var wins float64
	for _, a := range aiRaw {
		for _, h := range humanRaw {
			switch {
			case a > h:
				wins++
			case a == h:
				wins += 0.5
			}
		}
	}
	r.AUROC = wins / float64(r.Human*r.AI)
	r.TPRAt50 = frac(ai, 50)
	r.FPRAt50 = frac(human, 50)

	r.Threshold1 = 101
	for t := 100; t >= 0; t-- {
		if frac(human, t) > 0.01 {
			break
		}
		r.Threshold1 = t
	}
	r.TPRAtThresh = frac(ai, r.Threshold1)
	// Continuous version: the cutoff sits just above the (k+1)-th highest
	// human raw score, where k is 1% of the humans (rounded down).
	slices.SortFunc(humanRaw, func(a, b float64) int { return cmp.Compare(b, a) })
	cut := humanRaw[min(len(humanRaw)*1/100, len(humanRaw)-1)]
	n := 0
	for _, a := range aiRaw {
		if a > cut {
			n++
		}
	}
	r.Recall1 = float64(n) / float64(len(aiRaw))
	slices.SortFunc(r.Samples, func(a, b Sample) int { return b.Score - a.Score })
}

// frac returns the share of scores at or above t.
func frac(scores []int, t int) float64 {
	n := 0
	for _, s := range scores {
		if s >= t {
			n++
		}
	}
	return float64(n) / float64(len(scores))
}

// Subset recomputes the report over the samples keep accepts, without
// rescoring.
func (r *Report) Subset(keep Keep) *Report {
	s := &Report{}
	for _, x := range r.Samples {
		if keep(x.Path, x.ID) {
			s.Samples = append(s.Samples, x)
		}
	}
	s.compute()
	return s
}
