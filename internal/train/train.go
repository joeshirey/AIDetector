// Package train fits the detector's models and calibrates label cutoffs.
// The aidetector CLI and the overnight loop both use it; the loop passes a
// filter so frozen sets and the final holdout never take part.
package train

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/ngram"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/eval"
	"github.com/joeshirey/AIDetector/internal/split"
)

// FitResult describes a fit.
type FitResult struct {
	Model     *calibrate.Model
	Words     *ngram.Model
	Human, AI int
	ModelPath string
	NgramPath string
}

// Fit trains kind's word-pattern and calibration models on the train split
// of dirs (samples keep accepts) and writes them to outDir. Features come
// from the heuristic engine, so fitting never depends on a previous model.
func Fit(ctx context.Context, kind detect.Kind, dirs []string, outDir string, withLM bool, keep eval.Keep) (*FitResult, error) {
	he := engine.Heuristic()
	if withLM && he.Scorer() == nil {
		return nil, fmt.Errorf("-lm needs a running scorer at $AIDETECTOR_SCORER_URL")
	}
	r, err := eval.RunFiltered(ctx, he, dirs, kind, "train", keep)
	if err != nil {
		return nil, err
	}
	if !withLM {
		for _, s := range r.Samples {
			for n := range s.Feats {
				if strings.HasPrefix(n, "lm_") {
					delete(s.Feats, n)
				}
			}
		}
	}
	docs := make([]string, len(r.Samples))
	labels := make([]bool, len(r.Samples))
	for i, s := range r.Samples {
		b, err := os.ReadFile(s.Path)
		if err != nil {
			return nil, err
		}
		docs[i], labels[i] = string(b), s.IsAI
	}
	// Word-pattern scores for the calibration step come from 5-fold
	// out-of-fold models, so the calibrator sees how the classifier does on
	// text it wasn't trained on rather than its overconfident training fit.
	const folds = 5
	oof := make([]float64, len(docs))
	for f := range folds {
		var trDocs []string
		var trLabels []bool
		for i, s := range r.Samples {
			if Fold(s.ID, folds) != f {
				trDocs, trLabels = append(trDocs, docs[i]), append(trLabels, labels[i])
			}
		}
		fm := ngram.Train(string(kind), trDocs, trLabels, ngram.DefaultTrainOptions())
		for i, s := range r.Samples {
			if Fold(s.ID, folds) == f {
				oof[i], _ = fm.Logit(docs[i])
			}
		}
	}
	words := ngram.Train(string(kind), docs, labels, ngram.DefaultTrainOptions())
	rows := make([]calibrate.Row, len(r.Samples))
	for i, s := range r.Samples {
		feats := map[string]float64{"word_patterns": oof[i]}
		for n, v := range s.Feats {
			feats[n] = v
		}
		rows[i] = calibrate.Row{Feats: feats, Words: s.Words, IsAI: s.IsAI}
	}
	m := calibrate.Fit(string(kind), rows)
	res := &FitResult{Model: m, Words: words, Human: r.Human, AI: r.AI,
		ModelPath: filepath.Join(outDir, string(kind)+".json"), NgramPath: filepath.Join(outDir, string(kind)+".ngram.json")}
	if withLM {
		res.ModelPath = filepath.Join(outDir, string(kind)+".lm.json")
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, err
	}
	if err := WriteJSON(res.ModelPath, m); err != nil {
		return nil, err
	}
	return res, WriteJSON(res.NgramPath, words)
}

// Fold assigns a sample to one of n cross-validation folds by id, so a human
// text and its AI mirror always land in the same fold.
func Fold(id string, n int) int { return int(split.Hash(id) / 5 % uint32(n)) }

func WriteJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// Scored is one human file's raw document score and raw segment scores.
type Scored struct {
	Path, ID, Label string
	Score, Raw      int
	Segs, SegRaw    []int
	SegPreview      []string
}

// ScoreFiles scores every .txt in dirs (and one level of subdirectories)
// that keep accepts.
func ScoreFiles(ctx context.Context, e *engine.Engine, dirs []string, kind detect.Kind, keep eval.Keep) ([]Scored, error) {
	var all []Scored
	for _, dir := range dirs {
		paths, err := filepath.Glob(filepath.Join(dir, "*.txt"))
		if err != nil {
			return nil, err
		}
		sub, _ := filepath.Glob(filepath.Join(dir, "*", "*.txt"))
		for _, p := range append(paths, sub...) {
			id := strings.TrimSuffix(filepath.Base(p), ".txt")
			if keep != nil && !keep(p, id) {
				continue
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return nil, err
			}
			res, err := e.Analyze(ctx, detect.Input{Text: string(b), Kind: kind})
			if err != nil {
				return nil, err
			}
			sc := Scored{Path: p, ID: id, Label: res.Label, Score: res.Score, Raw: res.RawScore}
			for _, s := range res.Segments {
				sc.Segs = append(sc.Segs, s.Score)
				sc.SegRaw = append(sc.SegRaw, s.RawScore)
				sc.SegPreview = append(sc.SegPreview, s.Preview)
			}
			all = append(all, sc)
		}
	}
	return all, nil
}

// ThresholdOptions are the target shares of human text at each label and
// which cutoffs to calibrate ("documents", "segments", or "both").
type ThresholdOptions struct {
	Mixed, Likely, AI float64
	Only              string
}

func DefaultThresholdOptions() ThresholdOptions {
	return ThresholdOptions{Mixed: 0.05, Likely: 0.02, AI: 0.005, Only: "both"}
}

// ThresholdReport is what Thresholds calibrated.
type ThresholdReport struct {
	Key                    string
	Doc, Segment           *engine.Thresholds
	DocChecked, SegChecked int
}

// Thresholds calibrates kind's cutoffs on human-only text in dirs and
// updates the thresholds file at out. Half the texts (by id hash) set the
// cutoffs; the other half measures how often human text actually reaches
// each label. Segments are matched to the documents they came from.
// FitCutoffs picks the lowest raw cutoffs at which at most the given shares
// of the calibration half score at or above each label, and reports the
// shares on the check half.
func FitCutoffs(calib, check []int, mixed, likely, ai float64) engine.Thresholds {
	cutoff := func(allowed float64, floor int) int {
		for t := floor; t <= 100; t++ {
			if Share(calib, t) <= allowed {
				return t
			}
		}
		return 100
	}
	t := engine.Thresholds{}
	t.Mixed = cutoff(mixed, 1)
	t.LikelyAI = cutoff(likely, t.Mixed+1)
	t.AI = cutoff(ai, t.LikelyAI+1)
	t.HumanAtMixed = Share(check, t.Mixed)
	t.HumanAtLikelyAI = Share(check, t.LikelyAI)
	t.HumanAtAI = Share(check, t.AI)
	t.CalibratedOn = len(calib)
	return t
}

// Share is the fraction of scores at or above t.
func Share(scores []int, t int) float64 {
	if len(scores) == 0 {
		return 0
	}
	n := 0
	for _, s := range scores {
		if s >= t {
			n++
		}
	}
	return float64(n) / float64(len(scores))
}

func Thresholds(ctx context.Context, e *engine.Engine, kind detect.Kind, dirs []string, out string, o ThresholdOptions, keep eval.Keep) (*ThresholdReport, error) {
	all, err := ScoreFiles(ctx, e, dirs, kind, keep)
	if err != nil {
		return nil, err
	}
	key := e.ThresholdKey(kind)
	var calib, check, segCalib, segCheck, segDocRaw []int
	for _, s := range all {
		if split.Hash(s.ID)%2 == 0 {
			calib = append(calib, s.Raw)
			segCalib = append(segCalib, s.SegRaw...)
			if len(s.SegRaw) > 0 {
				segDocRaw = append(segDocRaw, s.Raw)
			}
		} else {
			check = append(check, s.Raw)
			segCheck = append(segCheck, s.SegRaw...)
		}
	}
	share := Share
	fit := FitCutoffs
	file := map[string]engine.Thresholds{}
	if b, err := os.ReadFile(out); err == nil {
		if err := json.Unmarshal(b, &file); err != nil {
			return nil, err
		}
	}
	rep := &ThresholdReport{Key: key, DocChecked: len(check), SegChecked: len(segCheck)}
	if o.Only != "segments" {
		t := fit(calib, check, o.Mixed, o.Likely, o.AI)
		file[key] = t
		rep.Doc = &t
	}
	if len(segCalib) > 0 && o.Only != "documents" {
		doc := file[key]
		floor := func(v float64) float64 { return max(v, 0.0025) }
		st := fit(segCalib, segCheck, floor(share(segDocRaw, doc.Mixed)), floor(share(segDocRaw, doc.LikelyAI)), floor(share(segDocRaw, doc.AI)))
		file[key+".segment"] = st
		rep.Segment = &st
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return nil, err
	}
	return rep, WriteJSON(out, file)
}
