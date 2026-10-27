// Package engine runs detectors over text and combines their signals into a
// single 0–100 score.
package engine

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/devtext"
	"github.com/joeshirey/AIDetector/internal/detect/ngram"
	"github.com/joeshirey/AIDetector/internal/detect/signs"
	"github.com/joeshirey/AIDetector/internal/detect/stylometric"
	"github.com/joeshirey/AIDetector/internal/text"
)

// Result is what the CLI and MCP tools return.
type Result struct {
	Score      int             `json:"score" jsonschema:"0-100, higher means more likely AI-written"`
	Label      string          `json:"label"`
	Confidence string          `json:"confidence" jsonschema:"low, medium, or high; mostly driven by text length"`
	Kind       detect.Kind     `json:"kind"`
	WordCount  int             `json:"word_count"`
	Signals    []detect.Signal `json:"signals"`
	Segments   []Segment       `json:"segments,omitempty" jsonschema:"per-chunk scores for longer documents"`
	AIFraction *float64        `json:"ai_fraction,omitempty" jsonschema:"share of segments scoring 50 or higher"`
	Notes      []string        `json:"notes,omitempty"`
}

type Segment struct {
	Index   int    `json:"index"`
	Score   int    `json:"score"`
	Words   int    `json:"words"`
	Preview string `json:"preview"`
}

type Engine struct {
	detectors []detect.Detector
	models    map[detect.Kind]*calibrate.Model
}

func New(ds ...detect.Detector) *Engine { return &Engine{detectors: ds} }

//go:embed models
var modelFS embed.FS

// Default returns an engine with every built-in detector plus the fitted
// word-pattern and calibration models embedded in the binary.
func Default() *Engine {
	e := Heuristic()
	e.models = map[detect.Kind]*calibrate.Model{}
	words := map[detect.Kind]*ngram.Model{}
	for _, k := range []detect.Kind{detect.KindProse, detect.KindCommit, detect.KindPR} {
		var m calibrate.Model
		if loadModel(string(k)+".json", &m) {
			e.models[k] = &m
		}
		var w ngram.Model
		if loadModel(string(k)+".ngram.json", &w) {
			words[k] = &w
		}
	}
	e.detectors = append(e.detectors, ngram.NewDetector(words))
	return e
}

func loadModel(name string, v any) bool {
	b, err := modelFS.ReadFile("models/" + name)
	if err != nil {
		return false
	}
	if err := json.Unmarshal(b, v); err != nil {
		panic(fmt.Sprintf("embedded model %s: %v", name, err))
	}
	return true
}

// Heuristic returns an engine that ignores fitted models and uses the
// hand-set signal weights, for comparison.
func Heuristic() *Engine { return New(stylometric.New(), devtext.New(), signs.New()) }

// segmentWords is the target chunk size for per-segment scoring.
const segmentWords = 200

func (e *Engine) Analyze(ctx context.Context, in detect.Input) (Result, error) {
	in.Text = strings.TrimSpace(in.Text)
	if in.Kind == "" || in.Kind == detect.KindAuto {
		in.Kind = detect.GuessKind(in.Text)
	}
	res := Result{Kind: in.Kind, WordCount: len(text.Words(in.Text))}
	if res.WordCount == 0 {
		res.Score, res.Label, res.Confidence = 50, label(50), "low"
		res.Notes = append(res.Notes, "no text to analyze")
		return res, nil
	}

	signals, err := e.signals(ctx, in)
	if err != nil {
		return res, err
	}
	conf := lengthConfidence(in.Kind, res.WordCount)
	score := e.score(in.Kind, signals, res.WordCount)
	res.Score = int(math.Round(score))
	res.Label = label(res.Score)
	res.Confidence = confidenceLabel(conf, signals)
	res.Signals = signals

	if conf < 0.4 && res.Score < 97 && e.models[in.Kind] == nil {
		res.Notes = append(res.Notes, fmt.Sprintf("only %d words; short texts are hard to judge, so the score is pulled toward 50", res.WordCount))
	}
	if len(signals) == 0 {
		res.Notes = append(res.Notes, "no detector produced a signal for this text")
	}

	if in.Kind == detect.KindProse && res.WordCount >= 2*segmentWords {
		if err := e.segments(ctx, in, &res); err != nil {
			return res, err
		}
	}
	return res, nil
}

func (e *Engine) signals(ctx context.Context, in detect.Input) ([]detect.Signal, error) {
	var all []detect.Signal
	for _, d := range e.detectors {
		s, err := d.Detect(ctx, in)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", d.Name(), err)
		}
		all = append(all, s...)
	}
	return all, nil
}

func (e *Engine) segments(ctx context.Context, in detect.Input, res *Result) error {
	windows := text.Windows(in.Text, segmentWords)
	if len(windows) < 2 {
		return nil
	}
	flagged := 0
	for i, w := range windows {
		s, err := e.signals(ctx, detect.Input{Text: w, Kind: in.Kind})
		if err != nil {
			return err
		}
		n := len(text.Words(w))
		score := int(math.Round(e.score(in.Kind, s, n)))
		if score >= 50 {
			flagged++
		}
		res.Segments = append(res.Segments, Segment{Index: i, Score: score, Words: n, Preview: preview(w)})
	}
	frac := float64(flagged) / float64(len(windows))
	res.AIFraction = &frac
	return nil
}

// score uses the fitted model for the kind if there is one, else the
// hand-set combination. A decisive signal always sets a floor of 97.
func (e *Engine) score(k detect.Kind, signals []detect.Signal, words int) float64 {
	m := e.models[k]
	if m == nil {
		return combine(signals, lengthConfidence(k, words))
	}
	feats := map[string]float64{}
	decisive := false
	for _, s := range signals {
		if s.Decisive {
			decisive = true
			continue // too rare to learn from; handled by the floor
		}
		feats[s.Name] = s.Value
	}
	score := 100 * m.Prob(feats, words)
	if decisive {
		score = max(score, 97)
	}
	return score
}

// combine merges signals in log-odds space. Dividing by sqrt(total weight)
// keeps many correlated weak signals from piling up into false certainty.
// Short texts are shrunk toward 50. A decisive signal sets a floor of 97.
//
// This is a placeholder for a logistic regression fitted on the eval set.
func combine(signals []detect.Signal, conf float64) float64 {
	var sum, wsum float64
	decisive := false
	for _, s := range signals {
		p := min(max(s.AIProb, 0.02), 0.98)
		sum += s.Weight * math.Log(p/(1-p))
		wsum += s.Weight
		decisive = decisive || s.Decisive
	}
	score := 50.0
	if wsum > 0 {
		logit := sum / math.Sqrt(wsum) * (0.25 + 0.75*conf)
		score = 100 / (1 + math.Exp(-logit))
	}
	if decisive {
		score = max(score, 97)
	}
	return score
}

// lengthConfidence is 0..1, reaching 1 at a word count typical for the kind.
func lengthConfidence(k detect.Kind, words int) float64 {
	full := 300.0
	switch k {
	case detect.KindCommit:
		full = 60
	case detect.KindPR:
		full = 150
	}
	return min(1, float64(words)/full)
}

func confidenceLabel(conf float64, signals []detect.Signal) string {
	for _, s := range signals {
		if s.Decisive {
			return "high"
		}
	}
	switch {
	case conf < 0.4 || len(signals) < 2:
		return "low"
	case conf < 0.8:
		return "medium"
	default:
		return "high"
	}
}

func label(score int) string {
	switch {
	case score >= 80:
		return "AI"
	case score >= 60:
		return "likely AI"
	case score > 40:
		return "uncertain"
	case score > 20:
		return "likely human"
	default:
		return "human"
	}
}

func preview(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 100 {
		return string(r[:100]) + "…"
	}
	return s
}
