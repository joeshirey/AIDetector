// Package engine runs detectors over text and combines their signals into a
// single 0–100 score.
package engine

import (
	"cmp"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/devtext"
	"github.com/joeshirey/AIDetector/internal/detect/ngram"
	"github.com/joeshirey/AIDetector/internal/detect/perplexity"
	"github.com/joeshirey/AIDetector/internal/detect/signs"
	"github.com/joeshirey/AIDetector/internal/detect/stylometric"
	"github.com/joeshirey/AIDetector/internal/text"
)

// Result is what the CLI and MCP tools return.
type Result struct {
	Score      int             `json:"score" jsonschema:"0-100 on a scale shared by all kinds: 40+ inconclusive, 60+ likely AI, 80+ AI; at most ~5%, ~2%, and ~0.5% of held-out human text of the same kind reaches each"`
	RawScore   int             `json:"raw_score" jsonschema:"the model's own 0-100 probability before mapping onto the shared scale"`
	Raw        float64         `json:"-"` // RawScore before rounding, for calibration and ranking metrics
	Label      string          `json:"label" jsonschema:"human, likely human, inconclusive, likely AI, AI, or too short to judge"`
	Confidence string          `json:"confidence" jsonschema:"low, medium, or high; mostly driven by text length"`
	Kind       detect.Kind     `json:"kind"`
	WordCount  int             `json:"word_count"`
	Reasons    []Reason        `json:"reasons" jsonschema:"why the text got its score, strongest first"`
	Signals    []detect.Signal `json:"signals" jsonschema:"raw measurements from every detector"`
	Segments   []Segment       `json:"segments,omitempty" jsonschema:"per-chunk scores for longer documents"`
	AIFraction *float64        `json:"ai_fraction,omitempty" jsonschema:"share of segments scoring 60+ on the segment scale (calibrated separately from whole documents)"`
	Cutoffs    Thresholds      `json:"cutoffs" jsonschema:"raw-score cutoffs for this kind that map to 40, 60, and 80, with measured human rates"`
	Scorer     string          `json:"scorer,omitempty" jsonschema:"token-probability model used, if the local scorer was available"`
	Notes      []string        `json:"notes,omitempty"`
}

// Reason explains one factor behind a score.
type Reason struct {
	Signal    string   `json:"signal"`
	Direction string   `json:"direction" jsonschema:"AI, human, warning, or info"`
	Impact    string   `json:"impact" jsonschema:"decisive, high, medium, or low"`
	Logit     float64  `json:"log_odds" jsonschema:"contribution to the score in log-odds; positive pushes toward AI"`
	Observed  float64  `json:"observed"`
	TypicalH  *float64 `json:"typical_human,omitempty" jsonschema:"median value in human training text"`
	TypicalAI *float64 `json:"typical_ai,omitempty" jsonschema:"median value in AI training text"`
	Detail    string   `json:"detail"`
}

type Segment struct {
	Index    int     `json:"index"`
	Score    int     `json:"score"`
	RawScore int     `json:"raw_score" jsonschema:"the model's own probability for this chunk before calibration"`
	Raw      float64 `json:"-"`
	Words    int     `json:"words"`
	Preview  string  `json:"preview"`
}

type Engine struct {
	detectors  []detect.Detector
	models     map[detect.Kind]*calibrate.Model
	thresholds map[string]Thresholds
	lm         *perplexity.Detector
	lmKinds    map[detect.Kind]bool // kinds scored by a model fitted with lm_* features
}

// Optional detectors may fail (e.g. a local service is down) without failing
// the whole analysis; their signals are simply absent.
type optional interface{ Optional() bool }

// Thresholds are raw-score cutoffs, chosen per kind so that a fixed share of
// held-out human text reaches each one. Scale maps them to 40, 60, and 80 so
// scores mean the same thing for every kind. The Human* fields record the
// share measured on the other half of that text.
type Thresholds struct {
	Mixed    int `json:"mixed" jsonschema:"at or above: inconclusive"`
	LikelyAI int `json:"likely_ai" jsonschema:"at or above: likely AI"`
	AI       int `json:"ai" jsonschema:"at or above: AI"`

	HumanAtMixed    float64 `json:"human_at_mixed,omitempty" jsonschema:"share of held-out human texts scoring at or above mixed"`
	HumanAtLikelyAI float64 `json:"human_at_likely_ai,omitempty" jsonschema:"share of held-out human texts scoring at or above likely AI"`
	HumanAtAI       float64 `json:"human_at_ai,omitempty" jsonschema:"share of held-out human texts scoring at or above AI"`
	CalibratedOn    int     `json:"calibrated_on,omitempty" jsonschema:"human texts used to set the cutoffs"`
}

// DefaultThresholds apply to kinds without fitted cutoffs; Scale is then the
// identity.
var DefaultThresholds = Thresholds{Mixed: 40, LikelyAI: 60, AI: 80}

// Scale maps a raw score onto the shared scale, piecewise linearly through
// (0,0), (Mixed,40), (LikelyAI,60), (AI,80), (100,100). It preserves order.
func (t Thresholds) Scale(raw float64) float64 {
	xs := []float64{0, float64(t.Mixed), float64(t.LikelyAI), float64(t.AI), 100}
	ys := []float64{0, 40, 60, 80, 100}
	for i := 1; i < len(xs); i++ {
		if raw <= xs[i] {
			if xs[i] == xs[i-1] {
				return ys[i]
			}
			return ys[i-1] + (raw-xs[i-1])/(xs[i]-xs[i-1])*(ys[i]-ys[i-1])
		}
	}
	return 100
}

// Label names a score on the shared scale.
func Label(score int) string {
	switch {
	case score >= 80:
		return "AI"
	case score >= 60:
		return "likely AI"
	case score >= 40:
		return "inconclusive"
	case score >= 20:
		return "likely human"
	default:
		return "human"
	}
}

// ThresholdKey names a kind's entry in thresholds.json. Models fitted with
// token-probability features have their own cutoffs ("prose.lm").
func (e *Engine) ThresholdKey(k detect.Kind) string {
	if e.lmKinds[k] {
		return string(k) + ".lm"
	}
	return string(k)
}

// Thresholds returns the label cutoffs for a kind.
func (e *Engine) Thresholds(k detect.Kind) Thresholds {
	if t, ok := e.thresholds[e.ThresholdKey(k)]; ok {
		return t
	}
	return DefaultThresholds
}

// SegmentThresholds returns the cutoffs for 200-word segments. Short windows
// score more extreme than whole documents, so they get their own
// calibration ("prose.segment"); without one they fall back to the
// document cutoffs.
func (e *Engine) SegmentThresholds(k detect.Kind) Thresholds {
	if t, ok := e.thresholds[e.ThresholdKey(k)+".segment"]; ok {
		return t
	}
	return e.Thresholds(k)
}

// Scorer returns the token-probability scorer in use, or nil.
func (e *Engine) Scorer() *perplexity.Detector { return e.lm }

func New(ds ...detect.Detector) *Engine { return &Engine{detectors: ds} }

//go:embed models
var modelFS embed.FS

// Default returns an engine with every built-in detector plus the fitted
// word-pattern and calibration models embedded in the binary. When the local
// token-probability scorer is reachable it also uses the models fitted with
// its features (<kind>.lm.json).
func Default() *Engine {
	sub, _ := fs.Sub(modelFS, "models")
	return fromFS(sub)
}

// EmbeddedFile returns one of the embedded production model files.
func EmbeddedFile(name string) ([]byte, error) { return modelFS.ReadFile("models/" + name) }

// EmbeddedNames lists the embedded production model files.
func EmbeddedNames() ([]string, error) {
	ents, err := modelFS.ReadDir("models")
	var out []string
	for _, e := range ents {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
			out = append(out, e.Name())
		}
	}
	return out, err
}

// FromDir loads candidate models from dir (prose.json, prose.ngram.json,
// pr.json, pr.ngram.json, thresholds.json). Files missing from dir fall back
// to the embedded ones, so a candidate can replace just one kind.
func FromDir(dir string) (*Engine, error) {
	if _, err := os.Stat(dir); err != nil {
		return nil, err
	}
	sub, _ := fs.Sub(modelFS, "models")
	return fromFS(overlayFS{os.DirFS(dir), sub}), nil
}

// overlayFS reads from top first, then base.
type overlayFS struct{ top, base fs.FS }

func (o overlayFS) Open(name string) (fs.File, error) {
	if f, err := o.top.Open(name); err == nil {
		return f, nil
	}
	return o.base.Open(name)
}

func fromFS(models fs.FS) *Engine {
	loadModel := func(name string, v any) bool {
		b, err := fs.ReadFile(models, name)
		if err != nil {
			return false
		}
		if err := json.Unmarshal(b, v); err != nil {
			panic(fmt.Sprintf("model %s: %v", name, err))
		}
		return true
	}
	e := Heuristic()
	e.models = map[detect.Kind]*calibrate.Model{}
	e.lmKinds = map[detect.Kind]bool{}
	words := map[detect.Kind]*ngram.Model{}
	for _, k := range []detect.Kind{detect.KindProse, detect.KindPR} {
		var m, lm calibrate.Model
		if e.lm != nil && loadModel(string(k)+".lm.json", &lm) {
			e.models[k], e.lmKinds[k] = &lm, true
		} else if loadModel(string(k)+".json", &m) {
			e.models[k] = &m
		}
		var w ngram.Model
		if loadModel(string(k)+".ngram.json", &w) {
			words[k] = &w
		}
	}
	e.detectors = append(e.detectors, ngram.NewDetector(words))
	loadModel("thresholds.json", &e.thresholds)
	return e
}

// Heuristic returns an engine that ignores fitted models and uses the
// hand-set signal weights, for comparison. It includes the token-probability
// scorer when AIDETECTOR_SCORER_URL points at a running one.
func Heuristic() *Engine {
	e := New(stylometric.New(), devtext.New(), signs.New())
	if lm := perplexity.FromEnv(); lm != nil {
		e.lm = lm
		e.detectors = append(e.detectors, lm)
	}
	return e
}

// minWords is the shortest text each kind's model can judge. The training
// data has almost no prose under 50 words (commercial detectors use a
// similar floor).
var minWords = map[detect.Kind]int{detect.KindProse: 50, detect.KindPR: 25}

// TooShort is the label for text below a kind's minimum length.
const TooShort = "too short to judge"

// segmentWords is the target chunk size for per-segment scoring.
const segmentWords = 200

func (e *Engine) Analyze(ctx context.Context, in detect.Input) (Result, error) {
	res, err := e.analyze(ctx, in)
	if err != nil || res.Kind != detect.KindProse || res.Score < 60 {
		return res, err
	}
	// If one embedded or quoted block carries most of the AI evidence, say so
	// instead of letting it silently set the label.
	clean, _ := text.Sanitize(in.Text)
	for _, b := range text.EmbeddedBlocks(clean) {
		without, err := e.analyze(ctx, detect.Input{Text: b.Without, Kind: detect.KindProse})
		if err != nil {
			return res, err
		}
		if (without.Score < 60 || res.Score-without.Score >= 20) && without.Label != TooShort {
			res.Reasons = append([]Reason{{
				Signal: "embedded_block", Direction: "warning", Impact: "high", Observed: float64(b.Words),
				Detail: fmt.Sprintf("most of the AI evidence comes from one %s (%d words, starting %q); without it this text scores %d (%s)",
					b.What, b.Words, b.Preview, without.Score, without.Label),
			}}, res.Reasons...)
			break
		}
	}
	return res, nil
}

// Prepare returns the text the word-pattern model sees at scoring time:
// sanitized, and for prose with code, tables, front matter, and label
// lines removed. Training uses it too, so both see the same text.
func Prepare(kind detect.Kind, s string) string {
	clean, _ := text.Sanitize(s)
	s = strings.TrimSpace(clean)
	if kind == detect.KindProse {
		s, _ = text.PrepareProse(s)
	}
	return s
}

func (e *Engine) analyze(ctx context.Context, in detect.Input) (Result, error) {
	clean, tamper := text.Sanitize(in.Text)
	in.Text = strings.TrimSpace(clean)
	if in.Kind == "" || in.Kind == detect.KindAuto {
		in.Kind = detect.GuessKind(in.Text)
	}
	// Score only the prose: code, tables, front matter, and label lines are
	// excluded, and sentences glued together by lost line breaks are split.
	var excluded text.Exclusions
	if in.Kind == detect.KindProse {
		in.Text, excluded = text.PrepareProse(in.Text)
	}
	res := Result{Kind: in.Kind, WordCount: len(text.Words(in.Text))}
	if e.lm != nil {
		res.Scorer = e.lm.Model()
	}
	if res.WordCount == 0 {
		res.Cutoffs = e.Thresholds(in.Kind)
		res.Score, res.RawScore, res.Raw, res.Label, res.Confidence = 50, 50, 50, TooShort, "low"
		res.Notes = append(res.Notes, "no prose to analyze")
		return res, nil
	}

	signals, err := e.signals(ctx, in)
	if err != nil {
		return res, err
	}
	if n := excluded.Total(); n > 0 {
		signals = append(signals, detect.Signal{
			Name: "excluded_content", Detector: "prepare", Value: float64(n), AIProb: 0.5, Weight: 0,
			Detail: exclusionDetail(excluded),
		})
	}
	if tamper.Any() {
		signals = append(signals, detect.Signal{
			Name: "unicode_tampering", Detector: "sanitizer", Value: float64(tamper.ZeroWidth + tamper.Homoglyphs),
			AIProb: 0.5, Weight: 0,
			Detail: fmt.Sprintf("removed %d invisible characters and %d look-alike letters before scoring; text may have been altered to evade or confuse detectors", tamper.ZeroWidth, tamper.Homoglyphs),
		})
		res.Notes = append(res.Notes, "the text contained hidden or look-alike characters, which were removed before scoring")
	}
	conf := lengthConfidence(in.Kind, res.WordCount)
	raw := e.score(in.Kind, signals, res.WordCount)
	res.Cutoffs = e.Thresholds(in.Kind)
	res.RawScore, res.Raw = int(math.Round(raw)), raw
	res.Score = int(math.Round(e.scale(in.Kind, raw, signals)))
	res.Label = Label(res.Score)
	res.Confidence = confidenceLabel(conf, signals)
	if n := minWords[in.Kind]; res.WordCount < n && res.Score < 97 {
		res.Label, res.Confidence = TooShort, "low"
		res.Notes = append(res.Notes, fmt.Sprintf("%d words is below the %d-word minimum for %s; the score is not meaningful at this length", res.WordCount, n, in.Kind))
	}
	res.Signals = signals
	res.Reasons = e.reasons(in.Kind, signals, res.WordCount)

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
			if o, ok := d.(optional); ok && o.Optional() {
				continue
			}
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
		raw := e.score(in.Kind, s, n)
		score := int(math.Round(e.scaleWith(e.SegmentThresholds(in.Kind), raw, s)))
		if score >= 60 {
			flagged++
		}
		res.Segments = append(res.Segments, Segment{Index: i, Score: score, RawScore: int(math.Round(raw)), Raw: raw, Words: n, Preview: preview(w)})
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

// scale maps a raw score onto the shared scale. A decisive tell still
// guarantees at least 97.
func (e *Engine) scale(k detect.Kind, raw float64, signals []detect.Signal) float64 {
	return e.scaleWith(e.Thresholds(k), raw, signals)
}

func (e *Engine) scaleWith(t Thresholds, raw float64, signals []detect.Signal) float64 {
	scaled := t.Scale(raw)
	for _, s := range signals {
		if s.Decisive {
			return max(scaled, 97)
		}
	}
	return scaled
}

// reasons ranks the factors behind a score. With a fitted model these are the
// model's actual per-feature contributions; otherwise the hand-set weights.
func (e *Engine) reasons(k detect.Kind, signals []detect.Signal, words int) []Reason {
	var out, info []Reason
	byName := map[string]detect.Signal{}
	for _, s := range signals {
		byName[s.Name] = s
		switch {
		case s.Decisive:
			out = append(out, Reason{Signal: s.Name, Direction: "AI", Impact: "decisive", Logit: 10, Observed: s.Value, Detail: s.Detail})
		case s.Detector == "sanitizer":
			out = append(out, Reason{Signal: s.Name, Direction: "warning", Impact: "high", Observed: s.Value, Detail: s.Detail})
		case s.Detector == "prepare":
			info = append(info, Reason{Signal: s.Name, Direction: "info", Impact: "low", Observed: s.Value, Detail: s.Detail})
		}
	}
	add := func(r Reason) {
		if math.Abs(r.Logit) < 0.15 {
			return
		}
		r.Direction = "human"
		if r.Logit > 0 {
			r.Direction = "AI"
		}
		switch a := math.Abs(r.Logit); {
		case a >= 1:
			r.Impact = "high"
		case a >= 0.4:
			r.Impact = "medium"
		default:
			r.Impact = "low"
		}
		out = append(out, r)
	}
	if m := e.models[k]; m != nil {
		feats := map[string]float64{}
		for _, s := range signals {
			if !s.Decisive && s.Detector != "sanitizer" && s.Detector != "prepare" {
				feats[s.Name] = s.Value
			}
		}
		for _, c := range m.Contributions(feats, words) {
			if !c.Present {
				continue
			}
			h, a := c.HumanMedian, c.AIMedian
			// With correlated features a weight can flip sign. Skip a reason
			// whose push contradicts which typical value it's closer to, since
			// it would read as nonsense even though the total is right.
			if c.Name != "word_patterns" && h != a && (math.Abs(c.Value-a) < math.Abs(c.Value-h)) != (c.Logit > 0) {
				continue
			}
			detail := byName[c.Name].Detail
			if c.Name == "log_words" {
				detail = fmt.Sprintf("length: %d words", words)
			}
			add(Reason{Signal: c.Name, Logit: c.Logit, Observed: c.Value, TypicalH: &h, TypicalAI: &a, Detail: detail})
		}
	} else {
		for _, s := range signals {
			if s.Decisive || s.Weight == 0 {
				continue
			}
			p := min(max(s.AIProb, 0.02), 0.98)
			add(Reason{Signal: s.Name, Logit: s.Weight * math.Log(p/(1-p)), Observed: s.Value, Detail: s.Detail})
		}
	}
	rank := map[string]int{"decisive": 0}
	slices.SortStableFunc(out, func(a, b Reason) int {
		_, okA := rank[a.Impact]
		_, okB := rank[b.Impact]
		if okA != okB {
			if okA {
				return -1
			}
			return 1
		}
		if a.Direction == "warning" != (b.Direction == "warning") {
			if a.Direction == "warning" {
				return -1
			}
			return 1
		}
		return cmp.Compare(math.Abs(b.Logit), math.Abs(a.Logit))
	})
	if len(out) > 8 {
		out = out[:8]
	}
	return append(out, info...)
}

func exclusionDetail(x text.Exclusions) string {
	var parts []string
	for _, p := range []struct {
		n    int
		what string
	}{{x.Code, "code"}, {x.Tables, "tables"}, {x.FrontMatter, "front matter"}, {x.Labels, "headings and labels"}} {
		if p.n > 0 {
			parts = append(parts, fmt.Sprintf("%d of %s", p.n, p.what))
		}
	}
	return fmt.Sprintf("excluded %d words that aren't prose before scoring (%s)", x.Total(), strings.Join(parts, ", "))
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

func preview(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 100 {
		return string(r[:100]) + "…"
	}
	return s
}
