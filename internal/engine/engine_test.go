package engine

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/ngram"
)

// These fixtures check that each hand-set signal points the right way, so
// they run against the heuristic engine. Fitted models are judged by
// `aidetector eval -split test` on the corpus, not by hand-written examples.
func analyze(t *testing.T, text string, kind detect.Kind) Result {
	t.Helper()
	res, err := Heuristic().Analyze(context.Background(), detect.Input{Text: text, Kind: kind})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// ai_blog.md is stock LLM style; casual_blog.md imitates a casual human
// writer. (Both were written by a model, so they test the hand-set signals'
// directions, not real-world accuracy.)
func TestProseDirection(t *testing.T) {
	ai := analyze(t, fixture(t, "ai_blog.md"), detect.KindAuto)
	casual := analyze(t, fixture(t, "casual_blog.md"), detect.KindAuto)
	if ai.Kind != detect.KindProse || casual.Kind != detect.KindProse {
		t.Fatalf("kinds = %s, %s; want prose", ai.Kind, casual.Kind)
	}
	if ai.Score < 80 {
		t.Errorf("stock-style blog scored %d, want >= 80", ai.Score)
	}
	if casual.Score > 30 {
		t.Errorf("casual-style blog scored %d, want <= 30", casual.Score)
	}
}

const agentPR = "## Summary\n\nFix a nil pointer in the config loader when the file is empty.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"

func TestAIAttributionIsDecisive(t *testing.T) {
	if res := analyze(t, agentPR, detect.KindAuto); res.Score < 97 || res.Kind != detect.KindPR {
		t.Errorf("got score %d kind %s, want >= 97 pr", res.Score, res.Kind)
	}
}

func TestShortTextIsTooShort(t *testing.T) {
	res := analyze(t, "wip", detect.KindAuto)
	if res.Kind != detect.KindProse || res.Label != TooShort || res.Confidence != "low" {
		t.Errorf("got %s / %q / %s, want prose / %q / low", res.Kind, res.Label, res.Confidence, TooShort)
	}
}

func TestSegments(t *testing.T) {
	text := fixture(t, "casual_blog.md") + "\n\n" + fixture(t, "ai_blog.md")
	res := analyze(t, text, detect.KindProse)
	if len(res.Segments) < 2 {
		t.Fatalf("got %d segments, want >= 2", len(res.Segments))
	}
	first, last := res.Segments[0], res.Segments[len(res.Segments)-1]
	if first.Score >= last.Score {
		t.Errorf("casual-first segment %d should score below stock-style segment %d", first.Score, last.Score)
	}
}

func TestEmpty(t *testing.T) {
	if res := analyze(t, "   ", detect.KindAuto); res.Score != 50 {
		t.Errorf("empty text scored %d, want 50", res.Score)
	}
}

func TestEmbeddedModelsLoad(t *testing.T) {
	e := Default()
	for _, k := range []detect.Kind{detect.KindProse, detect.KindPR} {
		if e.models[k] == nil {
			t.Errorf("no embedded model for %s", k)
		}
	}
	// A decisive tell still wins over the fitted model.
	res, err := e.Analyze(context.Background(), detect.Input{Text: agentPR})
	if err != nil || res.Score < 97 {
		t.Errorf("got %d, %v; want >= 97", res.Score, err)
	}
}

func TestTamperingWarning(t *testing.T) {
	text := "The re​sult of the review was fіne, and the team agreed to ship the change on Monday after one more pass over the tests."
	res, err := Default().Analyze(context.Background(), detect.Input{Text: text, Kind: detect.KindProse})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Reasons) == 0 || res.Reasons[0].Direction != "warning" {
		t.Errorf("want a leading warning reason, got %+v", res.Reasons)
	}
}

func TestScale(t *testing.T) {
	th := Thresholds{Mixed: 14, LikelyAI: 44, AI: 56}
	for _, c := range []struct{ raw, want float64 }{{0, 0}, {14, 40}, {44, 60}, {56, 80}, {100, 100}, {29, 50}} {
		if got := th.Scale(c.raw); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("Scale(%v) = %v, want %v", c.raw, got, c.want)
		}
	}
	if DefaultThresholds.Scale(37) != 37 {
		t.Error("default thresholds should be the identity")
	}
}

func TestTooShort(t *testing.T) {
	text := "Ok so the deploy broke again. I think its the cache thing from last week, Dave said he'd look but he is out til Monday."
	res, err := Default().Analyze(context.Background(), detect.Input{Text: text, Kind: detect.KindProse})
	if err != nil {
		t.Fatal(err)
	}
	if res.Label != TooShort || res.Confidence != "low" {
		t.Errorf("got %q / %q, want %q / low", res.Label, res.Confidence, TooShort)
	}
	// A decisive tell still counts on short text.
	res, _ = Default().Analyze(context.Background(), detect.Input{Text: "As an AI language model, I can't share opinions on this topic.", Kind: detect.KindProse})
	if res.Label != "AI" {
		t.Errorf("decisive short text got %q", res.Label)
	}
}

type shortGuardStubDetector struct {
	signals []detect.Signal
}

func (d shortGuardStubDetector) Name() string { return "short_guard_stub" }

func (d shortGuardStubDetector) Detect(context.Context, detect.Input) ([]detect.Signal, error) {
	return d.signals, nil
}

func notesContain(notes []string, sub string) bool {
	for _, n := range notes {
		if strings.Contains(n, sub) {
			return true
		}
	}
	return false
}

func TestShortProseHighScoreWithoutDecisiveIsTooShort(t *testing.T) {
	ctx := context.Background()

	t.Run("fitted_prose_under_50_words_high_score_no_decisive", func(t *testing.T) {
		prose := "In today's fast-paced digital landscape, observability has become more than just a buzzword — it's a crucial pillar of modern software engineering. Whether you're a seasoned SRE or a developer just starting out, understanding how to harness telemetry can be transformative for your team."
		res, err := Default().Analyze(ctx, detect.Input{Text: prose, Kind: detect.KindProse})
		if err != nil {
			t.Fatal(err)
		}
		if res.WordCount >= 50 {
			t.Fatalf("precondition failed: WordCount = %d, want < 50", res.WordCount)
		}
		if res.Score < 97 {
			t.Fatalf("precondition failed: Score = %d, want >= 97 to exercise the bug", res.Score)
		}
		for _, s := range res.Signals {
			if s.Decisive {
				t.Fatalf("precondition failed: unexpected decisive signal %q", s.Name)
			}
		}
		if res.Label != TooShort || res.Confidence != "low" {
			t.Errorf("got Label=%q Confidence=%q (Score=%d, WordCount=%d), want Label=%q Confidence=\"low\"",
				res.Label, res.Confidence, res.Score, res.WordCount, TooShort)
		}
		if !notesContain(res.Notes, "below the 50-word minimum for prose") {
			t.Errorf("Notes = %q, want minimum-length note containing %q", res.Notes, "below the 50-word minimum for prose")
		}
	})

	t.Run("synthetic_under_minimum_high_score_no_decisive", func(t *testing.T) {
		e := New(shortGuardStubDetector{signals: []detect.Signal{
			{Name: "synthetic_high", Detector: "short_guard_stub", Value: 1, AIProb: 0.98, Weight: 100, Decisive: false},
			{Name: "synthetic_aux", Detector: "short_guard_stub", Value: 1, AIProb: 0.98, Weight: 100, Decisive: false},
		}})

		prose49 := strings.TrimSpace(strings.Repeat("alpha ", 49))
		res, err := e.Analyze(ctx, detect.Input{Text: prose49, Kind: detect.KindProse})
		if err != nil {
			t.Fatal(err)
		}
		if res.WordCount != 49 || res.Score < 97 {
			t.Fatalf("precondition failed: WordCount=%d Score=%d, want 49 and >= 97", res.WordCount, res.Score)
		}
		if res.Label != TooShort || res.Confidence != "low" {
			t.Errorf("49-word prose: got Label=%q Confidence=%q, want %q / \"low\"", res.Label, res.Confidence, TooShort)
		}
		if !notesContain(res.Notes, "49 words is below the 50-word minimum for prose") {
			t.Errorf("49-word prose Notes = %q, want 50-word minimum note", res.Notes)
		}

		pr24 := strings.TrimSpace(strings.Repeat("alpha ", 24))
		resPR, err := e.Analyze(ctx, detect.Input{Text: pr24, Kind: detect.KindPR})
		if err != nil {
			t.Fatal(err)
		}
		if resPR.WordCount != 24 || resPR.Score < 97 {
			t.Fatalf("precondition failed: WordCount=%d Score=%d, want 24 and >= 97", resPR.WordCount, resPR.Score)
		}
		if resPR.Label != TooShort || resPR.Confidence != "low" {
			t.Errorf("24-word PR: got Label=%q Confidence=%q, want %q / \"low\"", resPR.Label, resPR.Confidence, TooShort)
		}
		if !notesContain(resPR.Notes, "24 words is below the 25-word minimum for pr") {
			t.Errorf("24-word PR Notes = %q, want 25-word minimum note", resPR.Notes)
		}
	})

	t.Run("short_pr_with_decisive_tell_gets_real_label", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			eng  *Engine
		}{
			{"default", Default()},
			{"heuristic", Heuristic()},
		} {
			res, err := tc.eng.Analyze(ctx, detect.Input{Text: agentPR, Kind: detect.KindPR})
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if res.Kind != detect.KindPR || res.WordCount >= 25 {
				t.Fatalf("%s precondition failed: Kind=%s WordCount=%d, want pr < 25 words", tc.name, res.Kind, res.WordCount)
			}
			if res.Score < 97 || res.Label != "AI" || res.Confidence != "high" {
				t.Errorf("%s: got Score=%d Label=%q Confidence=%q, want >=97 / \"AI\" / \"high\"",
					tc.name, res.Score, res.Label, res.Confidence)
			}
			if notesContain(res.Notes, "below the 25-word minimum") {
				t.Errorf("%s: decisive short PR should not get minimum-length note, got Notes=%q", tc.name, res.Notes)
			}
			if notesContain(res.Notes, "pulled toward 50") {
				t.Errorf("%s: decisive short PR should not get pulled-toward-50 note, got Notes=%q", tc.name, res.Notes)
			}
		}
	})

	t.Run("at_or_above_minimum_unchanged", func(t *testing.T) {
		e := New(shortGuardStubDetector{signals: []detect.Signal{
			{Name: "synthetic_high", Detector: "short_guard_stub", Value: 1, AIProb: 0.98, Weight: 100, Decisive: false},
			{Name: "synthetic_aux", Detector: "short_guard_stub", Value: 1, AIProb: 0.98, Weight: 100, Decisive: false},
		}})

		prose50 := strings.TrimSpace(strings.Repeat("alpha ", 50))
		resProse, err := e.Analyze(ctx, detect.Input{Text: prose50, Kind: detect.KindProse})
		if err != nil {
			t.Fatal(err)
		}
		if resProse.WordCount != 50 || resProse.Label != "AI" {
			t.Errorf("50-word prose: got WordCount=%d Label=%q, want 50 / \"AI\"", resProse.WordCount, resProse.Label)
		}
		if notesContain(resProse.Notes, "below the 50-word minimum") {
			t.Errorf("50-word prose Notes = %q, should not have minimum-length note", resProse.Notes)
		}

		pr25 := strings.TrimSpace(strings.Repeat("alpha ", 25))
		resPR, err := e.Analyze(ctx, detect.Input{Text: pr25, Kind: detect.KindPR})
		if err != nil {
			t.Fatal(err)
		}
		if resPR.WordCount != 25 || resPR.Label != "AI" {
			t.Errorf("25-word PR: got WordCount=%d Label=%q, want 25 / \"AI\"", resPR.WordCount, resPR.Label)
		}
		if notesContain(resPR.Notes, "below the 25-word minimum") {
			t.Errorf("25-word PR Notes = %q, should not have minimum-length note", resPR.Notes)
		}

		fittedProse, err := Default().Analyze(ctx, detect.Input{Text: fixture(t, "ai_blog.md"), Kind: detect.KindProse})
		if err != nil {
			t.Fatal(err)
		}
		if fittedProse.WordCount < 50 || fittedProse.Label == TooShort {
			t.Errorf("fitted >=50w prose: got WordCount=%d Label=%q, want >=50 and not %q", fittedProse.WordCount, fittedProse.Label, TooShort)
		}

		prLong := "## Summary\n\nFix a nil pointer dereference in the configuration loader when the file on disk is empty by returning defaults instead of dereferencing an uninitialized struct pointer during startup."
		fittedPR, err := Default().Analyze(ctx, detect.Input{Text: prLong, Kind: detect.KindPR})
		if err != nil {
			t.Fatal(err)
		}
		if fittedPR.WordCount < 25 || fittedPR.Label == TooShort {
			t.Errorf("fitted >=25w PR: got WordCount=%d Label=%q, want >=25 and not %q", fittedPR.WordCount, fittedPR.Label, TooShort)
		}
	})
}

func TestShortHeuristicNoteDependsOnDecisiveNotScore(t *testing.T) {
	ctx := context.Background()

	t.Run("non_decisive_high_score_includes_pulled_toward_50_note", func(t *testing.T) {
		e := New(shortGuardStubDetector{signals: []detect.Signal{
			{Name: "synthetic_high", Detector: "short_guard_stub", Value: 1, AIProb: 0.98, Weight: 100, Decisive: false},
		}})

		// 10 words (< 50 minWords and conf = 10/300 < 0.4) with Score >= 97 and no decisive signal:
		// must get both TooShort + minimum-length note AND the heuristic pulled-toward-50 note.
		short10 := strings.TrimSpace(strings.Repeat("alpha ", 10))
		res10, err := e.Analyze(ctx, detect.Input{Text: short10, Kind: detect.KindProse})
		if err != nil {
			t.Fatal(err)
		}
		if res10.WordCount != 10 || res10.Score < 97 {
			t.Fatalf("precondition failed: WordCount=%d Score=%d, want 10 and >= 97", res10.WordCount, res10.Score)
		}
		if res10.Label != TooShort || res10.Confidence != "low" {
			t.Errorf("10-word heuristic prose: got Label=%q Confidence=%q, want %q / \"low\"", res10.Label, res10.Confidence, TooShort)
		}
		if !notesContain(res10.Notes, "10 words is below the 50-word minimum for prose") {
			t.Errorf("10-word heuristic prose Notes = %q, want minimum-length note", res10.Notes)
		}
		if !notesContain(res10.Notes, "pulled toward 50") {
			t.Errorf("10-word heuristic prose Notes = %q, want \"pulled toward 50\" note", res10.Notes)
		}

		// 60 words (>= 50 minWords, but conf = 60/300 = 0.2 < 0.4) with Score >= 97 and no decisive signal:
		// not TooShort, yet engine.go:408 still applies because conf < 0.4 and no signal is Decisive.
		short60 := strings.TrimSpace(strings.Repeat("alpha ", 60))
		res60, err := e.Analyze(ctx, detect.Input{Text: short60, Kind: detect.KindProse})
		if err != nil {
			t.Fatal(err)
		}
		if res60.WordCount != 60 || res60.Score < 97 {
			t.Fatalf("precondition failed: WordCount=%d Score=%d, want 60 and >= 97", res60.WordCount, res60.Score)
		}
		if res60.Label != "AI" {
			t.Errorf("60-word heuristic prose: got Label=%q, want \"AI\"", res60.Label)
		}
		if !notesContain(res60.Notes, "pulled toward 50") {
			t.Errorf("60-word heuristic prose Notes = %q, want \"pulled toward 50\" note", res60.Notes)
		}
	})

	t.Run("decisive_signal_omits_pulled_toward_50_note", func(t *testing.T) {
		e := New(shortGuardStubDetector{signals: []detect.Signal{
			{Name: "synthetic_decisive", Detector: "short_guard_stub", Value: 1, AIProb: 0.99, Weight: 3, Decisive: true},
		}})

		for _, nWords := range []int{10, 60} {
			txt := strings.TrimSpace(strings.Repeat("alpha ", nWords))
			res, err := e.Analyze(ctx, detect.Input{Text: txt, Kind: detect.KindProse})
			if err != nil {
				t.Fatal(err)
			}
			if res.Score < 97 || res.Label != "AI" || res.Confidence != "high" {
				t.Errorf("%d-word decisive prose: got Score=%d Label=%q Confidence=%q, want >=97 / \"AI\" / \"high\"",
					nWords, res.Score, res.Label, res.Confidence)
			}
			if notesContain(res.Notes, "pulled toward 50") {
				t.Errorf("%d-word decisive prose Notes = %q, should not contain \"pulled toward 50\"", nWords, res.Notes)
			}
			if notesContain(res.Notes, "below the 50-word minimum") {
				t.Errorf("%d-word decisive prose Notes = %q, should not contain minimum-length note", nWords, res.Notes)
			}
		}
	})
}

func TestSegmentThresholdsFallBack(t *testing.T) {
	e := &Engine{thresholds: map[string]Thresholds{"prose": {Mixed: 20, LikelyAI: 50, AI: 65}}}
	if got := e.SegmentThresholds(detect.KindProse); got.LikelyAI != 50 {
		t.Errorf("without prose.segment, segments should use document cutoffs; got %+v", got)
	}
	e.thresholds["prose.segment"] = Thresholds{Mixed: 45, LikelyAI: 76, AI: 83}
	if got := e.SegmentThresholds(detect.KindProse); got.LikelyAI != 76 {
		t.Errorf("got %+v, want the segment cutoffs", got)
	}
}

func TestEmbeddedBlockReason(t *testing.T) {
	human := "I spent most of last week fighting our deploy pipeline, and honestly it was my own fault for skipping the staging step. "
	skill := "## Workflow\n\n1. **Assess**: Evaluate the user's requirements against key architectural dimensions and constraints.\n2. **Select**: Map the workload to the primary database skill and check for relevant guidance.\n\n> 📖 Consult the detailed [decision matrix](./references/matrix.md) for complete comparison tables.\n\n### Step 2: Select the Target Skills\n\n- **Install**: Seamlessly download and install the specialized skill into the local workspace.\n- **Execute**: Deliver a comprehensive, well-architected solution that ensures robust, scalable outcomes."
	text := strings.Repeat(human, 3) + "\n\n" + skill + "\n\n" + strings.Repeat(human, 3)
	res, err := Default().Analyze(context.Background(), detect.Input{Text: text, Kind: detect.KindProse})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res.Reasons {
		if r.Signal == "embedded_block" && res.Score < 60 {
			t.Errorf("embedded_block reason on a text scoring %d", res.Score)
		}
	}
}

// A fitted model's listed log-odds (signals, baseline, and other) add up to
// the raw score's log-odds, so explanations account for the whole score.
func TestReasonsAddUp(t *testing.T) {
	e := Default()
	txt := "I spent the weekend fixing the fence, which had been leaning for years. The posts were rotten at the base, so I dug them out one by one and set new ones in concrete. It took longer than I expected, mostly because the old concrete footings were huge. By Sunday night my back hurt, but the fence stood straight for the first time since we bought the house. Next spring I want to paint it, though my wife thinks we should just let it weather."
	r, err := e.Analyze(context.Background(), detect.Input{Text: txt, Kind: detect.KindProse})
	if err != nil {
		t.Fatal(err)
	}
	sum, baseline := 0.0, false
	for _, x := range r.Reasons {
		if x.Direction == "info" || x.Direction == "warning" || x.Impact == "decisive" {
			continue
		}
		sum += x.Logit
		baseline = baseline || x.Signal == "baseline"
	}
	if !baseline {
		t.Fatal("no baseline reason")
	}
	want := math.Log(r.Raw / (100 - r.Raw))
	if math.Abs(sum-want) > 0.01 {
		t.Errorf("reasons sum to %.3f log-odds, raw score is %.3f", sum, want)
	}
}

func TestBaselineReasonWording(t *testing.T) {
	e := Default()
	cases := []struct {
		name string
		kind detect.Kind
		text string
	}{
		{
			name: "prose",
			kind: detect.KindProse,
			text: "I spent the weekend fixing the fence, which had been leaning for years. The posts were rotten at the base, so I dug them out one by one and set new ones in concrete. It took longer than I expected, mostly because the old concrete footings were huge. By Sunday night my back hurt, but the fence stood straight for the first time since we bought the house. Next spring I want to paint it, though my wife thinks we should just let it weather.",
		},
		{
			name: "pr",
			kind: detect.KindPR,
			text: "## Summary\n\nFix a nil pointer dereference in the configuration loader when the file is empty on startup, and return a clear error to the caller instead of crashing during initialization. Tested locally with an empty path and a missing directory.",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := e.models[tc.kind]
			if m == nil {
				t.Fatalf("missing embedded model for %s", tc.kind)
			}
			res, err := e.Analyze(context.Background(), detect.Input{Text: tc.text, Kind: tc.kind})
			if err != nil {
				t.Fatal(err)
			}
			if res.Label == TooShort {
				t.Fatalf("unexpected %q label (%d words)", res.Label, res.WordCount)
			}

			var (
				base, other            Reason
				baseCount, otherCount  int
				shownLogit, totalLogit float64
			)
			for _, r := range res.Reasons {
				if r.Direction == "info" || r.Direction == "warning" || r.Impact == "decisive" {
					continue
				}
				totalLogit += r.Logit
				switch r.Signal {
				case "baseline":
					base = r
					baseCount++
				case "other_signals":
					other = r
					otherCount++
				default:
					shownLogit += r.Logit
				}
			}

			if baseCount != 1 {
				t.Fatalf("got %d baseline reasons, want 1", baseCount)
			}
			if otherCount != 1 {
				t.Fatalf("got %d other_signals reasons, want 1", otherCount)
			}

			wantTotal := math.Log(res.Raw / (100 - res.Raw))
			if math.Abs(totalLogit-wantTotal) > 0.01 {
				t.Errorf("reasons sum to %.4f log-odds, raw score log-odds is %.4f", totalLogit, wantTotal)
			}

			wantDir := func(x float64) string {
				if x > 0 {
					return "AI"
				}
				return "human"
			}
			if base.Impact != "baseline" {
				t.Errorf("baseline.Impact = %q, want %q", base.Impact, "baseline")
			}
			if base.Direction != wantDir(m.Bias) {
				t.Errorf("baseline.Direction = %q, want %q", base.Direction, wantDir(m.Bias))
			}
			if math.Abs(base.Logit-m.Bias) > 1e-12 {
				t.Errorf("baseline.Logit = %v, want m.Bias %v", base.Logit, m.Bias)
			}

			wantOther := wantTotal - m.Bias - shownLogit
			if other.Impact != "other" {
				t.Errorf("other_signals.Impact = %q, want %q", other.Impact, "other")
			}
			if other.Direction != wantDir(other.Logit) {
				t.Errorf("other_signals.Direction = %q, want %q", other.Direction, wantDir(other.Logit))
			}
			if math.Abs(other.Logit-wantOther) > 1e-6 {
				t.Errorf("other_signals.Logit = %v, want %v", other.Logit, wantOther)
			}
			const wantOtherDetail = "all remaining signals combined (each too small to list, or pointing against its typical value)"
			if other.Detail != wantOtherDetail {
				t.Errorf("other_signals.Detail = %q, want %q", other.Detail, wantOtherDetail)
			}

			detail := strings.TrimSpace(base.Detail)
			lower := strings.ToLower(detail)
			for _, forbidden := range []string{"starting point", "would get", "starting probability", "%"} {
				if strings.Contains(lower, forbidden) {
					t.Errorf("baseline.Detail must not claim a starting probability (found %q): %q", forbidden, detail)
				}
			}
			for _, required := range []string{
				"intercept",
				"training",
				"average",
				"mostly ai",
				"reference point",
				"not a prior",
				"score",
				"label",
				"cutoff",
				"calibrated",
				"human",
			} {
				if !strings.Contains(lower, required) {
					t.Errorf("baseline.Detail missing %q: %q", required, detail)
				}
			}

			trimmed := strings.TrimRight(detail, ".!?")
			sentenceBreaks := strings.Count(trimmed, ". ") + strings.Count(trimmed, "! ") + strings.Count(trimmed, "? ")
			if detail == "" || sentenceBreaks > 1 || len(detail) > 260 {
				t.Errorf("baseline.Detail should be 1-2 short sentences (got %d sentences, %d chars): %q", sentenceBreaks+1, len(detail), detail)
			}
		})
	}
}

func startFakeScorerServer(t *testing.T) *httptest.Server {
	t.Helper()
	const modelID = "fake-observer+fake-performer"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model":      modelID,
				"max_tokens": 512,
			})
		case r.Method == http.MethodPost && r.URL.Path == "/score":
			var req struct {
				Texts []string `json:"texts"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
				return
			}
			results := make([]map[string]any, len(req.Texts))
			for i := range req.Texts {
				results[i] = map[string]any{
					"tokens":     64,
					"log_ppl":    1.65,
					"x_ppl":      2.20,
					"binoculars": 0.750,
					"curvature":  1.85,
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model":   modelID,
				"results": results,
			})
		default:
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": "not found"})
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func writeModelJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), append(b, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func findSignal(t *testing.T, sigs []detect.Signal, name string) detect.Signal {
	t.Helper()
	for _, s := range sigs {
		if s.Name == name {
			return s
		}
	}
	t.Fatalf("missing signal %q in %+v", name, sigs)
	return detect.Signal{}
}

// AC2 + AC4: FromDir loads <kind>.lm.ngram.json when that kind is scored by
// <kind>.lm.json (with a reachable scorer), falls back to <kind>.ngram.json
// when <kind>.lm.ngram.json is absent (legacy experiment dirs), always uses
// <kind>.ngram.json when the LM model is not used, keeps kinds independent, and
// hashes only the actually loaded files in BuildInfo().Models.
func TestFromDirLMNgramIsolationAndFallback(t *testing.T) {
	ctx := context.Background()
	srv := startFakeScorerServer(t)

	// >=60-word prose containing "cedar", and >=35-word PR containing "deadlock".
	const proseSample = "Yesterday afternoon I patched three cedar shingles above the back porch overhang, " +
		"swept the pine needles out of the copper gutter, and sealed the corner seam before dusk. " +
		"The old flashing had worked loose during January windstorms, so I replaced two rusted nails, " +
		"checked the attic rafters with a flashlight when the rain started, and stacked the leftover cedar boards inside the shed."
	const prSample = "## Summary\n\nFix a deadlock in the worker shutdown path when closing idle connections after context cancellation, " +
		"and add a deterministic unit test covering concurrent callers waiting on pool drain completion."

	identityThresholds := map[string]Thresholds{
		"prose":    {Mixed: 40, LikelyAI: 60, AI: 80},
		"prose.lm": {Mixed: 40, LikelyAI: 60, AI: 80},
		"pr":       {Mixed: 40, LikelyAI: 60, AI: 80},
		"pr.lm":    {Mixed: 40, LikelyAI: 60, AI: 80},
	}

	regularProseCalib := calibrate.Model{
		Kind: "prose",
		Bias: 0,
		Features: []calibrate.Feature{
			{Name: "word_patterns", Mean: 0, Std: 1, Weight: 1.0, HumanMedian: -2, AIMedian: 2},
		},
		Missing: map[string]float64{},
		Trained: 50,
	}
	lmProseCalib := calibrate.Model{
		Kind: "prose",
		Bias: 0.25,
		Features: []calibrate.Feature{
			{Name: "word_patterns", Mean: 0, Std: 1, Weight: 1.0, HumanMedian: -2, AIMedian: 2},
			{Name: "lm_binoculars", Mean: 0.90, Std: 0.15, Weight: -0.5, HumanMedian: 0.95, AIMedian: 0.75},
		},
		Missing: map[string]float64{},
		Trained: 50,
	}
	regularProseNgram := ngram.Model{
		Kind:    "prose",
		Bias:    -4.5,
		Weights: map[string]float64{"cedar": -8.0},
	}
	lmProseNgram := ngram.Model{
		Kind:    "prose",
		Bias:    4.5,
		Weights: map[string]float64{"cedar": 8.0},
	}

	populateBothProse := func(t *testing.T, regNgram, lmNgram ngram.Model) string {
		t.Helper()
		dir := t.TempDir()
		writeModelJSON(t, dir, "thresholds.json", identityThresholds)
		writeModelJSON(t, dir, "prose.json", regularProseCalib)
		writeModelJSON(t, dir, "prose.lm.json", lmProseCalib)
		writeModelJSON(t, dir, "prose.ngram.json", regNgram)
		writeModelJSON(t, dir, "prose.lm.ngram.json", lmNgram)
		return dir
	}

	t.Run("case1_both_present_with_scorer_uses_lm_ngram_and_hashes_loaded_only", func(t *testing.T) {
		t.Setenv("AIDETECTOR_SCORER_URL", srv.URL)

		dirBase := populateBothProse(t, regularProseNgram, lmProseNgram)
		eBase, err := FromDir(dirBase)
		if err != nil {
			t.Fatalf("FromDir: %v", err)
		}
		res, err := eBase.Analyze(ctx, detect.Input{Text: proseSample, Kind: detect.KindProse})
		if err != nil {
			t.Fatalf("Analyze: %v", err)
		}
		if res.Scorer != "fake-observer+fake-performer" {
			t.Errorf("Scorer = %q, want fake-observer+fake-performer", res.Scorer)
		}
		wp := findSignal(t, res.Signals, "word_patterns")
		if wp.Value <= 4.0 || !strings.Contains(wp.Detail, `AI-leaning: "cedar"`) {
			t.Errorf("expected prose.lm.ngram.json to score word_patterns > 4 with AI-leaning \"cedar\", got Value=%.3f Detail=%q", wp.Value, wp.Detail)
		}
		if res.Score < 95 || res.Label != "AI" {
			t.Errorf("expected high AI score from prose.lm.json + prose.lm.ngram.json, got Score=%d Label=%q", res.Score, res.Label)
		}

		// Mutating loaded prose.lm.ngram.json must change BuildInfo().Models.
		mutatedLMNgram := ngram.Model{Kind: "prose", Bias: 5.25, Weights: map[string]float64{"cedar": 9.5}}
		dirMutLM := populateBothProse(t, regularProseNgram, mutatedLMNgram)
		eMutLM, err := FromDir(dirMutLM)
		if err != nil {
			t.Fatalf("FromDir(mutated lm.ngram): %v", err)
		}
		if eMutLM.BuildInfo().Models == eBase.BuildInfo().Models {
			t.Errorf("BuildInfo().Models did not change when loaded prose.lm.ngram.json changed: %q", eBase.BuildInfo().Models)
		}

		// Mutating unloaded prose.ngram.json must NOT change BuildInfo().Models.
		mutatedRegNgram := ngram.Model{Kind: "prose", Bias: -9.75, Weights: map[string]float64{"cedar": -15.0}}
		dirMutReg := populateBothProse(t, mutatedRegNgram, lmProseNgram)
		eMutReg, err := FromDir(dirMutReg)
		if err != nil {
			t.Fatalf("FromDir(mutated unloaded regular ngram): %v", err)
		}
		if eMutReg.BuildInfo().Models != eBase.BuildInfo().Models {
			t.Errorf("BuildInfo().Models changed when unloaded prose.ngram.json changed: got %q, want %q",
				eMutReg.BuildInfo().Models, eBase.BuildInfo().Models)
		}
	})

	t.Run("case2_both_present_without_scorer_uses_regular_ngram_and_hashes_loaded_only", func(t *testing.T) {
		t.Setenv("AIDETECTOR_SCORER_URL", "")

		dirBase := populateBothProse(t, regularProseNgram, lmProseNgram)
		eBase, err := FromDir(dirBase)
		if err != nil {
			t.Fatalf("FromDir: %v", err)
		}
		res, err := eBase.Analyze(ctx, detect.Input{Text: proseSample, Kind: detect.KindProse})
		if err != nil {
			t.Fatalf("Analyze: %v", err)
		}
		if res.Scorer != "" {
			t.Errorf("Scorer = %q, want empty without scorer env", res.Scorer)
		}
		wp := findSignal(t, res.Signals, "word_patterns")
		if wp.Value >= -4.0 || !strings.Contains(wp.Detail, `human-leaning: "cedar"`) {
			t.Errorf("expected prose.ngram.json to score word_patterns < -4 with human-leaning \"cedar\", got Value=%.3f Detail=%q", wp.Value, wp.Detail)
		}
		if res.Score > 5 || res.Label != "human" {
			t.Errorf("expected low human score from prose.json + prose.ngram.json, got Score=%d Label=%q", res.Score, res.Label)
		}

		// Mutating unloaded prose.lm.ngram.json must NOT change BuildInfo().Models.
		mutatedLMNgram := ngram.Model{Kind: "prose", Bias: 7.5, Weights: map[string]float64{"cedar": 12.0}}
		dirMutLM := populateBothProse(t, regularProseNgram, mutatedLMNgram)
		eMutLM, err := FromDir(dirMutLM)
		if err != nil {
			t.Fatalf("FromDir(mutated unloaded lm.ngram): %v", err)
		}
		if eMutLM.BuildInfo().Models != eBase.BuildInfo().Models {
			t.Errorf("BuildInfo().Models changed when unloaded prose.lm.ngram.json changed: got %q, want %q",
				eMutLM.BuildInfo().Models, eBase.BuildInfo().Models)
		}

		// Mutating loaded prose.ngram.json MUST change BuildInfo().Models.
		mutatedRegNgram := ngram.Model{Kind: "prose", Bias: -6.25, Weights: map[string]float64{"cedar": -11.0}}
		dirMutReg := populateBothProse(t, mutatedRegNgram, lmProseNgram)
		eMutReg, err := FromDir(dirMutReg)
		if err != nil {
			t.Fatalf("FromDir(mutated regular ngram): %v", err)
		}
		if eMutReg.BuildInfo().Models == eBase.BuildInfo().Models {
			t.Errorf("BuildInfo().Models did not change when loaded prose.ngram.json changed: %q", eBase.BuildInfo().Models)
		}
	})

	t.Run("case3_legacy_lm_dir_without_lm_ngram_falls_back_to_regular_ngram", func(t *testing.T) {
		t.Setenv("AIDETECTOR_SCORER_URL", srv.URL)

		dirLegacy := t.TempDir()
		writeModelJSON(t, dirLegacy, "thresholds.json", identityThresholds)
		writeModelJSON(t, dirLegacy, "prose.json", regularProseCalib)
		writeModelJSON(t, dirLegacy, "prose.lm.json", lmProseCalib)
		writeModelJSON(t, dirLegacy, "prose.ngram.json", regularProseNgram)

		eLegacy, err := FromDir(dirLegacy)
		if err != nil {
			t.Fatalf("FromDir(legacy): %v", err)
		}
		res, err := eLegacy.Analyze(ctx, detect.Input{Text: proseSample, Kind: detect.KindProse})
		if err != nil {
			t.Fatalf("Analyze(legacy): %v", err)
		}
		if res.Scorer != "fake-observer+fake-performer" {
			t.Errorf("Scorer = %q, want fake-observer+fake-performer", res.Scorer)
		}
		wp := findSignal(t, res.Signals, "word_patterns")
		if wp.Value >= -4.0 || !strings.Contains(wp.Detail, `human-leaning: "cedar"`) {
			t.Errorf("legacy dir without prose.lm.ngram.json should fall back to prose.ngram.json, got Value=%.3f Detail=%q", wp.Value, wp.Detail)
		}
		sawBinoReason := false
		for _, r := range res.Reasons {
			if r.Signal == "lm_binoculars" {
				sawBinoReason = true
				break
			}
		}
		if !sawBinoReason {
			t.Errorf("expected prose.lm.json lm_binoculars reason in legacy LM dir, got Reasons=%+v", res.Reasons)
		}

		// Both prose.lm.json and fallback prose.ngram.json are loaded and reflected in BuildInfo().Models.
		dirLegacyMutNgram := t.TempDir()
		writeModelJSON(t, dirLegacyMutNgram, "thresholds.json", identityThresholds)
		writeModelJSON(t, dirLegacyMutNgram, "prose.json", regularProseCalib)
		writeModelJSON(t, dirLegacyMutNgram, "prose.lm.json", lmProseCalib)
		writeModelJSON(t, dirLegacyMutNgram, "prose.ngram.json", ngram.Model{Kind: "prose", Bias: -5.5, Weights: map[string]float64{"cedar": -9.0}})
		eLegacyMutNgram, err := FromDir(dirLegacyMutNgram)
		if err != nil {
			t.Fatal(err)
		}
		if eLegacyMutNgram.BuildInfo().Models == eLegacy.BuildInfo().Models {
			t.Errorf("BuildInfo().Models did not include fallback prose.ngram.json: %q", eLegacy.BuildInfo().Models)
		}

		dirLegacyMutLM := t.TempDir()
		writeModelJSON(t, dirLegacyMutLM, "thresholds.json", identityThresholds)
		writeModelJSON(t, dirLegacyMutLM, "prose.json", regularProseCalib)
		mutLMCalib := lmProseCalib
		mutLMCalib.Bias = 0.55
		writeModelJSON(t, dirLegacyMutLM, "prose.lm.json", mutLMCalib)
		writeModelJSON(t, dirLegacyMutLM, "prose.ngram.json", regularProseNgram)
		eLegacyMutLM, err := FromDir(dirLegacyMutLM)
		if err != nil {
			t.Fatal(err)
		}
		if eLegacyMutLM.BuildInfo().Models == eLegacy.BuildInfo().Models {
			t.Errorf("BuildInfo().Models did not include prose.lm.json: %q", eLegacy.BuildInfo().Models)
		}
	})

	t.Run("case4_per_kind_independence_prose_lm_while_pr_has_no_lm_json", func(t *testing.T) {
		t.Setenv("AIDETECTOR_SCORER_URL", srv.URL)

		regularPRCalib := calibrate.Model{
			Kind: "pr",
			Bias: 0,
			Features: []calibrate.Feature{
				{Name: "word_patterns", Mean: 0, Std: 1, Weight: 1.0, HumanMedian: -2, AIMedian: 2},
			},
			Missing: map[string]float64{},
			Trained: 40,
		}
		regularPRNgram := ngram.Model{
			Kind:    "pr",
			Bias:    -4.5,
			Weights: map[string]float64{"deadlock": -8.0},
		}
		orphanPRLMNgram := ngram.Model{
			Kind:    "pr",
			Bias:    4.5,
			Weights: map[string]float64{"deadlock": 8.0},
		}

		dirMixed := populateBothProse(t, regularProseNgram, lmProseNgram)
		writeModelJSON(t, dirMixed, "pr.json", regularPRCalib)
		writeModelJSON(t, dirMixed, "pr.ngram.json", regularPRNgram)

		eMixed, err := FromDir(dirMixed)
		if err != nil {
			t.Fatalf("FromDir(mixed): %v", err)
		}

		resProse, err := eMixed.Analyze(ctx, detect.Input{Text: proseSample, Kind: detect.KindProse})
		if err != nil {
			t.Fatalf("Analyze(prose): %v", err)
		}
		wpProse := findSignal(t, resProse.Signals, "word_patterns")
		if wpProse.Value <= 4.0 || resProse.Score < 95 || eMixed.ThresholdKey(detect.KindProse) != "prose.lm" {
			t.Errorf("prose should use prose.lm.json + prose.lm.ngram.json: wp=%.3f Score=%d key=%q",
				wpProse.Value, resProse.Score, eMixed.ThresholdKey(detect.KindProse))
		}

		resPR, err := eMixed.Analyze(ctx, detect.Input{Text: prSample, Kind: detect.KindPR})
		if err != nil {
			t.Fatalf("Analyze(pr): %v", err)
		}
		wpPR := findSignal(t, resPR.Signals, "word_patterns")
		if wpPR.Value >= -4.0 || resPR.Score > 5 || eMixed.ThresholdKey(detect.KindPR) != "pr" {
			t.Errorf("pr without pr.lm.json should use pr.json + pr.ngram.json: wp=%.3f Score=%d key=%q",
				wpPR.Value, resPR.Score, eMixed.ThresholdKey(detect.KindPR))
		}

		// Even if an unused pr.lm.ngram.json sits in the directory without pr.lm.json, PR must still use pr.ngram.json and ignore pr.lm.ngram.json.
		dirWithOrphan := populateBothProse(t, regularProseNgram, lmProseNgram)
		writeModelJSON(t, dirWithOrphan, "pr.json", regularPRCalib)
		writeModelJSON(t, dirWithOrphan, "pr.ngram.json", regularPRNgram)
		writeModelJSON(t, dirWithOrphan, "pr.lm.ngram.json", orphanPRLMNgram)

		eOrphan, err := FromDir(dirWithOrphan)
		if err != nil {
			t.Fatalf("FromDir(orphan pr.lm.ngram.json): %v", err)
		}
		resPROrphan, err := eOrphan.Analyze(ctx, detect.Input{Text: prSample, Kind: detect.KindPR})
		if err != nil {
			t.Fatalf("Analyze(pr orphan): %v", err)
		}
		wpPROrphan := findSignal(t, resPROrphan.Signals, "word_patterns")
		if wpPROrphan.Value >= -4.0 || resPROrphan.Score > 5 {
			t.Errorf("pr without pr.lm.json must ignore pr.lm.ngram.json and use pr.ngram.json: wp=%.3f Score=%d",
				wpPROrphan.Value, resPROrphan.Score)
		}
		if eOrphan.BuildInfo().Models != eMixed.BuildInfo().Models {
			t.Errorf("unloaded pr.lm.ngram.json (when pr.lm.json is absent) changed BuildInfo().Models: got %q, want %q",
				eOrphan.BuildInfo().Models, eMixed.BuildInfo().Models)
		}
	})
}
