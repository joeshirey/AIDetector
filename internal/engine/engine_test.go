package engine

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/detect"
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
