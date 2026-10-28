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
