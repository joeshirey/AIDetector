package eval_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/eval"
)

type markerDetector struct{}

func (markerDetector) Name() string { return "marker" }

func (markerDetector) Detect(_ context.Context, in detect.Input) ([]detect.Signal, error) {
	p := 0.5
	for _, tok := range strings.Fields(in.Text) {
		if v, ok := strings.CutPrefix(tok, "PROB="); ok {
			if parsed, err := strconv.ParseFloat(v, 64); err == nil {
				p = parsed
			}
		}
	}
	return []detect.Signal{{
		Name:     "marker_prob",
		Detector: "marker",
		Value:    p,
		AIProb:   p,
		Weight:   1.0,
		Detail:   "synthetic probability marker",
	}}, nil
}

func makePRDoc(prob float64, tamper bool) string {
	prefix := ""
	if tamper {
		prefix = "\u200b" // triggers sanitizer unicode_tampering signal, which eval must omit from Sample.Feats
	}
	// >= 150 words so KindPR lengthConfidence == 1.0 and raw score equals 100 * prob.
	filler := strings.Repeat("steady engineering pull request context word ", 28)
	return fmt.Sprintf("%sPROB=%.2f %s\n", prefix, prob, filler)
}

func writeEvalFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestRunGoldenMetricsSplitKeepAndSubset(t *testing.T) {
	dir := t.TempDir()
	e := engine.New(markerDetector{})

	// Human files: 3 train-split (sample-001..003) + 1 test-split (sample-007, .md extension).
	writeEvalFile(t, filepath.Join(dir, "human", "sample-001.txt"), makePRDoc(0.10, true))
	writeEvalFile(t, filepath.Join(dir, "human", "sample-002.txt"), makePRDoc(0.30, false))
	writeEvalFile(t, filepath.Join(dir, "human", "sample-003.txt"), makePRDoc(0.60, false))
	writeEvalFile(t, filepath.Join(dir, "human", "sample-007.md"), makePRDoc(0.20, false))
	// Non-.txt/.md metadata must be ignored.
	writeEvalFile(t, filepath.Join(dir, "human", "sample-001.json"), `{"ignored":true}`)

	// AI files across subdirectories: 3 train-split + 1 test-split (.md).
	writeEvalFile(t, filepath.Join(dir, "ai", "gen-a", "sample-001.txt"), makePRDoc(0.90, false))
	writeEvalFile(t, filepath.Join(dir, "ai", "gen-a", "sample-002.txt"), makePRDoc(0.70, false))
	writeEvalFile(t, filepath.Join(dir, "ai", "gen-b", "sample-003.txt"), makePRDoc(0.60, false)) // ties human sample-003
	writeEvalFile(t, filepath.Join(dir, "ai", "gen-b", "sample-007.md"), makePRDoc(0.40, false))

	rep, err := eval.Run(context.Background(), e, []string{dir}, detect.KindPR, "")
	if err != nil {
		t.Fatalf("eval.Run error: %v", err)
	}
	if rep.Human != 4 || rep.AI != 4 {
		t.Fatalf("counts = (%d human, %d ai), want (4, 4)", rep.Human, rep.AI)
	}

	// Pairwise wins: 90->4, 70->4, 60->3.5 (tie with human 60), 40->3 => 14.5 / 16 = 0.90625.
	if math.Abs(rep.AUROC-0.90625) > 1e-9 {
		t.Errorf("AUROC = %.6f, want 0.90625", rep.AUROC)
	}
	if math.Abs(rep.TPRAt50-0.75) > 1e-9 {
		t.Errorf("TPRAt50 = %g, want 0.75 (3 of 4 AI >= 50)", rep.TPRAt50)
	}
	if math.Abs(rep.FPRAt50-0.25) > 1e-9 {
		t.Errorf("FPRAt50 = %g, want 0.25 (1 of 4 human >= 50)", rep.FPRAt50)
	}
	if rep.Threshold1 != 61 {
		t.Errorf("Threshold1 = %d, want 61 (lowest integer threshold above max human score 60)", rep.Threshold1)
	}
	if math.Abs(rep.TPRAtThresh-0.50) > 1e-9 || math.Abs(rep.Recall1-0.50) > 1e-9 {
		t.Errorf("TPRAtThresh=%g Recall1=%g, want both 0.50", rep.TPRAtThresh, rep.Recall1)
	}

	// Samples must be sorted descending by Score, and sanitizer signals must be omitted from Feats.
	for i := 1; i < len(rep.Samples); i++ {
		if rep.Samples[i].Score > rep.Samples[i-1].Score {
			t.Errorf("Samples not sorted descending at %d: %d > %d", i, rep.Samples[i].Score, rep.Samples[i-1].Score)
		}
	}
	for _, s := range rep.Samples {
		if _, hasSanitizer := s.Feats["unicode_tampering"]; hasSanitizer {
			t.Errorf("sample %s Feats included sanitizer signal unicode_tampering: %v", s.ID, s.Feats)
		}
		if _, ok := s.Feats["marker_prob"]; !ok {
			t.Errorf("sample %s missing marker_prob in Feats: %v", s.ID, s.Feats)
		}
	}

	// Split filtering: test split holds only sample-007 (human 20 vs AI 40 -> AUROC 1.0).
	testRep, err := eval.Run(context.Background(), e, []string{dir}, detect.KindPR, "test")
	if err != nil {
		t.Fatalf("eval.Run(test) error: %v", err)
	}
	if testRep.Human != 1 || testRep.AI != 1 || testRep.AUROC != 1.0 {
		t.Errorf("test split report = (H=%d, AI=%d, AUROC=%g), want (1, 1, 1.0)", testRep.Human, testRep.AI, testRep.AUROC)
	}

	// Subset recomputes metrics over kept samples without rescoring.
	subGenA := rep.Subset(func(path, id string) bool {
		return !strings.Contains(filepath.ToSlash(path), "/gen-b/")
	})
	if subGenA.Human != 4 || subGenA.AI != 2 || subGenA.AUROC != 1.0 || subGenA.Recall1 != 1.0 {
		t.Errorf("Subset(gen-a) = (H=%d, AI=%d, AUROC=%g, Recall1=%g), want (4, 2, 1.0, 1.0)",
			subGenA.Human, subGenA.AI, subGenA.AUROC, subGenA.Recall1)
	}

	// RunFiltered returning zero AI samples must report an error.
	if _, err := eval.RunFiltered(context.Background(), e, []string{dir}, detect.KindPR, "", func(path, id string) bool {
		return strings.Contains(filepath.ToSlash(path), "/human/")
	}); err == nil {
		t.Errorf("expected error when keep filters out all AI samples, got nil")
	}
}
