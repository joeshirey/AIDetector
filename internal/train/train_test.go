package train_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/split"
	"github.com/joeshirey/AIDetector/internal/train"
)

func writeTrainFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestShareAndFitCutoffsGolden(t *testing.T) {
	if got := train.Share(nil, 50); got != 0 {
		t.Fatalf("Share(nil, 50) = %g, want 0", got)
	}
	// 20 calibration scores: 10..19 (10 items), 45 (5 items), 68 (3 items), 84 (1 item), 91 (1 item).
	calib := []float64{
		10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
		45, 45, 45, 45, 45,
		68, 68, 68,
		84,
		91,
	}
	check := []float64{12, 30, 50, 70, 86, 95, 20, 25, 35, 40}

	// Allowed shares: mixed=0.25 (<=5/20 -> 46, since >=45 has 10/20=0.50 and >=46 has 5/20=0.25),
	// likely=0.10 (<=2/20 -> 69, since >=68 has 5/20=0.25 and >=69 has 2/20=0.10),
	// ai=0.05 (<=1/20 -> 85, since >=84 has 2/20=0.10 and >=85 has 1/20=0.05).
	got := train.FitCutoffs(calib, check, 0.25, 0.10, 0.05)
	if got.Mixed != 46 || got.LikelyAI != 69 || got.AI != 85 || got.CalibratedOn != 20 {
		t.Fatalf("FitCutoffs = %+v, want Mixed=46 LikelyAI=69 AI=85 CalibratedOn=20", got)
	}
	// On check (10 items): >=46 has {50, 70, 86, 95} = 0.4; >=69 has {70, 86, 95} = 0.3; >=85 has {86, 95} = 0.2.
	if math.Abs(got.HumanAtMixed-0.4) > 1e-9 || math.Abs(got.HumanAtLikelyAI-0.3) > 1e-9 || math.Abs(got.HumanAtAI-0.2) > 1e-9 {
		t.Errorf("check shares = (%g, %g, %g), want (0.4, 0.3, 0.2)", got.HumanAtMixed, got.HumanAtLikelyAI, got.HumanAtAI)
	}

	// Strictly monotonic floor invariant even when allowed shares are identical.
	mono := train.FitCutoffs([]float64{5, 5, 5, 5}, []float64{5}, 0.5, 0.5, 0.5)
	if !(mono.Mixed < mono.LikelyAI && mono.LikelyAI < mono.AI) {
		t.Errorf("expected strictly increasing cutoffs Mixed < LikelyAI < AI, got %+v", mono)
	}
}

func TestFoldKeepsVariantsTogether(t *testing.T) {
	for i := range 50 {
		base := fmt.Sprintf("sample-%03d", i)
		want := train.Fold(base, 5)
		if want < 0 || want >= 5 {
			t.Fatalf("Fold(%q, 5) = %d out of [0,5)", base, want)
		}
		for _, suffix := range []string{"~plain", "~persona", "~continue"} {
			if got := train.Fold(base+suffix, 5); got != want {
				t.Errorf("Fold(%q, 5) = %d, want %d", base+suffix, got, want)
			}
		}
	}
}

func TestFitWithWeightingAndThresholdsEndToEnd(t *testing.T) {
	ctx := context.Background()
	corpusDir := filepath.Join(t.TempDir(), "prose_corpus")
	trainIDs := []string{"sample-001", "sample-002", "sample-003", "sample-004", "sample-005", "sample-006"}

	for i, id := range trainIDs {
		human := fmt.Sprintf(
			"Yesterday morning I carried the wooden ladder out behind the barn, scraped loose paint off the window frame, and replaced two cracked panes before noon (%d). "+
				"The putty knife left linseed oil on my sleeves while sparrows nested under the eaves and rain clouds gathered over the ridge. "+
				"After lunch we stacked split oak by the woodshed, swept the gravel path, and brewed a pot of strong coffee in the kitchen.\n",
			i,
		)
		ai := fmt.Sprintf(
			"Modern cloud platforms leverage comprehensive observability, deterministic orchestration, and automated governance to streamline distributed workloads (%d). "+
				"Furthermore, standardized deployment pipelines strengthen operational resilience, eliminate configuration drift, and ensure scalable reliability across hybrid clusters. "+
				"Organizations that align telemetry metrics with architectural guardrails consistently accelerate delivery velocity.\n",
			i,
		)
		writeTrainFile(t, filepath.Join(corpusDir, "human", id+".txt"), human)
		writeTrainFile(t, filepath.Join(corpusDir, "ai", "claude-sonnet-5-5", id+".txt"), ai)
		if i < 3 {
			writeTrainFile(t, filepath.Join(corpusDir, "ai", "claude-sonnet-5-5", id+"~persona.txt"), ai+" Additional persona phrasing.\n")
			writeTrainFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", id+".txt"), ai+" Additional gemini phrasing.\n")
		}
	}
	// Test-split sample-007 must be ignored by FitWith (which fits on split == "train" only).
	writeTrainFile(t, filepath.Join(corpusDir, "human", "sample-007.txt"), strings.Repeat("quiet porch morning note ", 25))
	writeTrainFile(t, filepath.Join(corpusDir, "ai", "claude-sonnet-5-5", "sample-007.txt"), strings.Repeat("scalable telemetry framework ", 25))

	outDir := filepath.Join(t.TempDir(), "cand")
	opts, err := train.ReleaseFitOptions(outDir)
	if err != nil {
		t.Fatalf("ReleaseFitOptions: %v", err)
	}
	res, err := train.FitWith(ctx, detect.KindProse, []string{corpusDir}, outDir, opts)
	if err != nil {
		t.Fatalf("FitWith(ReleaseFitOptions): %v", err)
	}
	if res.Human != 6 || res.AI != 12 {
		t.Fatalf("FitResult counts = (%d human, %d ai), want (6, 12)", res.Human, res.AI)
	}
	for _, p := range []string{res.ModelPath, res.NgramPath} {
		if st, err := os.Stat(p); err != nil || st.Size() == 0 {
			t.Errorf("expected non-empty output file %s (err=%v)", p, err)
		}
	}

	// Family+balance-family weighting must also fit cleanly and preserve 1:1 human-vs-AI total weight balance.
	famDir := filepath.Join(t.TempDir(), "cand_family")
	famOpts := opts
	famOpts.Weighting = "family+balance-family"
	famRes, err := train.FitWith(ctx, detect.KindProse, []string{corpusDir}, famDir, famOpts)
	if err != nil {
		t.Fatalf("FitWith(family+balance-family): %v", err)
	}
	if famRes.Model.Trained != 18 {
		t.Errorf("family+balance-family Trained = %d, want 18", famRes.Model.Trained)
	}

	// Load candidate engine from outDir and calibrate both document and segment thresholds on >400-word human pool docs.
	cand, err := engine.FromDir(outDir)
	if err != nil {
		t.Fatalf("engine.FromDir: %v", err)
	}
	poolDir := filepath.Join(t.TempDir(), "pool")
	longHuman := strings.Repeat(
		"Last Saturday we walked along the creek trail behind the orchard, repaired the wooden footbridge, "+
			"cleared fallen alder branches from the culvert, and watched trout dart under the mossy stones. ",
		18, // ~500+ words -> >= 2 segments of 200 words
	)
	// Include both calib (sample-002, sample-008) and check (sample-001, sample-004) non-holdout IDs, plus holdout sample-015.
	for _, id := range []string{"sample-002", "sample-008", "sample-001", "sample-004", "sample-015"} {
		writeTrainFile(t, filepath.Join(poolDir, id+".txt"), longHuman)
	}

	threshPath := filepath.Join(outDir, "thresholds.json")
	keepNoHoldout := func(_, id string) bool { return !split.FinalHoldout(id) }
	tr, err := train.Thresholds(ctx, cand, detect.KindProse, []string{poolDir}, threshPath, train.DefaultThresholdOptions(), keepNoHoldout)
	if err != nil {
		t.Fatalf("Thresholds: %v", err)
	}
	if tr.Doc == nil || tr.Doc.CalibratedOn != 2 || tr.DocChecked != 2 {
		t.Errorf("Doc ThresholdReport = (%+v, checked=%d), want CalibratedOn=2 DocChecked=2 (holdout excluded)", tr.Doc, tr.DocChecked)
	}
	if tr.Segment == nil || tr.Segment.CalibratedOn < 4 || tr.SegChecked < 4 {
		t.Errorf("Segment ThresholdReport = (%+v, checked=%d), want >=4 calib and >=4 check segments", tr.Segment, tr.SegChecked)
	}
}
