package train_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

func startFakeScorer(t *testing.T) *httptest.Server {
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
			for i, txt := range req.Texts {
				logPPL, xPPL, bino, curv := 2.85, 3.05, 0.934, 0.32
				if strings.Contains(txt, "telemetry") || strings.Contains(txt, "comprehensive") {
					logPPL, xPPL, bino, curv = 1.42, 2.12, 0.670, 2.88
				}
				results[i] = map[string]any{
					"tokens":     64,
					"log_ppl":    logPPL,
					"x_ppl":      xPPL,
					"binoculars": bino,
					"curvature":  curv,
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

// AC1 + AC4: With WithLM: true, FitWith writes <kind>.lm.json and
// <kind>.lm.ngram.json, sets FitResult.NgramPath accordingly, and leaves any
// pre-existing <kind>.ngram.json byte-identical. Without WithLM, FitWith writes
// <kind>.json and <kind>.ngram.json and never writes <kind>.lm.ngram.json.
func TestFitWithLMNgramIsolation(t *testing.T) {
	ctx := context.Background()
	srv := startFakeScorer(t)
	t.Setenv("AIDETECTOR_SCORER_URL", srv.URL)

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
	}

	// 1. WithLM: false writes prose.json + prose.ngram.json and does not write prose.lm.json or prose.lm.ngram.json.
	noLMDir := filepath.Join(t.TempDir(), "no_lm_out")
	resNoLM, err := train.FitWith(ctx, detect.KindProse, []string{corpusDir}, noLMDir, train.FitOptions{
		WithLM:        false,
		SignConstrain: true,
	})
	if err != nil {
		t.Fatalf("FitWith(WithLM: false): %v", err)
	}
	wantRegModel := filepath.Join(noLMDir, "prose.json")
	wantRegNgram := filepath.Join(noLMDir, "prose.ngram.json")
	if resNoLM.ModelPath != wantRegModel {
		t.Errorf("WithLM=false ModelPath = %q, want %q", resNoLM.ModelPath, wantRegModel)
	}
	if resNoLM.NgramPath != wantRegNgram {
		t.Errorf("WithLM=false NgramPath = %q, want %q", resNoLM.NgramPath, wantRegNgram)
	}
	for _, mustExist := range []string{wantRegModel, wantRegNgram} {
		if st, err := os.Stat(mustExist); err != nil || st.Size() == 0 {
			t.Errorf("WithLM=false expected non-empty %s (err=%v)", mustExist, err)
		}
	}
	for _, mustNotExist := range []string{
		filepath.Join(noLMDir, "prose.lm.json"),
		filepath.Join(noLMDir, "prose.lm.ngram.json"),
	} {
		if _, err := os.Stat(mustNotExist); !os.IsNotExist(err) {
			t.Errorf("WithLM=false unexpectedly created %s (err=%v)", mustNotExist, err)
		}
	}

	// 2. WithLM: true writes prose.lm.json + prose.lm.ngram.json and leaves pre-existing prose.ngram.json byte-identical.
	lmDir := filepath.Join(t.TempDir(), "lm_out")
	sentinelNgram := []byte("{\n \"kind\": \"prose\",\n \"bias\": -0.4321,\n \"weights\": {\n  \"sentinel_untouched_term\": -1.75\n }\n}\n")
	sentinelModel := []byte("{\n \"kind\": \"prose\",\n \"bias\": -0.1111,\n \"features\": [],\n \"missing\": {},\n \"trained_on\": 7\n}\n")
	preNgramPath := filepath.Join(lmDir, "prose.ngram.json")
	preModelPath := filepath.Join(lmDir, "prose.json")
	writeTrainFile(t, preNgramPath, string(sentinelNgram))
	writeTrainFile(t, preModelPath, string(sentinelModel))

	resLM, err := train.FitWith(ctx, detect.KindProse, []string{corpusDir}, lmDir, train.FitOptions{
		WithLM:        true,
		SignConstrain: true,
	})
	if err != nil {
		t.Fatalf("FitWith(WithLM: true): %v", err)
	}

	wantLMModel := filepath.Join(lmDir, "prose.lm.json")
	wantLMNgram := filepath.Join(lmDir, "prose.lm.ngram.json")
	if resLM.ModelPath != wantLMModel {
		t.Errorf("WithLM=true ModelPath = %q, want %q", resLM.ModelPath, wantLMModel)
	}
	if resLM.NgramPath != wantLMNgram {
		t.Errorf("WithLM=true NgramPath = %q, want %q", resLM.NgramPath, wantLMNgram)
	}

	lmModelBytes, err := os.ReadFile(wantLMModel)
	if err != nil {
		t.Fatalf("expected %s to exist: %v", wantLMModel, err)
	}
	var parsedLM calibrate.Model
	if err := json.Unmarshal(lmModelBytes, &parsedLM); err != nil {
		t.Fatalf("unmarshal %s: %v", wantLMModel, err)
	}
	hasLMFeat := false
	for _, f := range parsedLM.Features {
		if strings.HasPrefix(f.Name, "lm_") {
			hasLMFeat = true
			break
		}
	}
	if !hasLMFeat {
		t.Errorf("expected %s to contain lm_* features, got %+v", wantLMModel, parsedLM.Features)
	}

	lmNgramBytes, err := os.ReadFile(wantLMNgram)
	if err != nil {
		t.Errorf("expected %s to exist: %v", wantLMNgram, err)
	} else {
		var parsedNgram ngram.Model
		if err := json.Unmarshal(lmNgramBytes, &parsedNgram); err != nil {
			t.Errorf("unmarshal %s: %v", wantLMNgram, err)
		} else if parsedNgram.Kind != "prose" || len(parsedNgram.Weights) == 0 {
			t.Errorf("unexpected %s content: Kind=%q len(Weights)=%d", wantLMNgram, parsedNgram.Kind, len(parsedNgram.Weights))
		}
	}

	gotPreNgram, err := os.ReadFile(preNgramPath)
	if err != nil {
		t.Fatalf("read pre-existing %s: %v", preNgramPath, err)
	}
	if !bytes.Equal(gotPreNgram, sentinelNgram) {
		t.Errorf("WithLM=true modified pre-existing prose.ngram.json:\ngot:  %s\nwant: %s", gotPreNgram, sentinelNgram)
	}
	gotPreModel, err := os.ReadFile(preModelPath)
	if err != nil {
		t.Fatalf("read pre-existing %s: %v", preModelPath, err)
	}
	if !bytes.Equal(gotPreModel, sentinelModel) {
		t.Errorf("WithLM=true modified pre-existing prose.json:\ngot:  %s\nwant: %s", gotPreModel, sentinelModel)
	}
}
