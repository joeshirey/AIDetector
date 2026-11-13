package ngram_test

import (
	"context"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/ngram"
)

func TestTermsStraightensQuotesAndHandlesFWTrigrams(t *testing.T) {
	input := "“It’s” in the same place, which is fine; it's in order to test!"

	base := ngram.Terms(input)
	// Curly apostrophe and ASCII apostrophe both become "it's" and deduplicate.
	countIts := 0
	for _, tok := range base {
		if tok == "it's" {
			countIts++
		}
	}
	if countIts != 1 {
		t.Errorf("Terms(%q) emitted \"it's\" %d times, want 1 (curly quotes straightened & deduplicated): %v", input, countIts, base)
	}
	for _, want := range []string{"it's", "in the", ", which", "which is", ";", "!"} {
		if !slices.Contains(base, want) {
			t.Errorf("Terms missing expected unigram/bigram %q in %v", want, base)
		}
	}
	for _, trigram := range []string{"in the same", ", which is"} {
		if slices.Contains(base, trigram) {
			t.Errorf("Terms(..., false) unexpectedly emitted trigram %q", trigram)
		}
	}

	withTri := ngram.TermsWith(input, true)
	for _, wantTri := range []string{"in the same", ", which is"} {
		if !slices.Contains(withTri, wantTri) {
			t.Errorf("TermsWith(..., true) missing function-word trigram %q in %v", wantTri, withTri)
		}
	}
	// Content nouns ("place", "order", "test") must never form function-word trigrams.
	for _, nonFWTri := range []string{"the same place", "in order to", "order to test"} {
		if slices.Contains(withTri, nonFWTri) {
			t.Errorf("TermsWith(..., true) emitted content-word trigram %q", nonFWTri)
		}
	}
}

func TestModelLogitUnitLengthScaling(t *testing.T) {
	m := &ngram.Model{
		Kind: "prose",
		Bias: -0.25,
		Weights: map[string]float64{
			"alpha":      2.0,
			"alpha beta": 1.0,
			"gamma":      -1.5,
		},
	}

	// No known terms -> returns Bias and empty map.
	zNone, cNone := m.Logit("delta epsilon")
	if zNone != -0.25 || len(cNone) != 0 {
		t.Fatalf("Logit(unknown) = (%g, %v), want (-0.25, empty)", zNone, cNone)
	}

	// "Alpha beta delta." -> normalized tokens: ["alpha", "beta", "delta", "."]
	// Distinct terms (7): "alpha", "beta", "alpha beta", "delta", "beta delta", ".", "delta ."
	// Scale = 1 / sqrt(7). Known weights: "alpha" (2.0) and "alpha beta" (1.0) -> sum = 3.0 / sqrt(7).
	z, contrib := m.Logit("Alpha beta delta.")
	wantScale := 1 / math.Sqrt(7)
	wantZ := -0.25 + 3.0*wantScale
	if math.Abs(z-wantZ) > 1e-12 {
		t.Errorf("Logit = %.12f, want %.12f", z, wantZ)
	}
	if len(contrib) != 2 || math.Abs(contrib["alpha"]-2.0*wantScale) > 1e-12 || math.Abs(contrib["alpha beta"]-1.0*wantScale) > 1e-12 {
		t.Errorf("contrib = %v, want alpha=%g and alpha beta=%g", contrib, 2.0*wantScale, 1.0*wantScale)
	}
}

func TestTrainDeterminismRegularizationAndDetector(t *testing.T) {
	docs := []string{
		"porch cedar hammer sawdust lantern garden kettle Sparrow1",
		"porch cedar hammer sawdust lantern creek workbench Sparrow2",
		"porch cedar hammer orchard lantern skillet timber Sparrow3",
		"comprehensive framework streamlined governance scalable telemetry",
		"comprehensive framework streamlined governance resilient orchestration",
		"comprehensive framework streamlined governance deterministic pipeline",
	}
	isAI := []bool{false, false, false, true, true, true}

	opts := ngram.TrainOptions{MinDocs: 2, MaxTerms: 12, Epochs: 40, L2: 1e-3, Seed: 7}
	m1 := ngram.Train("prose", docs, isAI, opts)
	m2 := ngram.Train("prose", docs, isAI, opts)

	if m1.Bias != m2.Bias || !maps.Equal(m1.Weights, m2.Weights) {
		t.Fatalf("Train with fixed Seed was not deterministic: (%g, %v) vs (%g, %v)", m1.Bias, m1.Weights, m2.Bias, m2.Weights)
	}
	if len(m1.Weights) > opts.MaxTerms {
		t.Errorf("len(Weights) = %d, exceeds MaxTerms=%d", len(m1.Weights), opts.MaxTerms)
	}
	// Singleton token "sparrow1" (df=1 < MinDocs=2) must be excluded.
	if _, ok := m1.Weights["sparrow1"]; ok {
		t.Errorf("singleton term \"sparrow1\" kept despite MinDocs=2")
	}

	zHuman, _ := m1.Logit("porch cedar hammer sawdust lantern")
	zAI, _ := m1.Logit("comprehensive framework streamlined governance")
	if !(zHuman < -1.0 && zAI > 1.0) {
		t.Errorf("expected clear separation: zHuman=%g (want < -1), zAI=%g (want > 1)", zHuman, zAI)
	}

	// Strong DenseL2 shrinks L2 norm of weights compared to sparse L2 on rare terms.
	sparseOpts := ngram.TrainOptions{MinDocs: 1, MaxTerms: 100, Epochs: 30, L2: 0.05, Seed: 7, DenseL2: false}
	denseOpts := ngram.TrainOptions{MinDocs: 1, MaxTerms: 100, Epochs: 30, L2: 0.05, Seed: 7, DenseL2: true}
	mSparse := ngram.Train("prose", docs, isAI, sparseOpts)
	mDense := ngram.Train("prose", docs, isAI, denseOpts)
	if !(math.Abs(mDense.Weights["sparrow1"]) < math.Abs(mSparse.Weights["sparrow1"])) {
		t.Errorf("expected DenseL2 to shrink rare term \"sparrow1\" more than sparse L2: dense=%g vs sparse=%g",
			mDense.Weights["sparrow1"], mSparse.Weights["sparrow1"])
	}

	// Detector contract: emits word_patterns for configured kind and nil for unconfigured kind.
	det := ngram.NewDetector(map[detect.Kind]*ngram.Model{detect.KindProse: m1})
	if det.Name() != "ngram" {
		t.Fatalf("Detector.Name() = %q, want \"ngram\"", det.Name())
	}
	none, err := det.Detect(context.Background(), detect.Input{Text: "comprehensive framework", Kind: detect.KindPR})
	if err != nil || len(none) != 0 {
		t.Fatalf("Detect(unconfigured KindPR) = (%v, %v), want (nil, nil)", none, err)
	}

	mixed := "comprehensive framework porch cedar"
	wantZ, _ := m1.Logit(mixed)
	sigs, err := det.Detect(context.Background(), detect.Input{Text: mixed, Kind: detect.KindProse})
	if err != nil || len(sigs) != 1 {
		t.Fatalf("Detect(KindProse) = (%v, %v), want 1 signal", sigs, err)
	}
	s := sigs[0]
	wantProb := 1 / (1 + math.Exp(-wantZ))
	if s.Name != "word_patterns" || s.Detector != "ngram" || s.Weight != 1.5 ||
		math.Abs(s.Value-wantZ) > 1e-12 || math.Abs(s.AIProb-wantProb) > 1e-12 {
		t.Errorf("word_patterns signal = %+v, want Value=%g AIProb=%g Weight=1.5", s, wantZ, wantProb)
	}
	if !strings.Contains(s.Detail, "AI-leaning:") || !strings.Contains(s.Detail, "human-leaning:") {
		t.Errorf("Detail = %q, want both AI-leaning and human-leaning explanations", s.Detail)
	}
}
