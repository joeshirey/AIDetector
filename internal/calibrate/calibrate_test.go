package calibrate_test

import (
	"math"
	"slices"
	"testing"

	"github.com/joeshirey/AIDetector/internal/calibrate"
)

func TestFitProbAndContributionsLogOddsIdentity(t *testing.T) {
	rows := []calibrate.Row{
		{Feats: map[string]float64{"word_patterns": -2.4, "em_dashes": 8.0, "const_feat": 5.0}, Words: 180, IsAI: false},
		{Feats: map[string]float64{"word_patterns": -2.0, "em_dashes": 6.0, "const_feat": 5.0}, Words: 220, IsAI: false},
		{Feats: map[string]float64{"word_patterns": -1.8, "const_feat": 5.0}, Words: 140, IsAI: false}, // em_dashes missing on human
		{Feats: map[string]float64{"word_patterns": -2.2, "const_feat": 5.0}, Words: 260, IsAI: false},
		{Feats: map[string]float64{"word_patterns": 2.3, "em_dashes": 0.5, "const_feat": 5.0}, Words: 200, IsAI: true},
		{Feats: map[string]float64{"word_patterns": 2.6, "em_dashes": 1.0, "const_feat": 5.0}, Words: 210, IsAI: true},
		{Feats: map[string]float64{"word_patterns": 1.9, "em_dashes": 0.0, "const_feat": 5.0}, Words: 190, IsAI: true},
		{Feats: map[string]float64{"word_patterns": 2.1, "em_dashes": 0.2, "const_feat": 5.0}, Words: 230, IsAI: true},
	}

	m := calibrate.Fit("prose", rows)
	if m.Kind != "prose" {
		t.Fatalf("Kind = %q, want \"prose\"", m.Kind)
	}
	if m.Trained != len(rows) {
		t.Fatalf("Trained = %d, want %d", m.Trained, len(rows))
	}

	var names []string
	byName := map[string]calibrate.Feature{}
	for _, f := range m.Features {
		names = append(names, f.Name)
		byName[f.Name] = f
	}
	wantNames := []string{"const_feat", "em_dashes", "log_words", "word_patterns"}
	if !slices.Equal(names, wantNames) {
		t.Fatalf("feature names = %v, want sorted %v", names, wantNames)
	}

	// Constant feature must fall back to Std == 1 instead of zero/NaN.
	if cf := byName["const_feat"]; cf.Mean != 5.0 || cf.Std != 1.0 {
		t.Errorf("const_feat Mean=%g Std=%g, want Mean=5 Std=1", cf.Mean, cf.Std)
	}
	if wp := byName["word_patterns"]; !(wp.Weight > 1.0 && wp.AIMedian > wp.HumanMedian) {
		t.Errorf("word_patterns = %+v, want positive Weight > 1.0 and AIMedian > HumanMedian", wp)
	}
	if ed := byName["em_dashes"]; !(ed.Weight < 0 && ed.HumanMedian > ed.AIMedian) {
		t.Errorf("em_dashes = %+v, want negative Weight and HumanMedian > AIMedian", ed)
	}
	if !(m.Missing["em_dashes"] < 0) {
		t.Errorf("Missing[em_dashes] = %g, want < 0 (missing only on human training rows)", m.Missing["em_dashes"])
	}

	pHuman := m.Prob(map[string]float64{"word_patterns": -2.2, "em_dashes": 7.0, "const_feat": 5.0}, 200)
	pAI := m.Prob(map[string]float64{"word_patterns": 2.4, "em_dashes": 0.3, "const_feat": 5.0}, 200)
	if !(pHuman < 0.08 && pAI > 0.92) {
		t.Errorf("Prob separation weak: pHuman=%g (want <0.08), pAI=%g (want >0.92)", pHuman, pAI)
	}

	// Verify Contributions log-odds sum exactly matches Prob for both present and missing features.
	for _, tc := range []struct {
		name  string
		feats map[string]float64
		words int
	}{
		{"all_present", map[string]float64{"word_patterns": 1.5, "em_dashes": 2.0, "const_feat": 5.0}, 250},
		{"missing_signals", map[string]float64{"word_patterns": -0.8}, 120},
	} {
		p := m.Prob(tc.feats, tc.words)
		wantLogit := math.Log(p / (1 - p))
		contribs := m.Contributions(tc.feats, tc.words)
		gotLogit := m.Bias
		for _, c := range contribs {
			gotLogit += c.Logit
			if c.Name == "log_words" {
				if !c.Present || c.Value != float64(tc.words) {
					t.Errorf("%s: log_words contribution = %+v, want Present=true Value=%d", tc.name, c, tc.words)
				}
			} else if _, ok := tc.feats[c.Name]; !ok {
				if c.Present || c.Logit != m.Missing[c.Name] {
					t.Errorf("%s: absent %s contribution = %+v, want Present=false Logit=%g",
						tc.name, c.Name, c, m.Missing[c.Name])
				}
			}
		}
		if math.Abs(gotLogit-wantLogit) > 1e-11 {
			t.Errorf("%s: Bias + sum(Contributions.Logit) = %.12f, want %.12f", tc.name, gotLogit, wantLogit)
		}
	}
}

func TestFitWithDropAndSampleWeights(t *testing.T) {
	rows := []calibrate.Row{
		{Feats: map[string]float64{"keep_sig": 0.1, "drop_sig": 9.0}, Words: 100, IsAI: false},
		{Feats: map[string]float64{"keep_sig": 0.2, "drop_sig": 8.0}, Words: 110, IsAI: false},
		{Feats: map[string]float64{"keep_sig": 1.0, "drop_sig": 0.0}, Words: 100, IsAI: true},
		{Feats: map[string]float64{"keep_sig": 1.1, "drop_sig": 1.0}, Words: 110, IsAI: true},
	}

	mDropped := calibrate.FitWith("pr", rows, calibrate.FitOptions{Drop: map[string]bool{"drop_sig": true}})
	for _, f := range mDropped.Features {
		if f.Name == "drop_sig" {
			t.Fatalf("Drop left drop_sig in Features: %+v", mDropped.Features)
		}
	}
	if _, exists := mDropped.Missing["drop_sig"]; exists {
		t.Fatalf("Drop left drop_sig in Missing: %+v", mDropped.Missing)
	}

	weightedRows := slices.Clone(rows)
	for i := range weightedRows {
		if weightedRows[i].IsAI {
			weightedRows[i].Weight = 6.0
		} else {
			weightedRows[i].Weight = 1.0
		}
	}
	mUnweighted := calibrate.Fit("pr", rows)
	mWeighted := calibrate.Fit("pr", weightedRows)
	if !(mWeighted.Bias > mUnweighted.Bias+0.5) {
		t.Errorf("upweighting AI rows 6:1 did not raise Bias: weighted=%g, unweighted=%g", mWeighted.Bias, mUnweighted.Bias)
	}
}

// TestFitWithSignConstrainAndEqualMedianFallback verifies both median-based sign
// clamping and the mean(ai)-mean(human) fallback when human_median == ai_median
// (e.g. sparse count features with median 0 on both sides).
func TestFitWithSignConstrainAndEqualMedianFallback(t *testing.T) {
	// Construct rows where word_patterns strongly predicts AI, except two human
	// rows have moderate word_patterns AND positive overlap_sig / sparse_count,
	// causing an unconstrained fit to assign negative compensating weights to
	// overlap_sig (AIMedian=2 > HumanMedian=0) and sparse_count (AIMedian=0 ==
	// HumanMedian=0, but mean(AI)=2.0 > mean(Human)=0.8).
	rows := []calibrate.Row{
		// 5 human rows: 3 clean (0, 0), 2 noisy with high word_patterns + positive features
		{Feats: map[string]float64{"word_patterns": -3.0, "overlap_sig": 0, "sparse_count": 0}, Words: 150, IsAI: false},
		{Feats: map[string]float64{"word_patterns": -3.0, "overlap_sig": 0, "sparse_count": 0}, Words: 150, IsAI: false},
		{Feats: map[string]float64{"word_patterns": -3.0, "overlap_sig": 0, "sparse_count": 0}, Words: 150, IsAI: false},
		{Feats: map[string]float64{"word_patterns": 1.2, "overlap_sig": 4, "sparse_count": 2}, Words: 150, IsAI: false},
		{Feats: map[string]float64{"word_patterns": 1.2, "overlap_sig": 4, "sparse_count": 2}, Words: 150, IsAI: false},
		// 5 AI rows: strong word_patterns=3.5; overlap_sig median is 2 (>0);
		// sparse_count is 0 on 3/5 rows (median=0) and 5 on 2/5 rows (mean=2.0 > human mean 0.8).
		{Feats: map[string]float64{"word_patterns": 3.5, "overlap_sig": 2, "sparse_count": 0}, Words: 150, IsAI: true},
		{Feats: map[string]float64{"word_patterns": 3.5, "overlap_sig": 2, "sparse_count": 0}, Words: 150, IsAI: true},
		{Feats: map[string]float64{"word_patterns": 3.5, "overlap_sig": 2, "sparse_count": 0}, Words: 150, IsAI: true},
		{Feats: map[string]float64{"word_patterns": 3.5, "overlap_sig": 0, "sparse_count": 5}, Words: 150, IsAI: true},
		{Feats: map[string]float64{"word_patterns": 3.5, "overlap_sig": 0, "sparse_count": 5}, Words: 150, IsAI: true},
	}

	free := calibrate.FitWith("prose", rows, calibrate.FitOptions{SignConstrain: false})
	constrained := calibrate.FitWith("prose", rows, calibrate.FitOptions{SignConstrain: true})

	featMap := func(m *calibrate.Model) map[string]calibrate.Feature {
		out := map[string]calibrate.Feature{}
		for _, f := range m.Features {
			out[f.Name] = f
		}
		return out
	}
	fFree := featMap(free)
	fCon := featMap(constrained)

	if !(fFree["overlap_sig"].AIMedian > fFree["overlap_sig"].HumanMedian && fFree["overlap_sig"].Weight < -0.1) {
		t.Fatalf("fixture setup: expected unconstrained overlap_sig weight < -0.1 with AIMedian > HumanMedian, got %+v", fFree["overlap_sig"])
	}
	if fFree["sparse_count"].HumanMedian != 0 || fFree["sparse_count"].AIMedian != 0 || !(fFree["sparse_count"].Weight < -0.1) {
		t.Fatalf("fixture setup: expected unconstrained sparse_count medians both 0 and weight < -0.1, got %+v", fFree["sparse_count"])
	}

	if fCon["overlap_sig"].Weight < 0 {
		t.Errorf("SignConstrain failed on median direction: overlap_sig weight = %g, want >= 0", fCon["overlap_sig"].Weight)
	}
	if fCon["sparse_count"].Weight < 0 {
		t.Errorf("SignConstrain failed on equal-median mean fallback: sparse_count weight = %g, want >= 0", fCon["sparse_count"].Weight)
	}
	if !(fCon["word_patterns"].Weight > 1.0) {
		t.Errorf("word_patterns weight under SignConstrain = %g, want > 1.0", fCon["word_patterns"].Weight)
	}
}
