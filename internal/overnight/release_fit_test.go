package overnight

import (
	"fmt"
	"slices"
	"testing"

	"github.com/joeshirey/AIDetector/internal/split"
	"github.com/joeshirey/AIDetector/internal/train"
)

func TestReleaseFitAndHoldoutStayInSync(t *testing.T) {
	const total = 5000
	holdout := 0
	for i := range total {
		base := fmt.Sprintf("sample-%04d", i)
		variant := base + "~persona"
		want := split.FinalHoldout(base)
		if want {
			holdout++
		}
		for _, id := range []string{base, variant} {
			if got := split.FinalHoldout(id); got != want {
				t.Fatalf("split.FinalHoldout(%q) = %v, want %v", id, got, want)
			}
			if got := FinalHoldout(id); got != want {
				t.Fatalf("overnight.FinalHoldout(%q) = %v, want %v", id, got, want)
			}
			if got := train.FinalHoldout(id); got != want {
				t.Fatalf("train.FinalHoldout(%q) = %v, want %v", id, got, want)
			}
		}
	}
	if holdout < 400 || holdout > 600 {
		t.Fatalf("holdout count across %d synthetic IDs = %d, want ~500 (10%%)", total, holdout)
	}

	if len(train.ExperimentalFeatures) == 0 {
		t.Fatalf("train.ExperimentalFeatures is empty")
	}
	if !slices.Equal(ExperimentalFeatures, train.ExperimentalFeatures) {
		t.Fatalf("overnight.ExperimentalFeatures = %v, train.ExperimentalFeatures = %v", ExperimentalFeatures, train.ExperimentalFeatures)
	}

	to, err := train.ReleaseFitOptions("")
	if err != nil {
		t.Fatalf("train.ReleaseFitOptions(\"\"): %v", err)
	}
	oo := ReleaseFitOptions()
	if oo.Weighting != to.Weighting {
		t.Errorf("Weighting = %q, want %q", oo.Weighting, to.Weighting)
	}
	if oo.SignConstrain != to.SignConstrain {
		t.Errorf("SignConstrain = %v, want %v", oo.SignConstrain, to.SignConstrain)
	}
	if oo.Engine != to.Engine {
		t.Errorf("Engine = %+v, want %+v", oo.Engine, to.Engine)
	}
	for _, f := range ExperimentalFeatures {
		if !to.Drop[f] {
			t.Errorf("train.ReleaseFitOptions(\"\").Drop[%q] = false, want true", f)
		}
	}
}
