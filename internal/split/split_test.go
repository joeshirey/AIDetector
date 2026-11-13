package split_test

import (
	"testing"

	"github.com/joeshirey/AIDetector/internal/split"
)

func TestBaseStripsVariantSuffix(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"sample-015", "sample-015"},
		{"sample-015~persona", "sample-015"},
		{"sample-015~continue", "sample-015"},
		{"sample-015~edit~extra", "sample-015"},
		{"pr-golang-go-555~plain", "pr-golang-go-555"},
		{"~leading", ""},
		{"", ""},
	}
	for _, tc := range cases {
		if got := split.Base(tc.in); got != tc.want {
			t.Errorf("Base(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestFinalHoldoutAndSplitGolden pins concrete sample IDs across FinalHoldout,
// Calib, and Of so any accidental change to the "final-holdout:" or "calib:"
// SHA-256 salt, big-endian byte order, modulus, or FNV-1a split hash fails CI.
func TestFinalHoldoutAndSplitGolden(t *testing.T) {
	cases := []struct {
		id      string
		splitOf string
		calib   bool
		holdout bool
	}{
		{id: "sample-001", splitOf: "train", calib: false, holdout: false},
		{id: "sample-002", splitOf: "train", calib: true, holdout: false},
		{id: "sample-003", splitOf: "train", calib: false, holdout: true},
		{id: "sample-004", splitOf: "train", calib: false, holdout: false},
		{id: "sample-005", splitOf: "train", calib: false, holdout: false},
		{id: "sample-006", splitOf: "train", calib: false, holdout: false},
		{id: "sample-008", splitOf: "train", calib: true, holdout: false},
		{id: "sample-009", splitOf: "train", calib: true, holdout: false},
		{id: "sample-012", splitOf: "test", calib: true, holdout: false},
		{id: "sample-015", splitOf: "train", calib: true, holdout: true},
		{id: "sample-016", splitOf: "test", calib: true, holdout: true},
		{id: "sample-031", splitOf: "train", calib: false, holdout: true},
		{id: "doc-1", splitOf: "train", calib: true, holdout: false},
		{id: "doc-4", splitOf: "test", calib: true, holdout: false},
	}

	variants := []string{"", "~plain", "~persona", "~continue", "~edit"}
	for _, tc := range cases {
		baseHash := split.Hash(tc.id)
		for _, suffix := range variants {
			vid := tc.id + suffix
			if got := split.FinalHoldout(vid); got != tc.holdout {
				t.Errorf("FinalHoldout(%q) = %v, want %v", vid, got, tc.holdout)
			}
			if got := split.Calib(vid); got != tc.calib {
				t.Errorf("Calib(%q) = %v, want %v", vid, got, tc.calib)
			}
			if got := split.Of(vid); got != tc.splitOf {
				t.Errorf("Of(%q) = %q, want %q", vid, got, tc.splitOf)
			}
			if got := split.Hash(vid); got != baseHash {
				t.Errorf("Hash(%q) = %d, want %d (must match Base)", vid, got, baseHash)
			}
		}
	}

	// Pin sample-007 as a known test-split ID as well, and verify variant invariance.
	for _, suffix := range variants {
		vid := "sample-007" + suffix
		if got := split.Of(vid); got != "test" {
			t.Errorf("Of(%q) = %q, want \"test\"", vid, got)
		}
		if split.FinalHoldout(vid) != split.FinalHoldout("sample-007") {
			t.Errorf("FinalHoldout(%q) diverged from base", vid)
		}
		if split.Calib(vid) != split.Calib("sample-007") {
			t.Errorf("Calib(%q) diverged from base", vid)
		}
	}
}
