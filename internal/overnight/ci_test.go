package overnight_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestCIWorkflowActionsAreSHAPinned verifies that every GitHub Action step in
// .github/workflows/ci.yml is pinned to a full 40-char lowercase commit SHA
// with a trailing version comment (e.g. "# v4").
func TestCIWorkflowActionsAreSHAPinned(t *testing.T) {
	ciPath := filepath.Join("..", "..", ".github", "workflows", "ci.yml")
	b, err := os.ReadFile(ciPath)
	if err != nil {
		t.Fatalf("read %s: %v", ciPath, err)
	}

	usesLineRE := regexp.MustCompile(`^\s*(?:-\s*)?uses:\s*(.+?)\s*$`)
	shaPinnedRE := regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}\s+#\s*v\d+\b`)

	var usesSteps []string
	hasCheckout, hasSetupGo := false, false
	for lineNo, line := range strings.Split(string(b), "\n") {
		m := usesLineRE.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		spec := m[1]
		usesSteps = append(usesSteps, spec)
		if !shaPinnedRE.MatchString(spec) {
			t.Errorf("ci.yml:%d uses unpinned action %q; want owner/repo@<40-hex-sha> # vN", lineNo+1, spec)
		}
		if strings.HasPrefix(spec, "actions/checkout@") {
			hasCheckout = true
		}
		if strings.HasPrefix(spec, "actions/setup-go@") {
			hasSetupGo = true
		}
	}
	if len(usesSteps) < 2 || !hasCheckout || !hasSetupGo {
		t.Errorf("expected SHA-pinned actions/checkout and actions/setup-go steps in ci.yml, found: %v", usesSteps)
	}
}
