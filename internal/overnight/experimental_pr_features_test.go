package overnight

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/split"
)

func TestExperimentalPRFeaturesGatedUntilEnabled(t *testing.T) {
	for _, want := range []string{"pr_section_count", "pr_checklist"} {
		if !slices.Contains(ExperimentalFeatures, want) {
			t.Fatalf("ExperimentalFeatures = %v, missing %q", ExperimentalFeatures, want)
		}
	}

	root := t.TempDir()
	d := Data{Root: root}
	for _, g := range Genres {
		for _, sub := range []string{"human", filepath.Join("ai", "gpt-test")} {
			if err := os.MkdirAll(filepath.Join(d.Dir(g), sub), 0o755); err != nil {
				t.Fatal(err)
			}
		}
	}

	var trainIDs []string
	for i := 0; len(trainIDs) < 6; i++ {
		id := fmt.Sprintf("synth-%d", i)
		if split.Of(id) == "train" {
			trainIDs = append(trainIDs, id)
		}
	}

	const (
		humanProse = "We shipped the parser fix yesterday afternoon after tracing the buffer reset bug in local logs."
		aiProse    = "Furthermore, this comprehensive overview delves into the intricate landscape of modern distributed architectures."
		humanPR    = "Fix a nil pointer dereference when loading an empty config file on startup."
		aiPR       = "## Summary\n\nGuard the config loader against empty files.\n\n## Test plan\n\n- [x] Run unit tests\n- [ ] Smoke test CLI\n"
	)

	blog, _ := GenreByName("blog")
	pr, _ := GenreByName("pr")
	for _, id := range trainIDs {
		for _, wf := range []struct {
			path string
			body string
		}{
			{filepath.Join(d.Dir(blog), "human", id+".txt"), humanProse},
			{filepath.Join(d.Dir(blog), "ai", "gpt-test", id+".txt"), aiProse},
			{filepath.Join(d.Dir(pr), "human", id+".txt"), humanPR},
			{filepath.Join(d.Dir(pr), "ai", "gpt-test", id+".txt"), aiPR},
		} {
			if err := os.WriteFile(wf.path, []byte(wf.body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}

	readPRFeatures := func(t *testing.T, modelDir string) map[string]bool {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(modelDir, "pr.json"))
		if err != nil {
			t.Fatalf("read pr.json: %v", err)
		}
		var m calibrate.Model
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatalf("unmarshal pr.json: %v", err)
		}
		got := make(map[string]bool, len(m.Features))
		for _, f := range m.Features {
			got[f.Name] = true
		}
		return got
	}

	ctx := context.Background()

	defaultDir := filepath.Join(root, "default-fit")
	defOpts := ReleaseFitOptions()
	defOpts.KeepCutoffs = true
	if err := RetrainCandidate(ctx, d, defaultDir, defOpts); err != nil {
		t.Fatalf("RetrainCandidate(default): %v", err)
	}
	defFeats := readPRFeatures(t, defaultDir)
	for _, exp := range []string{"pr_section_count", "pr_checklist"} {
		if defFeats[exp] {
			t.Errorf("default ReleaseFitOptions pr.json unexpectedly includes experimental feature %q", exp)
		}
	}
	if !defFeats["agent_pr_template"] {
		t.Errorf("default ReleaseFitOptions pr.json missing adopted feature %q", "agent_pr_template")
	}

	enabledDir := filepath.Join(root, "enabled-fit")
	enOpts := ReleaseFitOptions()
	enOpts.KeepCutoffs = true
	enOpts.EnableFeatures = []string{"pr_section_count", "pr_checklist"}
	if err := RetrainCandidate(ctx, d, enabledDir, enOpts); err != nil {
		t.Fatalf("RetrainCandidate(enabled): %v", err)
	}
	enFeats := readPRFeatures(t, enabledDir)
	for _, exp := range []string{"agent_pr_template", "pr_section_count", "pr_checklist"} {
		if !enFeats[exp] {
			t.Errorf("enabled FitOptions pr.json missing %q; got features %v", exp, enFeats)
		}
	}
}
