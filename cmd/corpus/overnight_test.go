package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/overnight"
)

func writeCorpusTestFile(t *testing.T, path, body string, mtime time.Time) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if !mtime.IsZero() {
		if err := os.Chtimes(path, mtime, mtime); err != nil {
			t.Fatal(err)
		}
	}
}

// TestOvernightGateFiltersIgnoreParentAINames exercises cmd/corpus overnight gate
// flags (-exclude-since, -exclude-human-since, -subsample, -exclude-gen-style)
// when the data root lives beneath parent directories named "ai", a generator
// name, and "human".
func TestOvernightGateFiltersIgnoreParentAINames(t *testing.T) {
	t.Setenv("AIDETECTOR_SCORER_URL", "")

	dataRoot := filepath.Join(t.TempDir(), "ai", "claude-sonnet-5-5", "human", "data")
	runRoot := filepath.Join(t.TempDir(), "runs")
	ledgerPath := filepath.Join(t.TempDir(), "ledger.jsonl")
	fixListPath := filepath.Join(t.TempDir(), "fixtures.json")
	fixDir := filepath.Join(t.TempDir(), "fixtures_cache")

	writeCorpusTestFile(t, fixListPath, `{"known_human":[],"monitoring":[]}`, time.Time{})
	if err := os.MkdirAll(fixDir, 0o755); err != nil {
		t.Fatal(err)
	}

	d := overnight.Data{Root: dataRoot}
	for _, g := range overnight.Genres {
		if err := os.MkdirAll(filepath.Join(d.Dir(g), "human"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(d.Dir(g), "ai"), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	before := time.Date(2026, 11, 5, 7, 0, 0, 0, time.UTC)
	after := time.Date(2026, 11, 5, 10, 0, 0, 0, time.UTC)
	cutoff := "2026-11-05T08:30:00Z"

	proseHuman := "Yesterday afternoon I walked out to the back shed, oiled the rusty iron hinges on the garden gate, " +
		"swept dry maple leaves off the brick path, and stacked seasoned firewood beside the porch before dark.\n"
	proseAI := "Modern distributed architectures leverage comprehensive telemetry, standardized deployment workflows, " +
		"and automated governance guardrails to ensure resilient production operations across hybrid cloud clusters.\n"

	prHuman := "Fix race condition when draining idle HTTP connections during graceful server shutdown.\n" +
		"Added a deterministic loopback test exercising concurrent close and dial paths.\n"
	prAI := "This pull request resolves a synchronization issue in connection lifecycle management, " +
		"improving reliability and ensuring clean resource teardown under concurrent load.\n"

	blog, _ := overnight.GenreByName("blog")
	pr, _ := overnight.GenreByName("pr")

	// Minimal prose train (sample-001) + test (sample-007) files so RetrainCandidate and Measure succeed.
	writeCorpusTestFile(t, filepath.Join(d.Dir(blog), "human", "sample-001.txt"), proseHuman, before)
	writeCorpusTestFile(t, filepath.Join(d.Dir(blog), "ai", "gemini-3.8-flash", "sample-001.txt"), proseAI, before)
	writeCorpusTestFile(t, filepath.Join(d.Dir(blog), "human", "sample-007.txt"), proseHuman, before)
	writeCorpusTestFile(t, filepath.Join(d.Dir(blog), "ai", "gemini-3.8-flash", "sample-007.txt"), proseAI, before)

	// PR test-split files (sample-007).
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "human", "sample-007.txt"), prHuman, before)
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "ai", "gemini-3.8-flash", "sample-007.txt"), prAI, before)

	// PR train-split files (6 total = 2 human + 4 AI):
	//   - human/sample-001.txt (before), human/sample-002.txt (after)
	//   - ai/claude-sonnet-5-5/sample-001.txt (before), ai/claude-sonnet-5-5/sample-002~persona.txt (after)
	//   - ai/gemini-3.8-flash/sample-001.txt (before), ai/gemini-3.8-flash/sample-002~persona.txt (after)
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "human", "sample-001.txt"), prHuman+" h1", before)
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "human", "sample-002.txt"), prHuman+" h2", after)
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "ai", "claude-sonnet-5-5", "sample-001.txt"), prAI+" c1", before)
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "ai", "claude-sonnet-5-5", "sample-002~persona.txt"), prAI+" c2", after)
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "ai", "gemini-3.8-flash", "sample-001.txt"), prAI+" g1", before)
	writeCorpusTestFile(t, filepath.Join(d.Dir(pr), "ai", "gemini-3.8-flash", "sample-002~persona.txt"), prAI+" g2", after)

	runGate := func(t *testing.T, name string, extraArgs ...string) int {
		t.Helper()
		baseArgs := []string{
			"gate",
			"-data", dataRoot,
			"-run-root", runRoot,
			"-ledger", ledgerPath,
			"-fixture-list", fixListPath,
			"-fixtures", fixDir,
			"-shipped",
			"-keep-cutoffs",
			"-name", name,
		}
		err := runOvernight(context.Background(), append(baseArgs, extraArgs...))
		if corpus.Spend != nil {
			_ = corpus.Spend.Close()
		}
		if err != nil {
			t.Fatalf("overnight gate (%s) failed: %v", name, err)
		}
		b, err := os.ReadFile(filepath.Join(runRoot, name, "pr.json"))
		if err != nil {
			t.Fatalf("read %s/pr.json: %v", name, err)
		}
		var m calibrate.Model
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatalf("unmarshal %s/pr.json: %v", name, err)
		}
		return m.Trained
	}

	// 1. -exclude-since without -exclude-gen drops only the 2 post-cutoff AI files (keeps post-cutoff human/sample-002.txt) -> 4.
	t.Run("exclude-since-all", func(t *testing.T) {
		if got := runGate(t, "exclude-since-all", "-exclude-since", cutoff); got != 4 {
			t.Errorf("-exclude-since Trained = %d, want 4 (2 human + 2 pre-cutoff AI)", got)
		}
	})

	// 2. -exclude-since with -exclude-gen gemini drops only gemini's post-cutoff AI file -> 5.
	t.Run("exclude-since-gemini", func(t *testing.T) {
		if got := runGate(t, "exclude-since-gemini", "-exclude-since", cutoff, "-exclude-gen", "gemini"); got != 5 {
			t.Errorf("-exclude-since -exclude-gen gemini Trained = %d, want 5", got)
		}
	})

	// 3. -exclude-human-since drops only post-cutoff human/sample-002.txt (keeps all 4 AI files) -> 5.
	t.Run("exclude-human-since", func(t *testing.T) {
		if got := runGate(t, "exclude-human-since", "-exclude-human-since", cutoff); got != 5 {
			t.Errorf("-exclude-human-since Trained = %d, want 5 (1 pre-cutoff human + 4 AI)", got)
		}
	})

	// 4. -subsample claude-sonnet-5-5=0 drops only claude-sonnet-5-5's 2 AI files (keeps 2 human + 2 gemini AI) -> 4.
	t.Run("subsample-claude", func(t *testing.T) {
		if got := runGate(t, "subsample-claude", "-subsample", "claude-sonnet-5-5=0"); got != 4 {
			t.Errorf("-subsample claude-sonnet-5-5=0 Trained = %d, want 4 (2 human + 2 gemini AI)", got)
		}
	})

	// 5. -exclude-gen-style claude-sonnet-5-5:persona drops only claude's ~persona file (keeps gemini's ~persona file) -> 5.
	t.Run("exclude-gen-style", func(t *testing.T) {
		if got := runGate(t, "exclude-gen-style", "-exclude-gen-style", "claude-sonnet-5-5:persona"); got != 5 {
			t.Errorf("-exclude-gen-style claude-sonnet-5-5:persona Trained = %d, want 5", got)
		}
	})
}
