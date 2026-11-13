package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/mcpserver"
)

func TestMain(m *testing.M) {
	if os.Getenv("AIDETECTOR_TEST_CLI") == "1" {
		main()
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// runCLI invokes main() in a child process rooted at workDir so flag errors
// and os.Exit codes are observed cleanly without touching repository files.
func runCLI(t *testing.T, workDir string, args ...string) (stdout, stderr string, exitCode int) {
	t.Helper()
	cmd := exec.Command(os.Args[0], args...)
	cmd.Dir = workDir
	cmd.Env = append(os.Environ(), "AIDETECTOR_TEST_CLI=1")
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf
	err := cmd.Run()
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return outBuf.String(), errBuf.String(), exitErr.ExitCode()
		}
		t.Fatalf("exec CLI %v: %v", args, err)
	}
	return outBuf.String(), errBuf.String(), 0
}

func writeSyntheticFile(t *testing.T, path, content string, modTime time.Time) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if !modTime.IsZero() {
		if err := os.Chtimes(path, modTime, modTime); err != nil {
			t.Fatal(err)
		}
	}
}

// buildSyntheticProseCorpus creates a small synthetic prose corpus in a temp
// directory with deterministic train-split IDs:
//   - 6 human documents (with greeting/link/sign-off envelope so AlignNormalization
//     is observable, >=6 sentences & >=10 clauses so stylometric emits
//     prose_burstiness and clause_uniformity, and em-dashes on 2/6 docs so
//     HumanMedian(em_dashes) == 0).
//   - 12 AI documents across 2 generators (9 in claude-sonnet-5-5 including
//     style mirrors, 3 in gemini-3.8-flash; 8/12 include em-dashes so
//     AIMedian(em_dashes) > 0 while strong AI wording drives word_patterns).
func buildSyntheticProseCorpus(t *testing.T) string {
	t.Helper()
	corpusDir := filepath.Join(t.TempDir(), "prose_corpus")
	trainIDs := []string{"sample-001", "sample-002", "sample-003", "sample-004", "sample-005", "sample-006"}

	for i, id := range trainIDs {
		dash := ","
		shared := "everyday workshop notes"
		if i >= 4 {
			// Two human docs carry em-dashes and a few AI-ish words so an
			// unconstrained fit pushes em_dashes negative despite AIMedian > HumanMedian.
			dash = " —"
			shared = "syntheticmarkerone syntheticmarkertwo"
		}
		humanText := fmt.Sprintf(
			"Hi team,\n"+
				"Yesterday afternoon I walked down to the neighborhood hardware store, picked up three brass hinges%s and fixed the squeaky pantry door before dinner (%d). "+
				"The old screws had rusted solid after twenty damp winters, so I soaked them in kerosene, tapped the screwdriver handle, andpried the bracket loose; see https://example.com/notes/%s for @alice. "+
				"My neighbor stopped by with fresh sourdough bread, we talked about tomato seedlings, and the dog chased sparrows across the muddy lawn.\n\n"+
				"Later that evening we patched the garden fence, swept sawdust off the porch steps%s and stacked dry kindling beside the stove (%s). "+
				"Nothing felt urgent, although wind rattled the attic window, rain drummed on the tin roof, and the kettle whistled right at nine. "+
				"Tomorrow morning I will oil the latch, paint the trim, and return the borrowed wrench.\n"+
				"Thanks, Jordan\n",
			dash, i, id, dash, shared,
		)
		writeSyntheticFile(t, filepath.Join(corpusDir, "human", id+".txt"), humanText, time.Time{})

		aiDash := " —"
		if i >= 4 {
			aiDash = ","
		}
		aiBase := fmt.Sprintf(
			"syntheticmarkerone syntheticmarkertwo syntheticmarkerthree %s. "+
				"Modern distributed systems require comprehensive observability%s resilient telemetry pipelines, and deterministic deployment workflows across hybrid environments. "+
				"When organizations streamline operational governance, teams reduce incident latency, improve service reliability, and maintain consistent auditability.\n\n"+
				"Furthermore, standardized configuration templates establish clear ownership boundaries%s simplify cross-functional collaboration, and eliminate manual provisioning drift. "+
				"Engineering leaders systematically evaluate throughput metrics, calibrate capacity thresholds, and align platform architecture with long-term scalability objectives. "+
				"Ultimately, disciplined automation fosters sustainable delivery velocity, strengthens security posture, and ensures predictable production outcomes.\n",
			id, aiDash, aiDash,
		)
		writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "claude-sonnet-5-5", id+".txt"), aiBase, time.Time{})
		if i < 3 {
			writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "claude-sonnet-5-5", id+"~persona.txt"), aiBase+" Additional persona variant alpha beta.\n", time.Time{})
			writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", id+".txt"), aiBase+" Gemini generator phrasing gamma delta.\n", time.Time{})
		}
	}
	return corpusDir
}

func readCalibrateModel(t *testing.T, path string) calibrate.Model {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var m calibrate.Model
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal %s: %v", path, err)
	}
	return m
}

func featureMap(m calibrate.Model) map[string]calibrate.Feature {
	out := make(map[string]calibrate.Feature, len(m.Features))
	for _, f := range m.Features {
		out[f.Name] = f
	}
	return out
}

// AC1 + AC5: aidetector fit defaults to ReleaseFitOptions (Weighting:
// "source+balance-generator", SignConstrain: true, dropping ExperimentalFeatures,
// and loading engine.json from the output or embedded model directory).
func TestFitReleaseDefaults(t *testing.T) {
	workDir := t.TempDir()
	corpusDir := buildSyntheticProseCorpus(t)
	defaultOut := filepath.Join(t.TempDir(), "default_out")

	_, stderr, code := runCLI(t, workDir, "fit", "-kind", "prose", "-out", defaultOut, corpusDir)
	if code != 0 {
		t.Fatalf("fit failed (exit %d): %s", code, stderr)
	}

	m := readCalibrateModel(t, filepath.Join(defaultOut, "prose.json"))
	if m.Trained != 18 {
		t.Fatalf("Trained = %d, want 18 (6 human + 12 AI train samples)", m.Trained)
	}
	feats := featureMap(m)

	// 1. ExperimentalFeatures must be dropped by default.
	for _, exp := range []string{"prose_burstiness", "clause_uniformity"} {
		if _, exists := feats[exp]; exists {
			t.Errorf("default fit included experimental feature %q in prose.json; want dropped", exp)
		}
	}

	// 2. SignConstrain must be enabled by default: no hand-built feature may
	// have a weight pointing opposite its (AIMedian - HumanMedian) direction.
	for _, f := range m.Features {
		if f.Name == "word_patterns" || f.Name == "log_words" {
			continue
		}
		if d := f.AIMedian - f.HumanMedian; d != 0 && f.Weight*d < 0 {
			t.Errorf("feature %q violated SignConstrain: weight=%g, human_median=%g, ai_median=%g",
				f.Name, f.Weight, f.HumanMedian, f.AIMedian)
		}
	}

	// 3. engine.json must be loaded from embedded models by default (and
	// respected when present in -out). With align_normalization=false in -out,
	// greeting/link/sign-off tokens stay in human prose so log_words.mean is higher.
	engBytes, err := os.ReadFile(filepath.Join(defaultOut, "engine.json"))
	if err != nil {
		t.Fatalf("expected engine.json in %s: %v", defaultOut, err)
	}
	var engOpts engine.Options
	if err := json.Unmarshal(engBytes, &engOpts); err != nil {
		t.Fatalf("unmarshal engine.json: %v", err)
	}
	if !engOpts.AlignNormalization || engOpts.ShortProse.Below != 150 || engOpts.ShortProse.Above != 350 {
		t.Errorf("default engine.json = %+v, want AlignNormalization=true and ShortProse={150 350}", engOpts)
	}

	noAlignOut := filepath.Join(t.TempDir(), "no_align_out")
	writeSyntheticFile(t, filepath.Join(noAlignOut, "engine.json"), `{"align_normalization": false}`, time.Time{})
	_, stderr, code = runCLI(t, workDir, "fit", "-kind", "prose", "-out", noAlignOut, corpusDir)
	if code != 0 {
		t.Fatalf("fit with custom engine.json failed (exit %d): %s", code, stderr)
	}
	mNoAlign := readCalibrateModel(t, filepath.Join(noAlignOut, "prose.json"))
	if !(featureMap(mNoAlign)["log_words"].Mean > feats["log_words"].Mean) {
		t.Errorf("expected custom engine.json align_normalization=false to keep envelope/links (log_words.mean %g <= aligned %g)",
			featureMap(mNoAlign)["log_words"].Mean, feats["log_words"].Mean)
	}

	// 4. Default fit must match explicit -weighting=source+balance-generator -sign-constrain=true
	// byte-for-byte, and differ from unweighted (-weighting="") where 12 AI vs 6 human skews bias upward.
	explicitOut := filepath.Join(t.TempDir(), "explicit_out")
	_, stderr, code = runCLI(t, workDir, "fit", "-kind", "prose",
		"-weighting", "source+balance-generator", "-sign-constrain=true", "-out", explicitOut, corpusDir)
	if code != 0 {
		t.Fatalf("explicit release-flag fit failed (exit %d): %s", code, stderr)
	}
	gotDefault, _ := os.ReadFile(filepath.Join(defaultOut, "prose.json"))
	gotExplicit, _ := os.ReadFile(filepath.Join(explicitOut, "prose.json"))
	if !bytes.Equal(gotDefault, gotExplicit) {
		t.Errorf("default fit prose.json differs from explicit -weighting=source+balance-generator -sign-constrain=true")
	}

	unweightedOut := filepath.Join(t.TempDir(), "unweighted_out")
	_, stderr, code = runCLI(t, workDir, "fit", "-kind", "prose", "-weighting", "", "-out", unweightedOut, corpusDir)
	if code != 0 {
		t.Fatalf("unweighted fit failed (exit %d): %s", code, stderr)
	}
	mUnweighted := readCalibrateModel(t, filepath.Join(unweightedOut, "prose.json"))
	if !(mUnweighted.Bias > m.Bias+0.1) {
		t.Errorf("expected unweighted fit bias (%g) > source+balance-generator bias (%g) on 2:1 AI-heavy corpus",
			mUnweighted.Bias, m.Bias)
	}
}

// AC1 + AC5: Override flags on aidetector fit enable experimental features,
// turn off sign constraints, and override engine.json options.
func TestFitOverrideFlags(t *testing.T) {
	workDir := t.TempDir()
	corpusDir := buildSyntheticProseCorpus(t)
	overrideOut := filepath.Join(t.TempDir(), "override_out")

	_, stderr, code := runCLI(t, workDir, "fit", "-kind", "prose",
		"-enable-features", "prose_burstiness,clause_uniformity",
		"-sign-constrain=false",
		"-weighting", "",
		"-align-norm=false",
		"-short-prose", "0:0",
		"-out", overrideOut,
		corpusDir,
	)
	if code != 0 {
		t.Fatalf("fit with override flags failed (exit %d): %s", code, stderr)
	}

	m := readCalibrateModel(t, filepath.Join(overrideOut, "prose.json"))
	feats := featureMap(m)
	for _, exp := range []string{"prose_burstiness", "clause_uniformity"} {
		if _, exists := feats[exp]; !exists {
			t.Errorf("-enable-features did not include %q in prose.json", exp)
		}
	}
	unconstrainedFlip := false
	for _, f := range m.Features {
		if f.Name == "word_patterns" || f.Name == "log_words" {
			continue
		}
		if d := f.AIMedian - f.HumanMedian; d != 0 && f.Weight*d < 0 {
			unconstrainedFlip = true
			break
		}
	}
	if !unconstrainedFlip {
		t.Errorf("expected -sign-constrain=false to allow opposite-median feature weights on synthetic fixture")
	}

	engBytes, err := os.ReadFile(filepath.Join(overrideOut, "engine.json"))
	if err != nil {
		t.Fatalf("read override engine.json: %v", err)
	}
	var engOpts engine.Options
	if err := json.Unmarshal(engBytes, &engOpts); err != nil {
		t.Fatalf("unmarshal override engine.json: %v", err)
	}
	if engOpts.AlignNormalization || engOpts.ShortProse.Below != 0 || engOpts.ShortProse.Above != 0 {
		t.Errorf("override engine.json = %+v, want all zero", engOpts)
	}
}

// AC2 + AC5: aidetector thresholds excludes the 10% final holdout by default,
// accepts a model directory for -out (writing <out>/thresholds.json), and
// scores using candidate models in <out> via engine.FromDir rather than engine.Default().
func TestThresholdsExcludesFinalHoldoutAndUsesCandidateDir(t *testing.T) {
	workDir := t.TempDir()
	poolDir := filepath.Join(t.TempDir(), "pr_pool")

	// Deterministic IDs partitioned by split.Calib and FinalHoldout:
	//   calib + non-holdout (4): sample-002, sample-008, sample-009, sample-012
	//   calib + holdout     (2): sample-015, sample-016
	//   check + non-holdout (4): sample-001, sample-004, sample-005, sample-006
	//   check + holdout     (2): sample-003, sample-031
	ids := []string{
		"sample-002", "sample-008", "sample-009", "sample-012",
		"sample-015", "sample-016",
		"sample-001", "sample-004", "sample-005", "sample-006",
		"sample-003", "sample-031",
	}
	prBody := "Fix nil pointer dereference when closing idle HTTP keep-alive connections after server shutdown timeout expires in worker pool.\n" +
		"Added a unit test reproducing the race under load and verified connection draining completes cleanly without leaking goroutines.\n"
	for _, id := range ids {
		writeSyntheticFile(t, filepath.Join(poolDir, id+".txt"), prBody, time.Time{})
	}

	// Write a custom candidate pr.json with bias=4.5 (raw score ~98.9 on every
	// doc) so thresholds calibrated via engine.FromDir(candDir) must set mixed >= 98,
	// whereas engine.Default() scores prBody well below 70.
	candDir := filepath.Join(t.TempDir(), "cand_models")
	candModel := calibrate.Model{
		Kind:     "pr",
		Bias:     4.5,
		Features: []calibrate.Feature{},
		Missing:  map[string]float64{},
		Trained:  10,
	}
	b, _ := json.MarshalIndent(candModel, "", " ")
	writeSyntheticFile(t, filepath.Join(candDir, "pr.json"), string(b)+"\n", time.Time{})

	stdout, stderr, code := runCLI(t, workDir, "thresholds", "-kind", "pr", "-only", "documents", "-out", candDir, poolDir)
	if code != 0 {
		t.Fatalf("thresholds failed (exit %d): %s", code, stderr)
	}
	if !strings.Contains(stdout, "(checked on 4 held out)") {
		t.Errorf("stdout = %q, want check-half count '(checked on 4 held out)' with holdout excluded", stdout)
	}

	tb, err := os.ReadFile(filepath.Join(candDir, "thresholds.json"))
	if err != nil {
		t.Fatalf("read %s/thresholds.json: %v", candDir, err)
	}
	var tf map[string]engine.Thresholds
	if err := json.Unmarshal(tb, &tf); err != nil {
		t.Fatalf("unmarshal thresholds.json: %v", err)
	}
	prT, ok := tf["pr"]
	if !ok {
		t.Fatalf("thresholds.json missing 'pr' key: %v", tf)
	}
	if prT.CalibratedOn != 4 {
		t.Errorf("CalibratedOn = %d, want 4 (2 final-holdout calib files must be excluded)", prT.CalibratedOn)
	}
	if prT.Mixed < 98 {
		t.Errorf("pr.Mixed = %d, want >= 98 (runThresholds must score with candidate models in -out, not embedded Default)", prT.Mixed)
	}

	// Override flag -exclude-holdout=false includes all 6 calib and 6 check files.
	allOut := filepath.Join(t.TempDir(), "all_thresholds.json")
	stdoutAll, stderr, code := runCLI(t, workDir, "thresholds", "-kind", "pr", "-only", "documents", "-exclude-holdout=false", "-out", allOut, poolDir)
	if code != 0 {
		t.Fatalf("thresholds -exclude-holdout=false failed (exit %d): %s", code, stderr)
	}
	if !strings.Contains(stdoutAll, "(checked on 6 held out)") {
		t.Errorf("stdout with -exclude-holdout=false = %q, want '(checked on 6 held out)'", stdoutAll)
	}
	tbAll, err := os.ReadFile(allOut)
	if err != nil {
		t.Fatalf("read %s: %v", allOut, err)
	}
	var tfAll map[string]engine.Thresholds
	if err := json.Unmarshal(tbAll, &tfAll); err != nil {
		t.Fatal(err)
	}
	if tfAll["pr"].CalibratedOn != 6 {
		t.Errorf("CalibratedOn with -exclude-holdout=false = %d, want 6", tfAll["pr"].CalibratedOn)
	}
}

// AC2: README.md ("Rebuilding the models") documents both the reproducible
// corpus overnight gate path and the aligned aidetector fit / thresholds
// scratch-dir commands (including all 3 CalibrateCutoffs passes in order).
func TestReadmeRebuildDocs(t *testing.T) {
	readmePath := filepath.Join("..", "..", "README.md")
	b, err := os.ReadFile(readmePath)
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	full := string(b)

	start := strings.Index(full, "## Rebuilding the models")
	if start < 0 {
		t.Fatal("README.md missing '## Rebuilding the models' section")
	}
	section := full[start+len("## Rebuilding the models"):]
	if next := strings.Index(section, "\n## "); next >= 0 {
		section = section[:next]
	}

	if !strings.Contains(section, "overnight gate") {
		t.Errorf("Rebuilding the models section must document 'corpus overnight gate'")
	}

	// Every CLI invocation of aidetector fit or aidetector thresholds in README
	// must pass an explicit -out flag rather than defaulting to internal/engine/models.
	for lineNo, line := range strings.Split(full, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			continue
		}
		if (strings.Contains(trimmed, "aidetector fit ") || strings.Contains(trimmed, "aidetector thresholds ")) &&
			!strings.Contains(trimmed, "-out ") {
			t.Errorf("README.md:%d invokes fit/thresholds without explicit -out: %s", lineNo+1, trimmed)
		}
	}

	// Verify the 3 CalibrateCutoffs passes appear in order inside Rebuilding the models:
	//   1) PR documents on corpus_fresh/pr/pool and corpus_prpool/pr/pool
	//   2) Prose documents on corpus_fresh/{blog,memo,essay}/pool
	//   3) Prose segments on those plus corpus_tech/blog/pool and corpus_nontech/blog/pool
	var idxPR, idxProseDoc, idxProseSeg int = -1, -1, -1
	for i, line := range strings.Split(section, "\n") {
		if !strings.Contains(line, "aidetector thresholds") {
			continue
		}
		switch {
		case strings.Contains(line, "-kind pr") && strings.Contains(line, "-only documents") &&
			strings.Contains(line, "corpus_fresh/pr/pool") && strings.Contains(line, "corpus_prpool/pr/pool"):
			if idxPR < 0 {
				idxPR = i
			}
		case strings.Contains(line, "-kind prose") && strings.Contains(line, "-only documents") &&
			strings.Contains(line, "corpus_fresh/blog/pool") && strings.Contains(line, "corpus_fresh/memo/pool") &&
			strings.Contains(line, "corpus_fresh/essay/pool"):
			if idxProseDoc < 0 {
				idxProseDoc = i
			}
		case strings.Contains(line, "-kind prose") && strings.Contains(line, "-only segments") &&
			strings.Contains(line, "corpus_fresh/blog/pool") && strings.Contains(line, "corpus_fresh/memo/pool") &&
			strings.Contains(line, "corpus_fresh/essay/pool") && strings.Contains(line, "corpus_tech/blog/pool") &&
			strings.Contains(line, "corpus_nontech/blog/pool"):
			if idxProseSeg < 0 {
				idxProseSeg = i
			}
		}
	}
	if idxPR < 0 || idxProseDoc < 0 || idxProseSeg < 0 {
		t.Fatalf("Rebuilding the models missing one or more of the 3 CalibrateCutoffs passes (pr=%d, proseDoc=%d, proseSeg=%d)",
			idxPR, idxProseDoc, idxProseSeg)
	}
	if !(idxPR < idxProseDoc && idxProseDoc < idxProseSeg) {
		t.Errorf("thresholds passes out of order: line indices pr=%d, proseDoc=%d, proseSeg=%d (want pr < proseDoc < proseSeg)",
			idxPR, idxProseDoc, idxProseSeg)
	}
}

// AC3 + AC5: aidetector fit -exclude-since <RFC 3339> leaves out AI files whose
// mtime is after the cutoff while keeping human files (regardless of mtime) and
// fitting from split.Of(id) == "train" only.
func TestFitExcludeSinceAndTrainSplitOnly(t *testing.T) {
	workDir := t.TempDir()
	corpusDir := filepath.Join(t.TempDir(), "pr_corpus")

	before := time.Date(2026, 11, 5, 7, 0, 0, 0, time.FixedZone("PST", -8*3600))
	after := time.Date(2026, 11, 5, 9, 6, 18, 0, time.FixedZone("PST", -8*3600))
	cutoff := "2026-11-05T08:15:40-08:00"

	humanPR := "Fix race condition in connection pool cleanup when context cancels early during TLS handshake retry.\n" +
		"Verified with go test -race across twenty iterations on local loopback sockets.\n"
	aiPR := "This pull request resolves a concurrency issue in connection lifecycle management, ensuring deterministic resource cleanup and improved reliability across retry paths.\n" +
		"Comprehensive unit tests validate graceful cancellation semantics.\n"

	// 4 train-split human files: 2 before cutoff, 2 after cutoff (all 4 must be kept).
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-001.txt"), humanPR+" alpha", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-002.txt"), humanPR+" beta", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-003.txt"), humanPR+" gamma", after)
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-004.txt"), humanPR+" delta", after)

	// 4 train-split AI files before cutoff (kept) + 2 train-split AI files after cutoff (excluded by -exclude-since).
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-001.txt"), aiPR+" alpha", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-002.txt"), aiPR+" beta", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-003.txt"), aiPR+" gamma", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-004.txt"), aiPR+" delta", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-005.txt"), aiPR+" epsilon", after)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-006.txt"), aiPR+" zeta", after)

	// 2 test-split files (sample-007 has split.Of == "test") before cutoff (must never enter fit).
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-007.txt"), humanPR+" testsplit", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-007.txt"), aiPR+" testsplit", before)

	filteredOut := filepath.Join(t.TempDir(), "filtered_out")
	_, stderr, code := runCLI(t, workDir, "fit", "-kind", "pr", "-exclude-since", cutoff, "-out", filteredOut, corpusDir)
	if code != 0 {
		t.Fatalf("fit -exclude-since failed (exit %d): %s", code, stderr)
	}
	mFiltered := readCalibrateModel(t, filepath.Join(filteredOut, "pr.json"))
	if mFiltered.Trained != 8 {
		t.Errorf("Trained with -exclude-since = %d, want 8 (4 train human + 4 pre-cutoff train AI)", mFiltered.Trained)
	}

	unfilteredOut := filepath.Join(t.TempDir(), "unfiltered_out")
	_, stderr, code = runCLI(t, workDir, "fit", "-kind", "pr", "-out", unfilteredOut, corpusDir)
	if code != 0 {
		t.Fatalf("fit without -exclude-since failed (exit %d): %s", code, stderr)
	}
	mUnfiltered := readCalibrateModel(t, filepath.Join(unfilteredOut, "pr.json"))
	if mUnfiltered.Trained != 10 {
		t.Errorf("Trained without -exclude-since = %d, want 10 (4 train human + 6 train AI; test split excluded)", mUnfiltered.Trained)
	}

	badOut := filepath.Join(t.TempDir(), "bad_out")
	_, stderr, code = runCLI(t, workDir, "fit", "-kind", "pr", "-exclude-since", "not-a-timestamp", "-out", badOut, corpusDir)
	if code == 0 || !strings.Contains(stderr, "exclude-since") {
		t.Errorf("expected non-zero exit and -exclude-since error for invalid timestamp, got code=%d stderr=%q", code, stderr)
	}
}

// AC4 + AC5: aidetector fit and aidetector thresholds require an explicit -out
// and never default to writing under internal/engine/models/.
func TestFitAndThresholdsRequireOut(t *testing.T) {
	workDir := t.TempDir()
	corpusDir := filepath.Join(t.TempDir(), "mini_pr")
	poolDir := filepath.Join(t.TempDir(), "mini_pool")

	prHuman := "Fix deadlock in channel shutdown path when consumer exits early during batch flush.\n" +
		"Added regression test covering buffered and unbuffered cases.\n"
	prAI := "This change addresses a synchronization edge case during worker shutdown, ensuring clean resource teardown and reliable error propagation across concurrent consumers.\n"

	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-001.txt"), prHuman, time.Time{})
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "claude-sonnet-5-5", "sample-001.txt"), prAI, time.Time{})
	writeSyntheticFile(t, filepath.Join(poolDir, "sample-002.txt"), prHuman, time.Time{})
	writeSyntheticFile(t, filepath.Join(poolDir, "sample-001.txt"), prHuman, time.Time{})

	_, fitErr, fitCode := runCLI(t, workDir, "fit", "-kind", "pr", corpusDir)
	if fitCode == 0 {
		t.Errorf("fit without -out succeeded (exit 0); want non-zero error requiring -out")
	}
	if !strings.Contains(fitErr, "-out") {
		t.Errorf("fit without -out stderr = %q; want mention of -out", fitErr)
	}

	_, threshErr, threshCode := runCLI(t, workDir, "thresholds", "-kind", "pr", poolDir)
	if threshCode == 0 {
		t.Errorf("thresholds without -out succeeded (exit 0); want non-zero error requiring -out")
	}
	if !strings.Contains(threshErr, "-out") {
		t.Errorf("thresholds without -out stderr = %q; want mention of -out", threshErr)
	}

	shippedRel := filepath.Join(workDir, "internal", "engine", "models")
	if _, err := os.Stat(shippedRel); err == nil {
		t.Errorf("CLI wrote to default %s when -out was omitted; must never write to internal/engine/models", shippedRel)
	}
}

func assertValidModelsSHA(t *testing.T, label, id string) {
	t.Helper()
	if !strings.HasPrefix(id, "sha256:") || len(id) != len("sha256:")+64 {
		t.Errorf("%s = %q, want sha256:<64-hex>", label, id)
	}
}

// AC4 + AC5: aidetector -version, --version, and version exit 0 and print
// version, commit (or "unknown" when unstamped), Go version, and the embedded
// models sha256 identifier; usage() includes version / -version.
func TestCLIVersionAndUsage(t *testing.T) {
	t.Setenv("AIDETECTOR_SCORER_URL", "")
	workDir := t.TempDir()

	bi := engine.Default().BuildInfo()
	assertValidModelsSHA(t, "engine.Default().BuildInfo().Models", bi.Models)

	wantCommit := bi.Commit
	if wantCommit == "" {
		wantCommit = "unknown"
	}

	var firstOut string
	for _, arg := range []string{"-version", "--version", "version"} {
		stdout, stderr, code := runCLI(t, workDir, arg)
		if code != 0 {
			t.Errorf("aidetector %s exited %d (stderr=%q), want 0", arg, code, stderr)
			continue
		}
		if stderr != "" {
			t.Errorf("aidetector %s wrote unexpected stderr %q", arg, stderr)
		}
		if bi.Version == "" || !strings.Contains(stdout, bi.Version) {
			t.Errorf("aidetector %s stdout %q missing version %q", arg, stdout, bi.Version)
		}
		if !strings.Contains(stdout, wantCommit) {
			t.Errorf("aidetector %s stdout %q missing commit %q", arg, stdout, wantCommit)
		}
		if !strings.Contains(stdout, runtime.Version()) {
			t.Errorf("aidetector %s stdout %q missing Go version %q", arg, stdout, runtime.Version())
		}
		if !strings.HasPrefix(bi.Models, "sha256:") || !strings.Contains(stdout, bi.Models) {
			t.Errorf("aidetector %s stdout %q missing models digest %q", arg, stdout, bi.Models)
		}
		if firstOut == "" {
			firstOut = stdout
		} else if stdout != firstOut {
			t.Errorf("aidetector %s stdout %q != -version stdout %q", arg, stdout, firstOut)
		}
	}

	_, helpErr, helpCode := runCLI(t, workDir, "-h")
	if helpCode != 0 {
		t.Fatalf("aidetector -h exited %d, want 0", helpCode)
	}
	if !strings.Contains(helpErr, "version") || !strings.Contains(helpErr, "-version") {
		t.Errorf("usage() output missing version / -version:\n%s", helpErr)
	}
}

// AC3 + AC5: CLI -json output and MCP detect_ai_text tool results over an
// in-memory transport both include Result.Build with valid jsonschema tags.
func TestCLIJSONAndMCPServerIncludeBuildInfo(t *testing.T) {
	t.Setenv("AIDETECTOR_SCORER_URL", "")
	workDir := t.TempDir()

	sampleText := "Yesterday afternoon I patched three cedar shingles above the back porch overhang, " +
		"swept the pine needles out of the copper gutter, and sealed the corner seam before dusk. " +
		"The old flashing had worked loose during January windstorms, so I replaced two rusted nails " +
		"and checked the attic rafters with a flashlight when the rain started."
	samplePath := filepath.Join(workDir, "note.txt")
	writeSyntheticFile(t, samplePath, sampleText, time.Time{})

	wantBuild := engine.Default().BuildInfo()
	assertValidModelsSHA(t, "Default().BuildInfo().Models", wantBuild.Models)

	stdout, stderr, code := runCLI(t, workDir, "-json", samplePath)
	if code != 0 {
		t.Fatalf("aidetector -json failed (exit %d): %s", code, stderr)
	}
	var cliRes engine.Result
	if err := json.Unmarshal([]byte(stdout), &cliRes); err != nil {
		t.Fatalf("unmarshal CLI JSON: %v", err)
	}
	if cliRes.Build != wantBuild {
		t.Errorf("CLI -json Result.Build = %+v, want %+v", cliRes.Build, wantBuild)
	}
	assertValidModelsSHA(t, "CLI -json Result.Build.Models", cliRes.Build.Models)

	// Verify MCP server tool schema registration and detect_ai_text structured output.
	ctx := context.Background()
	srv := mcpserver.New(engine.Default(), "test")
	serverTransport, clientTransport := mcp.NewInMemoryTransports()
	serverSession, err := srv.Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatalf("srv.Connect: %v", err)
	}
	t.Cleanup(func() { _ = serverSession.Close() })

	client := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "v0.0.1"}, nil)
	clientSession, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatalf("client.Connect: %v", err)
	}
	t.Cleanup(func() { _ = clientSession.Close() })

	toolsRes, err := clientSession.ListTools(ctx, nil)
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	foundDetectText := false
	for _, tool := range toolsRes.Tools {
		if tool.Name == "detect_ai_text" {
			foundDetectText = true
			schemaBytes, err := json.Marshal(tool.OutputSchema)
			if err != nil {
				t.Fatalf("marshal OutputSchema: %v", err)
			}
			if !strings.Contains(string(schemaBytes), `"build"`) {
				t.Errorf("detect_ai_text OutputSchema missing \"build\" property: %s", schemaBytes)
			}
		}
	}
	if !foundDetectText {
		t.Fatalf("detect_ai_text tool not registered in %+v", toolsRes.Tools)
	}

	for _, tc := range []struct {
		name string
		text string
	}{
		{name: "prose", text: sampleText},
		{name: "empty", text: ""},
	} {
		callRes, err := clientSession.CallTool(ctx, &mcp.CallToolParams{
			Name:      "detect_ai_text",
			Arguments: map[string]any{"text": tc.text, "kind": "prose"},
		})
		if err != nil {
			t.Fatalf("CallTool(%s): %v", tc.name, err)
		}
		if callRes.IsError {
			t.Fatalf("CallTool(%s) returned tool error: %+v", tc.name, callRes.Content)
		}
		scBytes, err := json.Marshal(callRes.StructuredContent)
		if err != nil {
			t.Fatalf("marshal StructuredContent(%s): %v", tc.name, err)
		}
		var mcpRes engine.Result
		if err := json.Unmarshal(scBytes, &mcpRes); err != nil {
			t.Fatalf("unmarshal StructuredContent(%s): %v", tc.name, err)
		}
		if mcpRes.Build != wantBuild {
			t.Errorf("MCP detect_ai_text(%s) Build = %+v, want %+v", tc.name, mcpRes.Build, wantBuild)
		}
		assertValidModelsSHA(t, "MCP detect_ai_text("+tc.name+").Build.Models", mcpRes.Build.Models)
		if mcpRes.Build.Version == "" || mcpRes.Build.GoVersion != runtime.Version() {
			t.Errorf("MCP detect_ai_text(%s) Build = %+v, want non-empty Version and GoVersion %q",
				tc.name, mcpRes.Build, runtime.Version())
		}
	}
}

// AC6: README.md documents aidetector -version and the Result.build JSON field.
func TestReadmeVersionAndBuildDocs(t *testing.T) {
	readmePath := filepath.Join("..", "..", "README.md")
	b, err := os.ReadFile(readmePath)
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	full := string(b)

	if !strings.Contains(full, "aidetector -version") {
		t.Errorf("README.md missing documentation for 'aidetector -version'")
	}
	if !strings.Contains(full, "- `build`") {
		t.Errorf("README.md missing bullet documenting the `build` JSON field")
	}
}
