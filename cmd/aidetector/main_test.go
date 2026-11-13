package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/detect/ngram"
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

// AC2: Every CLI invocation of aidetector fit or aidetector thresholds in
// README.md passes an explicit -out flag (and at least one exists), without
// coupling to surrounding README prose or ordering.
func TestReadmeRebuildDocs(t *testing.T) {
	readmePath := filepath.Join("..", "..", "README.md")
	b, err := os.ReadFile(readmePath)
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}

	inFence := false
	cmdLines := 0
	for lineNo, line := range strings.Split(string(b), "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			continue
		}
		isShellCmd := inFence || strings.HasPrefix(trimmed, "aidetector ") || strings.HasPrefix(trimmed, "bin/aidetector ")
		if !isShellCmd {
			continue
		}
		if idx := strings.Index(trimmed, " #"); idx >= 0 {
			trimmed = strings.TrimSpace(trimmed[:idx])
		}
		if strings.Contains(trimmed, "aidetector fit ") || strings.HasSuffix(trimmed, "aidetector fit") ||
			strings.Contains(trimmed, "aidetector thresholds ") || strings.HasSuffix(trimmed, "aidetector thresholds") {
			cmdLines++
			if !strings.Contains(trimmed, "-out ") && !strings.Contains(trimmed, "-out=") {
				t.Errorf("README.md:%d invokes fit/thresholds without explicit -out: %s", lineNo+1, trimmed)
			}
		}
	}
	if cmdLines == 0 {
		t.Fatal("README.md contains no aidetector fit or aidetector thresholds command lines")
	}
}

// AC1: aidetector fit prints the engine.Options used (align_normalization and
// short_prose) and where they came from (embedded engine.json, <out>/engine.json,
// and/or CLI flag overrides) at the start of fit; aidetector fit -h describes
// -align-norm with "default: from engine.json" without "(default false)", while
// -align-norm, -align-norm=true, and -align-norm=false continue to work identically.
func TestFitEngineOptionsProvenanceAndAlignNormHelp(t *testing.T) {
	workDir := t.TempDir()
	corpusDir := buildSyntheticProseCorpus(t)

	// 1. Fresh -out directory with no CLI overrides -> embedded engine.json.
	freshOut := filepath.Join(t.TempDir(), "fresh_out")
	stdout, stderr, code := runCLI(t, workDir, "fit", "-kind", "prose", "-out", freshOut, corpusDir)
	if code != 0 {
		t.Fatalf("fit freshOut failed (exit %d): %s", code, stderr)
	}
	engIdx := strings.Index(stdout, "engine options")
	wpIdx := strings.Index(stdout, "word-pattern model:")
	if engIdx < 0 || wpIdx < 0 || !(engIdx < wpIdx) {
		t.Errorf("expected engine options line printed at start of fit before word-pattern model, got stdout:\n%s", stdout)
	}
	for _, want := range []string{"align_normalization=true", "short_prose=150:350", "embedded engine.json"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("fresh fit stdout missing %q:\n%s", want, stdout)
		}
	}

	// 2. CLI overrides (-align-norm=false, -short-prose 0:0) into reusedOut.
	reusedOut := filepath.Join(t.TempDir(), "reused_out")
	stdoutOverride, stderr, code := runCLI(t, workDir, "fit", "-kind", "prose",
		"-align-norm=false", "-short-prose", "0:0", "-out", reusedOut, corpusDir)
	if code != 0 {
		t.Fatalf("fit override failed (exit %d): %s", code, stderr)
	}
	for _, want := range []string{"align_normalization=false", "short_prose=0:0", "-align-norm", "-short-prose"} {
		if !strings.Contains(stdoutOverride, want) {
			t.Errorf("override fit stdout missing %q:\n%s", want, stdoutOverride)
		}
	}

	// 3. Reusing reusedOut WITHOUT flags must visibly report inheritance from <reusedOut>/engine.json.
	reusedEnginePath := filepath.Join(reusedOut, "engine.json")
	stdoutReused, stderr, code := runCLI(t, workDir, "fit", "-kind", "prose", "-out", reusedOut, corpusDir)
	if code != 0 {
		t.Fatalf("fit reusedOut failed (exit %d): %s", code, stderr)
	}
	for _, want := range []string{"align_normalization=false", "short_prose=0:0", reusedEnginePath} {
		if !strings.Contains(stdoutReused, want) {
			t.Errorf("reused fit stdout missing %q:\n%s", want, stdoutReused)
		}
	}
	if strings.Contains(stdoutReused, "embedded engine.json") {
		t.Errorf("reused fit stdout falsely claimed embedded engine.json:\n%s", stdoutReused)
	}

	// 4. Bare -align-norm and -align-norm=true override align_normalization back to true while preserving short_prose=0:0 from <reusedOut>/engine.json.
	for _, flagForm := range []string{"-align-norm", "-align-norm=true"} {
		subOut := filepath.Join(t.TempDir(), "sub_"+strings.ReplaceAll(flagForm, "=", "_"))
		writeSyntheticFile(t, filepath.Join(subOut, "engine.json"), `{"align_normalization": false, "short_prose": {"below": 110, "above": 260}}`, time.Time{})
		outStr, errStr, exit := runCLI(t, workDir, "fit", "-kind", "prose", flagForm, "-out", subOut, corpusDir)
		if exit != 0 {
			t.Fatalf("fit %s failed (exit %d): %s", flagForm, exit, errStr)
		}
		for _, want := range []string{"align_normalization=true", "short_prose=110:260", filepath.Join(subOut, "engine.json"), "-align-norm"} {
			if !strings.Contains(outStr, want) {
				t.Errorf("fit %s stdout missing %q:\n%s", flagForm, want, outStr)
			}
		}
		b, err := os.ReadFile(filepath.Join(subOut, "engine.json"))
		if err != nil {
			t.Fatal(err)
		}
		var gotOpts engine.Options
		if err := json.Unmarshal(b, &gotOpts); err != nil {
			t.Fatal(err)
		}
		if !gotOpts.AlignNormalization || gotOpts.ShortProse.Below != 110 || gotOpts.ShortProse.Above != 260 {
			t.Errorf("fit %s wrote engine.json = %+v, want AlignNormalization=true ShortProse={110 260}", flagForm, gotOpts)
		}
	}

	// 5. Help text for aidetector fit -h must say "default: from engine.json" and never emit "(default false)".
	_, helpErr, _ := runCLI(t, workDir, "fit", "-h")
	if !strings.Contains(helpErr, "default: from engine.json") {
		t.Errorf("fit -h stderr missing 'default: from engine.json':\n%s", helpErr)
	}
	if strings.Contains(helpErr, "(default false)") {
		t.Errorf("fit -h stderr emitted misleading '(default false)':\n%s", helpErr)
	}
}

// AC2: aidetector thresholds prints whether calibration texts were scored with
// embedded production models or candidate models loaded from <modelDir>.
func TestThresholdsPrintsModelSource(t *testing.T) {
	workDir := t.TempDir()
	poolDir := filepath.Join(t.TempDir(), "pr_pool")
	prBody := "Fix nil pointer dereference when closing idle HTTP keep-alive connections after server shutdown timeout expires in worker pool.\n" +
		"Added a unit test reproducing the race under load and verified connection draining completes cleanly without leaking goroutines.\n"
	for _, id := range []string{"sample-002", "sample-008", "sample-001", "sample-004"} {
		writeSyntheticFile(t, filepath.Join(poolDir, id+".txt"), prBody, time.Time{})
	}

	// 1. Fresh non-existent -out directory -> embedded production models.
	freshOut := filepath.Join(t.TempDir(), "nonexistent_dir")
	stdoutFresh, stderr, code := runCLI(t, workDir, "thresholds", "-kind", "pr", "-only", "documents", "-out", freshOut, poolDir)
	if code != 0 {
		t.Fatalf("thresholds freshOut failed (exit %d): %s", code, stderr)
	}
	if !strings.Contains(stdoutFresh, "embedded production models") {
		t.Errorf("fresh non-existent -out stdout = %q, want mention of 'embedded production models'", stdoutFresh)
	}
	if strings.Contains(stdoutFresh, "models loaded from") {
		t.Errorf("fresh non-existent -out stdout = %q, must not claim models loaded from dir", stdoutFresh)
	}

	// 2. Empty existing scratch -out directory (e.g. mktemp -d) -> still embedded production models.
	emptyOut := t.TempDir()
	stdoutEmpty, stderr, code := runCLI(t, workDir, "thresholds", "-kind", "pr", "-only", "documents", "-out", emptyOut, poolDir)
	if code != 0 {
		t.Fatalf("thresholds emptyOut failed (exit %d): %s", code, stderr)
	}
	if !strings.Contains(stdoutEmpty, "embedded production models") {
		t.Errorf("empty scratch -out stdout = %q, want mention of 'embedded production models'", stdoutEmpty)
	}

	// 3. Candidate directory containing pr.json -> models loaded from <candDir>.
	candDir := filepath.Join(t.TempDir(), "cand_models")
	candModel := calibrate.Model{Kind: "pr", Bias: 4.5, Features: []calibrate.Feature{}, Missing: map[string]float64{}, Trained: 10}
	b, _ := json.MarshalIndent(candModel, "", " ")
	writeSyntheticFile(t, filepath.Join(candDir, "pr.json"), string(b)+"\n", time.Time{})

	stdoutCand, stderr, code := runCLI(t, workDir, "thresholds", "-kind", "pr", "-only", "documents", "-out", candDir, poolDir)
	if code != 0 {
		t.Fatalf("thresholds candDir failed (exit %d): %s", code, stderr)
	}
	if !strings.Contains(stdoutCand, "models loaded from "+candDir) {
		t.Errorf("candidate -out stdout = %q, want 'models loaded from %s'", stdoutCand, candDir)
	}
	if strings.Contains(stdoutCand, "embedded production models") {
		t.Errorf("candidate -out stdout = %q, must not claim embedded production models", stdoutCand)
	}
}

// AC4: aidetector fit -exclude-since inside a .../ai/human/... parent directory
// must not misclassify human files as AI generations.
func TestFitExcludeSinceIgnoresParentAIDir(t *testing.T) {
	workDir := t.TempDir()
	corpusDir := filepath.Join(t.TempDir(), "ai", "human", "pr_corpus")

	before := time.Date(2026, 11, 5, 7, 0, 0, 0, time.UTC)
	after := time.Date(2026, 11, 5, 10, 0, 0, 0, time.UTC)
	cutoff := "2026-11-05T08:30:00Z"

	humanPR := "Fix race condition in connection pool cleanup when context cancels early during TLS handshake retry.\n" +
		"Verified with go test -race across twenty iterations on local loopback sockets.\n"
	aiPR := "This pull request resolves a concurrency issue in connection lifecycle management, ensuring deterministic resource cleanup and improved reliability across retry paths.\n" +
		"Comprehensive unit tests validate graceful cancellation semantics.\n"

	// 4 train human files (2 before, 2 after cutoff — all 4 must stay in training).
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-001.txt"), humanPR+" alpha", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-002.txt"), humanPR+" beta", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-003.txt"), humanPR+" gamma", after)
	writeSyntheticFile(t, filepath.Join(corpusDir, "human", "sample-004.txt"), humanPR+" delta", after)

	// 4 pre-cutoff train AI files (kept) + 2 post-cutoff train AI files (excluded).
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-001.txt"), aiPR+" alpha", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-002.txt"), aiPR+" beta", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-003.txt"), aiPR+" gamma", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-004.txt"), aiPR+" delta", before)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-005.txt"), aiPR+" epsilon", after)
	writeSyntheticFile(t, filepath.Join(corpusDir, "ai", "gemini-3.8-flash", "sample-006.txt"), aiPR+" zeta", after)

	outDir := filepath.Join(t.TempDir(), "nested_filtered_out")
	_, stderr, code := runCLI(t, workDir, "fit", "-kind", "pr", "-exclude-since", cutoff, "-out", outDir, corpusDir)
	if code != 0 {
		t.Fatalf("fit -exclude-since under /ai/human/ failed (exit %d): %s", code, stderr)
	}
	m := readCalibrateModel(t, filepath.Join(outDir, "pr.json"))
	if m.Trained != 8 {
		t.Errorf("Trained under /ai/human/ parent = %d, want 8 (4 train human + 4 pre-cutoff train AI)", m.Trained)
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

// AC6: aidetector detect version scores a file literally named "version" (whereas
// aidetector version prints build version), and usage() documents detect version.
func TestCLIDetectFileNamedVersion(t *testing.T) {
	t.Setenv("AIDETECTOR_SCORER_URL", "")
	workDir := t.TempDir()

	prose := "Yesterday afternoon I patched three cedar shingles above the back porch overhang, " +
		"swept the pine needles out of the copper gutter, and sealed the corner seam before dusk. " +
		"The old flashing had worked loose during January windstorms, so I replaced two rusted nails " +
		"and checked the attic rafters with a flashlight when the rain started.\n"
	writeSyntheticFile(t, filepath.Join(workDir, "version"), prose, time.Time{})

	// Bare "aidetector version" still prints the build/models version header.
	verOut, verErr, verCode := runCLI(t, workDir, "version")
	if verCode != 0 || verErr != "" || !strings.HasPrefix(verOut, "aidetector ") {
		t.Fatalf("aidetector version = (code=%d, stdout=%q, stderr=%q), want exit 0 version output", verCode, verOut, verErr)
	}

	// Explicit "aidetector detect version" (and "-json") scores the file named "version".
	detOut, detErr, detCode := runCLI(t, workDir, "detect", "version")
	if detCode != 0 || detErr != "" || !strings.Contains(detOut, "Score:") {
		t.Fatalf("aidetector detect version = (code=%d, stdout=%q, stderr=%q), want exit 0 score output", detCode, detOut, detErr)
	}
	jsonOut, jsonErr, jsonCode := runCLI(t, workDir, "detect", "-json", "version")
	if jsonCode != 0 || jsonErr != "" {
		t.Fatalf("aidetector detect -json version failed (code=%d): %s", jsonCode, jsonErr)
	}
	var res engine.Result
	if err := json.Unmarshal([]byte(jsonOut), &res); err != nil {
		t.Fatalf("unmarshal detect -json version: %v", err)
	}
	if res.WordCount < 50 || res.Kind != "prose" {
		t.Errorf("detect -json version Result = (Kind=%q, WordCount=%d), want Kind=prose WordCount>=50", res.Kind, res.WordCount)
	}

	_, helpErr, helpCode := runCLI(t, workDir, "-h")
	if helpCode != 0 {
		t.Fatalf("aidetector -h exited %d, want 0", helpCode)
	}
	if !strings.Contains(helpErr, "detect version") {
		t.Errorf("usage() output missing 'aidetector detect version' edge-case note:\n%s", helpErr)
	}
}

// AC6: README.md documents aidetector -version, aidetector detect version, the
// Result.build JSON field, and the Docker build context .git / commit: unknown note.
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
	if !strings.Contains(full, "aidetector detect version") {
		t.Errorf("README.md missing documentation for 'aidetector detect version' to score a file named version")
	}
	if !strings.Contains(full, "- `build`") {
		t.Errorf("README.md missing bullet documenting the `build` JSON field")
	}
	if !strings.Contains(full, "Docker") || !strings.Contains(full, ".git") || !strings.Contains(full, "commit: unknown") {
		t.Errorf("README.md missing Docker note explaining builds without .git report 'commit: unknown'")
	}
}

// AC6 + CLI polish: README.md's fit -lm section documents both <kind>.lm.json /
// prose.lm.json and <kind>.lm.ngram.json / prose.lm.ngram.json; aidetector fit -h
// mentions lm.ngram.json; aidetector fit -lm writes prose.lm.ngram.json without
// touching prose.ngram.json; and aidetector thresholds -out <dir> recognizes
// prose.lm.ngram.json and pr.lm.ngram.json in candidate directories.
func TestCLIFitLMThresholdsAndReadmeLMNgramDocs(t *testing.T) {
	workDir := t.TempDir()

	// 1. README.md fit -lm section must document both <kind>.lm.json / prose.lm.json and <kind>.lm.ngram.json / prose.lm.ngram.json.
	readmeBytes, err := os.ReadFile(filepath.Join("..", "..", "README.md"))
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	readme := string(readmeBytes)
	secStart := strings.Index(readme, "## Optional: token-probability scoring")
	if secStart < 0 {
		t.Fatalf("README.md missing '## Optional: token-probability scoring' section")
	}
	sec := readme[secStart:]
	if nextSec := strings.Index(sec[1:], "\n## "); nextSec >= 0 {
		sec = sec[:nextSec+1]
	}
	hasLMJSON := strings.Contains(sec, "prose.lm.json") || strings.Contains(sec, "<kind>.lm.json")
	hasLMNgramJSON := strings.Contains(sec, "prose.lm.ngram.json") || strings.Contains(sec, "<kind>.lm.ngram.json")
	if !hasLMJSON || !hasLMNgramJSON {
		t.Errorf("README.md token-probability scoring section must mention both prose.lm.json/<kind>.lm.json and prose.lm.ngram.json/<kind>.lm.ngram.json; got:\n%s", sec)
	}

	// 2. aidetector fit -h must mention lm.ngram.json alongside KIND.lm.json.
	_, helpErr, _ := runCLI(t, workDir, "fit", "-h")
	if !strings.Contains(helpErr, "lm.json") || !strings.Contains(helpErr, "lm.ngram.json") {
		t.Errorf("aidetector fit -h stderr must mention both KIND.lm.json and KIND.lm.ngram.json; got:\n%s", helpErr)
	}

	// 3. aidetector thresholds -out <dir> must recognize prose.lm.ngram.json and pr.lm.ngram.json in candidate directories.
	prosePoolDir := filepath.Join(t.TempDir(), "prose_pool")
	proseBody := "Yesterday afternoon I patched three cedar shingles above the back porch overhang, " +
		"swept the pine needles out of the copper gutter, and sealed the corner seam before dusk. " +
		"The old flashing had worked loose during January windstorms, so I replaced two rusted nails " +
		"and checked the attic rafters with a flashlight when the rain started.\n"
	prPoolDir := filepath.Join(t.TempDir(), "pr_pool")
	prBody := "Fix nil pointer dereference when closing idle HTTP keep-alive connections after server shutdown timeout expires in worker pool.\n" +
		"Added a unit test reproducing the race under load and verified connection draining completes cleanly without leaking goroutines.\n"
	for _, id := range []string{"sample-002", "sample-008", "sample-001", "sample-004"} {
		writeSyntheticFile(t, filepath.Join(prosePoolDir, id+".txt"), proseBody, time.Time{})
		writeSyntheticFile(t, filepath.Join(prPoolDir, id+".txt"), prBody, time.Time{})
	}

	t.Setenv("AIDETECTOR_SCORER_URL", "")
	candProseNgramDir := filepath.Join(t.TempDir(), "cand_prose_lm_ngram")
	proseLMNgramBytes, _ := json.MarshalIndent(ngram.Model{Kind: "prose", Bias: 0.5, Weights: map[string]float64{"cedar": 1.0}}, "", " ")
	writeSyntheticFile(t, filepath.Join(candProseNgramDir, "prose.lm.ngram.json"), string(proseLMNgramBytes)+"\n", time.Time{})

	stdoutProseThresh, stderrProseThresh, codeProseThresh := runCLI(t, workDir, "thresholds", "-kind", "prose", "-only", "documents", "-out", candProseNgramDir, prosePoolDir)
	if codeProseThresh != 0 {
		t.Fatalf("thresholds prose.lm.ngram.json failed (exit %d): %s", codeProseThresh, stderrProseThresh)
	}
	if !strings.Contains(stdoutProseThresh, "models loaded from "+candProseNgramDir) || !strings.Contains(stdoutProseThresh, "prose.lm.ngram.json") {
		t.Errorf("thresholds -out with prose.lm.ngram.json stdout = %q, want 'models loaded from %s (... prose.lm.ngram.json ...)'",
			stdoutProseThresh, candProseNgramDir)
	}

	candPRNgramDir := filepath.Join(t.TempDir(), "cand_pr_lm_ngram")
	prLMNgramBytes, _ := json.MarshalIndent(ngram.Model{Kind: "pr", Bias: 0.5, Weights: map[string]float64{"idle": 1.0}}, "", " ")
	writeSyntheticFile(t, filepath.Join(candPRNgramDir, "pr.lm.ngram.json"), string(prLMNgramBytes)+"\n", time.Time{})

	stdoutPRThresh, stderrPRThresh, codePRThresh := runCLI(t, workDir, "thresholds", "-kind", "pr", "-only", "documents", "-out", candPRNgramDir, prPoolDir)
	if codePRThresh != 0 {
		t.Fatalf("thresholds pr.lm.ngram.json failed (exit %d): %s", codePRThresh, stderrPRThresh)
	}
	if !strings.Contains(stdoutPRThresh, "models loaded from "+candPRNgramDir) || !strings.Contains(stdoutPRThresh, "pr.lm.ngram.json") {
		t.Errorf("thresholds -out with pr.lm.ngram.json stdout = %q, want 'models loaded from %s (... pr.lm.ngram.json ...)'",
			stdoutPRThresh, candPRNgramDir)
	}

	// 4. End-to-end aidetector fit -lm + thresholds with synthetic httptest scorer.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "fake-observer+fake-performer", "max_tokens": 512})
		case r.Method == http.MethodPost && r.URL.Path == "/score":
			var req struct {
				Texts []string `json:"texts"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			results := make([]map[string]any, len(req.Texts))
			for i := range req.Texts {
				results[i] = map[string]any{"tokens": 64, "log_ppl": 1.85, "x_ppl": 2.35, "binoculars": 0.787, "curvature": 1.45}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "fake-observer+fake-performer", "results": results})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	t.Setenv("AIDETECTOR_SCORER_URL", srv.URL)

	corpusDir := buildSyntheticProseCorpus(t)
	lmFitDir := filepath.Join(t.TempDir(), "cli_lm_fit")
	sentinelBytes := []byte("{\n \"kind\": \"prose\",\n \"bias\": -0.9876,\n \"weights\": {\n  \"sentinel_cli_term\": -2.5\n }\n}\n")
	writeSyntheticFile(t, filepath.Join(lmFitDir, "prose.ngram.json"), string(sentinelBytes), time.Time{})

	_, fitStderr, fitCode := runCLI(t, workDir, "fit", "-lm", "-kind", "prose", "-out", lmFitDir, corpusDir)
	if fitCode != 0 {
		t.Fatalf("aidetector fit -lm failed (exit %d): %s", fitCode, fitStderr)
	}
	if _, err := os.Stat(filepath.Join(lmFitDir, "prose.lm.ngram.json")); err != nil {
		t.Errorf("aidetector fit -lm did not write prose.lm.ngram.json: %v", err)
	}
	gotSentinel, err := os.ReadFile(filepath.Join(lmFitDir, "prose.ngram.json"))
	if err != nil {
		t.Fatalf("read pre-existing prose.ngram.json: %v", err)
	}
	if !bytes.Equal(gotSentinel, sentinelBytes) {
		t.Errorf("aidetector fit -lm overwrote pre-existing prose.ngram.json")
	}

	threshOut, threshErr, threshCode := runCLI(t, workDir, "thresholds", "-kind", "prose", "-only", "documents", "-out", lmFitDir, prosePoolDir)
	if threshCode != 0 {
		t.Fatalf("aidetector thresholds on lmFitDir failed (exit %d): %s", threshCode, threshErr)
	}
	if !strings.Contains(threshOut, "prose.lm.ngram.json") {
		t.Errorf("aidetector thresholds stdout missing prose.lm.ngram.json in candidate file list:\n%s", threshOut)
	}
}
