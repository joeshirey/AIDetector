package engine

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/detect"
)

func assertSHA256ModelsID(t *testing.T, label, id string) {
	t.Helper()
	const prefix = "sha256:"
	if !strings.HasPrefix(id, prefix) {
		t.Errorf("%s = %q, want %q prefix", label, id, prefix)
		return
	}
	hexPart := strings.TrimPrefix(id, prefix)
	if len(hexPart) != 64 {
		t.Errorf("%s hex length = %d (%q), want 64", label, len(hexPart), id)
		return
	}
	for i := 0; i < len(hexPart); i++ {
		c := hexPart[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			t.Errorf("%s contains non-lowercase-hex byte %q in %q", label, c, id)
			return
		}
	}
}

// AC1 + AC5: Embedded models identifier has "sha256:" + 64 lowercase hex chars,
// is identical across two Default() calls, and is computed once at load time
// rather than re-reading the filesystem per Analyze/BuildInfo call.
func TestDefaultBuildInfoModelsStableAndLoadTime(t *testing.T) {
	t.Setenv("AIDETECTOR_SCORER_URL", "")

	e1 := Default()
	e2 := Default()
	bi1 := e1.BuildInfo()
	bi2 := e2.BuildInfo()

	assertSHA256ModelsID(t, "Default().BuildInfo().Models", bi1.Models)
	if bi1.Models != bi2.Models {
		t.Errorf("Default() models digest not stable: %q != %q", bi1.Models, bi2.Models)
	}
	if bi1.Version == "" {
		t.Errorf("Default().BuildInfo().Version is empty; want module version, devel, or unknown")
	}
	if bi1.GoVersion != runtime.Version() {
		t.Errorf("Default().BuildInfo().GoVersion = %q, want %q", bi1.GoVersion, runtime.Version())
	}

	// Verify load-time computation: mutating or deleting the overlay file in
	// tempDir after FromDir returns must not change BuildInfo().Models or Analyze.
	dir := t.TempDir()
	threshPath := filepath.Join(dir, "thresholds.json")
	customThresh := `{"pr":{"mixed":71,"likely_ai":86,"ai":94}}`
	if err := os.WriteFile(threshPath, []byte(customThresh), 0o644); err != nil {
		t.Fatal(err)
	}
	cand, err := FromDir(dir)
	if err != nil {
		t.Fatalf("FromDir: %v", err)
	}
	atLoad := cand.BuildInfo().Models
	assertSHA256ModelsID(t, "cand.BuildInfo().Models", atLoad)

	if err := os.Remove(threshPath); err != nil {
		t.Fatal(err)
	}
	if afterRemove := cand.BuildInfo().Models; afterRemove != atLoad {
		t.Errorf("BuildInfo().Models changed after overlay file removal: got %q, want %q", afterRemove, atLoad)
	}
	res, err := cand.Analyze(context.Background(), detect.Input{Text: "short sample text", Kind: detect.KindProse})
	if err != nil {
		t.Fatalf("cand.Analyze: %v", err)
	}
	if res.Build.Models != atLoad {
		t.Errorf("Analyze Result.Build.Models = %q after overlay removal, want load-time digest %q", res.Build.Models, atLoad)
	}
}

// AC1 + AC5: FromDir overlays produce the same digest when identical or when
// extra unloaded files exist, produce different digests when loaded model files
// change, and Heuristic() reports an empty Models field.
func TestFromDirModelsDigestOverlaysAndHeuristic(t *testing.T) {
	t.Setenv("AIDETECTOR_SCORER_URL", "")

	defID := Default().BuildInfo().Models
	assertSHA256ModelsID(t, "Default().BuildInfo().Models", defID)

	t.Run("empty_overlay_matches_default", func(t *testing.T) {
		e, err := FromDir(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		got := e.BuildInfo().Models
		assertSHA256ModelsID(t, "FromDir(empty).BuildInfo().Models", got)
		if got != defID {
			t.Errorf("empty FromDir digest = %q, want Default() %q", got, defID)
		}
	})

	t.Run("identical_thresholds_copy_matches_default", func(t *testing.T) {
		dir := t.TempDir()
		raw, err := EmbeddedFile("thresholds.json")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "thresholds.json"), raw, 0o644); err != nil {
			t.Fatal(err)
		}
		e, err := FromDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		got := e.BuildInfo().Models
		assertSHA256ModelsID(t, "FromDir(identical thresholds).BuildInfo().Models", got)
		if got != defID {
			t.Errorf("identical thresholds.json copy digest = %q, want Default() %q", got, defID)
		}
	})

	t.Run("unloaded_extra_files_do_not_change_digest", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("scratch benchmark notes\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "unused.json"), []byte(`{"ignored":true}`), 0o644); err != nil {
			t.Fatal(err)
		}
		// ShortModel.Max is 0 in shipped engine.json, so prose.short.json is not loaded.
		if err := os.WriteFile(filepath.Join(dir, "prose.short.json"), []byte(`{"kind":"prose","bias":1.2}`), 0o644); err != nil {
			t.Fatal(err)
		}
		e, err := FromDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		got := e.BuildInfo().Models
		assertSHA256ModelsID(t, "FromDir(unloaded files).BuildInfo().Models", got)
		if got != defID {
			t.Errorf("unloaded files changed digest: got %q, want Default() %q", got, defID)
		}
	})

	t.Run("modified_thresholds_and_pr_models_differ", func(t *testing.T) {
		threshDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(threshDir, "thresholds.json"), []byte(`{"pr":{"mixed":65,"likely_ai":80,"ai":90}}`), 0o644); err != nil {
			t.Fatal(err)
		}
		eThresh, err := FromDir(threshDir)
		if err != nil {
			t.Fatal(err)
		}
		threshID := eThresh.BuildInfo().Models
		assertSHA256ModelsID(t, "FromDir(modified thresholds)", threshID)
		if threshID == defID {
			t.Errorf("modified thresholds.json produced same digest as Default(): %q", threshID)
		}

		prDir := t.TempDir()
		candPR, err := json.Marshal(calibrate.Model{
			Kind:     "pr",
			Bias:     0.75,
			Features: []calibrate.Feature{},
			Missing:  map[string]float64{},
			Trained:  12,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(prDir, "pr.json"), candPR, 0o644); err != nil {
			t.Fatal(err)
		}
		ePR, err := FromDir(prDir)
		if err != nil {
			t.Fatal(err)
		}
		prID := ePR.BuildInfo().Models
		assertSHA256ModelsID(t, "FromDir(modified pr.json)", prID)
		if prID == defID {
			t.Errorf("modified pr.json produced same digest as Default(): %q", prID)
		}
		if prID == threshID {
			t.Errorf("modified pr.json and modified thresholds.json collided on digest %q", prID)
		}
	})

	t.Run("loaded_short_model_affects_digest", func(t *testing.T) {
		dirA := t.TempDir()
		dirB := t.TempDir()
		engCfg := []byte(`{"align_normalization":true,"short_model":{"max":120,"blend":40}}`)
		for _, d := range []string{dirA, dirB} {
			if err := os.WriteFile(filepath.Join(d, "engine.json"), engCfg, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(dirA, "prose.short.json"), []byte(`{"kind":"prose","bias":0.25}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dirB, "prose.short.json"), []byte(`{"kind":"prose","bias":0.95}`), 0o644); err != nil {
			t.Fatal(err)
		}
		eA, err := FromDir(dirA)
		if err != nil {
			t.Fatal(err)
		}
		eB, err := FromDir(dirB)
		if err != nil {
			t.Fatal(err)
		}
		assertSHA256ModelsID(t, "FromDir(shortA)", eA.BuildInfo().Models)
		assertSHA256ModelsID(t, "FromDir(shortB)", eB.BuildInfo().Models)
		if eA.BuildInfo().Models == eB.BuildInfo().Models {
			t.Errorf("loaded prose.short.json changes did not alter digest: %q", eA.BuildInfo().Models)
		}
	})

	t.Run("heuristic_has_empty_models", func(t *testing.T) {
		h := Heuristic().BuildInfo()
		if h.Models != "" {
			t.Errorf("Heuristic().BuildInfo().Models = %q, want empty string", h.Models)
		}
		if h.Version == "" {
			t.Errorf("Heuristic().BuildInfo().Version is empty; want runtime version or devel/unknown")
		}
		if h.GoVersion != runtime.Version() {
			t.Errorf("Heuristic().BuildInfo().GoVersion = %q, want %q", h.GoVersion, runtime.Version())
		}
	})
}

// AC2: parseBuildInfo handles tagged versions, (devel), 40-char vcs.revision
// shortening, -dirty suffix iff vcs.modified == "true", and clean degradation
// when !ok, bi == nil, or vcs.* settings are absent (e.g. Docker build without .git).
func TestParseBuildInfo(t *testing.T) {
	revA := strings.Repeat("a", 12) + strings.Repeat("0", 28)
	revB := strings.Repeat("b", 12) + strings.Repeat("0", 28)

	cases := []struct {
		name string
		bi   *debug.BuildInfo
		ok   bool
		want BuildInfo
	}{
		{
			name: "not_ok_nil",
			bi:   nil,
			ok:   false,
			want: BuildInfo{Version: "unknown"},
		},
		{
			name: "not_ok_non_nil",
			bi:   &debug.BuildInfo{GoVersion: "go1.27.3", Main: debug.Module{Version: "v1.0.0"}},
			ok:   false,
			want: BuildInfo{Version: "unknown"},
		},
		{
			name: "ok_with_nil_buildinfo",
			bi:   nil,
			ok:   true,
			want: BuildInfo{Version: "unknown"},
		},
		{
			name: "tagged_release_clean_vcs",
			bi: &debug.BuildInfo{
				GoVersion: "go1.27.3",
				Main:      debug.Module{Path: "github.com/joeshirey/AIDetector", Version: "v0.4.2"},
				Settings: []debug.BuildSetting{
					{Key: "vcs", Value: "git"},
					{Key: "vcs.revision", Value: revA},
					{Key: "vcs.modified", Value: "false"},
				},
			},
			ok: true,
			want: BuildInfo{
				Version:   "v0.4.2",
				Commit:    strings.Repeat("a", 12),
				GoVersion: "go1.27.3",
			},
		},
		{
			name: "devel_dirty_vcs",
			bi: &debug.BuildInfo{
				GoVersion: "go1.27.3",
				Main:      debug.Module{Path: "github.com/joeshirey/AIDetector", Version: "(devel)"},
				Settings: []debug.BuildSetting{
					{Key: "vcs.revision", Value: revB},
					{Key: "vcs.modified", Value: "true"},
				},
			},
			ok: true,
			want: BuildInfo{
				Version:   "devel",
				Commit:    strings.Repeat("b", 12) + "-dirty",
				GoVersion: "go1.27.3",
			},
		},
		{
			name: "short_revision_dirty_no_bounds_panic",
			bi: &debug.BuildInfo{
				GoVersion: "go1.27.3",
				Main:      debug.Module{Version: "(devel)"},
				Settings: []debug.BuildSetting{
					{Key: "vcs.revision", Value: "aaaaaaa"},
					{Key: "vcs.modified", Value: "true"},
				},
			},
			ok: true,
			want: BuildInfo{
				Version:   "devel",
				Commit:    "aaaaaaa-dirty",
				GoVersion: "go1.27.3",
			},
		},
		{
			name: "docker_build_without_vcs_settings",
			bi: &debug.BuildInfo{
				GoVersion: "go1.27.3",
				Main:      debug.Module{Version: "(devel)"},
				Settings: []debug.BuildSetting{
					{Key: "-compiler", Value: "gc"},
				},
			},
			ok: true,
			want: BuildInfo{
				Version:   "devel",
				Commit:    "",
				GoVersion: "go1.27.3",
			},
		},
		{
			name: "modified_without_revision_stays_empty_commit",
			bi: &debug.BuildInfo{
				GoVersion: "go1.27.3",
				Main:      debug.Module{Version: "(devel)"},
				Settings: []debug.BuildSetting{
					{Key: "vcs.modified", Value: "true"},
				},
			},
			ok: true,
			want: BuildInfo{
				Version:   "devel",
				Commit:    "",
				GoVersion: "go1.27.3",
			},
		},
		{
			name: "empty_module_version_falls_back_to_unknown",
			bi: &debug.BuildInfo{
				GoVersion: "go1.27.3",
				Main:      debug.Module{Version: ""},
			},
			ok: true,
			want: BuildInfo{
				Version:   "unknown",
				Commit:    "",
				GoVersion: "go1.27.3",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseBuildInfo(tc.bi, tc.ok)
			if got != tc.want {
				t.Errorf("parseBuildInfo() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// AC3 + AC6: Result.Build is populated on both normal Analyze and 0-word empty
// Analyze, marshals additively under "build" without changing existing Result
// fields, and omits empty optional fields on BuildInfo.
func TestAnalyzeIncludesBuildInfoAndJSONShape(t *testing.T) {
	t.Setenv("AIDETECTOR_SCORER_URL", "")

	e := Default()
	wantBuild := e.BuildInfo()
	assertSHA256ModelsID(t, "Default().BuildInfo().Models", wantBuild.Models)

	sample := "Yesterday morning I walked down to the hardware store on Elm Street, bought two brass hinges, " +
		"and spent an hour fixing the back porch screen door before rain rolled in from the bay. " +
		"The old screws had rusted tight over ten damp winters, so I soaked them in oil, pried the bent frame " +
		"straight with a flat bar, andplaned a sliver off the bottom edge until the latch clicked shut cleanly."

	res, err := e.Analyze(context.Background(), detect.Input{Text: sample, Kind: detect.KindProse})
	if err != nil {
		t.Fatalf("Analyze(prose): %v", err)
	}
	if res.Build != wantBuild {
		t.Errorf("Analyze(prose).Build = %+v, want %+v", res.Build, wantBuild)
	}

	// Empty / whitespace input hits the early return at WordCount == 0 and must still carry Build.
	emptyRes, err := e.Analyze(context.Background(), detect.Input{Text: "", Kind: detect.KindProse})
	if err != nil {
		t.Fatalf("Analyze(empty): %v", err)
	}
	if emptyRes.WordCount != 0 || emptyRes.Label != TooShort {
		t.Fatalf("Analyze(empty) = WordCount %d Label %q, want 0 / %q", emptyRes.WordCount, emptyRes.Label, TooShort)
	}
	if emptyRes.Build != wantBuild {
		t.Errorf("Analyze(empty).Build = %+v, want %+v", emptyRes.Build, wantBuild)
	}

	rawJSON, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("json.Marshal(Result): %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(rawJSON, &m); err != nil {
		t.Fatalf("json.Unmarshal(Result): %v", err)
	}
	for _, key := range []string{"score", "raw_score", "label", "confidence", "kind", "word_count", "reasons", "signals", "cutoffs", "build"} {
		if _, ok := m[key]; !ok {
			t.Errorf("Result JSON missing expected top-level key %q", key)
		}
	}
	buildObj, ok := m["build"].(map[string]any)
	if !ok {
		t.Fatalf("Result JSON \"build\" = %T (%v), want JSON object", m["build"], m["build"])
	}
	if gotModels, _ := buildObj["models"].(string); gotModels != wantBuild.Models {
		t.Errorf("Result JSON build.models = %q, want %q", gotModels, wantBuild.Models)
	}
	if gotVer, _ := buildObj["version"].(string); gotVer == "" || gotVer != wantBuild.Version {
		t.Errorf("Result JSON build.version = %q, want non-empty %q", gotVer, wantBuild.Version)
	}
	if gotGo, _ := buildObj["go_version"].(string); gotGo != runtime.Version() {
		t.Errorf("Result JSON build.go_version = %q, want %q", gotGo, runtime.Version())
	}

	// Verify omitempty on unstamped BuildInfo.
	minJSON, err := json.Marshal(BuildInfo{Version: "unknown"})
	if err != nil {
		t.Fatalf("json.Marshal(minimal BuildInfo): %v", err)
	}
	if string(minJSON) != `{"version":"unknown"}` {
		t.Errorf("minimal BuildInfo JSON = %s, want {\"version\":\"unknown\"}", minJSON)
	}
}
