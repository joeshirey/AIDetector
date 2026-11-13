package overnight

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/fixtures"
	"github.com/joeshirey/AIDetector/internal/split"
)

func passingFixtures() fixtures.CheckResult { return fixtures.CheckResult{} }

// synthMetrics builds metrics with per-sample data: a pool of n human
// documents (fp of them at 60+, fp80 at 80+, each with 3 segments) and a
// test set of human/AI raw scores.
func synthMetrics(n, fp, fp80 int, human, ai []float64) *Metrics {
	m := &Metrics{Test: map[string]AR{}, PoolDocs: map[string][]PoolDoc{}, Frozen: map[string]AR{}, Slices: map[string]AR{},
		Fixtures: passingFixtures()}
	for i := range n {
		d := PoolDoc{ID: fmt.Sprintf("doc%04d", i), Score: 10, Segs: []int{10, 10, 10}}
		if i < fp {
			d.Score, d.Segs[0] = 65, 65
		}
		if i < fp80 {
			d.Score = 85
		}
		m.PoolDocs["fresh-blog"] = append(m.PoolDocs["fresh-blog"], d)
	}
	ar := AR{}
	for i, v := range human {
		ar.S = append(ar.S, RankSample{Key: fmt.Sprintf("h%d", i), Group: fmt.Sprintf("s%d", i), Raw: v})
	}
	for i, v := range ai {
		ar.S = append(ar.S, RankSample{Key: fmt.Sprintf("a%d", i), Group: fmt.Sprintf("s%d", i), AI: true, Raw: v})
	}
	m.Test["prose"] = ar
	return m
}

func scores(r *rand.Rand, n int, mean float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = mean + r.NormFloat64()*15
	}
	return out
}

func TestGate(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	h, a := scores(r, 400, 20), scores(r, 400, 75)
	base := synthMetrics(1000, 20, 5, h, a)
	if _, ok := Gate(base, synthMetrics(1000, 20, 5, h, a)); !ok {
		t.Fatal("identical candidate should pass")
	}
	if _, ok := Gate(base, synthMetrics(1000, 21, 5, h, a)); !ok {
		t.Error("one extra false positive in 1000 should pass")
	}
	if _, ok := Gate(base, synthMetrics(1000, 60, 5, h, a)); ok {
		t.Error("false positives tripling (2% -> 6%) should fail")
	}
	worseAI := make([]float64, len(a))
	for i, v := range a {
		worseAI[i] = v - 25
	}
	if _, ok := Gate(base, synthMetrics(1000, 20, 5, h, worseAI)); ok {
		t.Error("a large AUROC drop should fail")
	}
	noFrozen := synthMetrics(1000, 20, 5, h, a)
	withFrozen := synthMetrics(1000, 20, 5, h, a)
	withFrozen.Frozen["x/prose"] = withFrozen.Test["prose"]
	if _, ok := Gate(withFrozen, noFrozen); ok {
		t.Error("a frozen set missing from the candidate should fail")
	}
	missingFix := synthMetrics(1000, 20, 5, h, a)
	missingFix.Fixtures.Missing = 3
	if _, ok := Gate(base, missingFix); ok {
		t.Error("missing fixtures should fail")
	}
	// Neutral candidates: every score jittered independently, nothing
	// systematically worse. The Holm-corrected gate should reject few.
	rejected := 0
	const trials = 30
	for range trials {
		jit := func(xs []float64) []float64 {
			out := make([]float64, len(xs))
			for i, v := range xs {
				out[i] = v + r.NormFloat64()*3
			}
			return out
		}
		c := synthMetrics(1000, 20, 5, jit(h), jit(a))
		// Pool: move a few documents across 60 in both directions.
		for k := range 3 {
			i := r.IntN(1000)
			if c.PoolDocs["fresh-blog"][i].Score < 60 {
				c.PoolDocs["fresh-blog"][i].Score = 61
			}
			c.PoolDocs["fresh-blog"][k].Score = 50
		}
		if _, ok := Gate(base, c); !ok {
			rejected++
		}
	}
	t.Logf("neutral candidates rejected %d/%d", rejected, trials)
	if rejected > 4 { // ~5% expected; 4/30 leaves room for sampling noise
		t.Errorf("neutral candidates rejected %d/%d times", rejected, trials)
	}
}

func TestReqMetaRoundTrip(t *testing.T) {
	m := ReqMeta{Track: "train", Genre: "techblog", Style: StyleContinue, Band: 2, ID: "medium-abc|weird"}
	got, err := DecodeReq(m.Encode())
	if err != nil || got != m {
		t.Fatalf("round trip: %+v, %v", got, err)
	}
}

func TestPromptsPassCheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	text := "First paragraph about the topic at hand.\n\nSecond paragraph with more detail and a few more words.\n\nThird paragraph closing it out."
	for _, g := range Genres {
		for _, s := range Styles {
			sm := corpus.Sample{ID: "x", Text: text, Context: map[string]string{"title": "T", "diff": "+a\n-b"}}
			p, _, err := BuildPrompt(g, s, sm, "a brief", rng)
			if err != nil {
				t.Errorf("%s/%s: %v\n%s", g.Name, s, err, p)
			}
		}
	}
}

func TestPromptCheckIgnoresContent(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	g, _ := GenreByName("techblog")
	sm := corpus.Sample{ID: "x", Text: "Humanizer is a .NET library that makes strings readable.\n\nMore text here."}
	for _, s := range Styles {
		if _, _, err := BuildPrompt(g, s, sm, "A post about the Humanizer library", rng); err != nil {
			t.Errorf("%s: content about Humanizer rejected: %v", s, err)
		}
	}
}

func TestSampleNamesShareSplit(t *testing.T) {
	for _, id := range []string{"a1", "medium-123", "pr-golang-go-555"} {
		for _, s := range Styles {
			if split.Of(SampleName(id, s)) != split.Of(id) {
				t.Errorf("%s~%s changed split", id, s)
			}
		}
	}
}

func TestFinalHoldoutShare(t *testing.T) {
	n, test, both := 20000, 0, 0
	for i := range n {
		id := "pool-" + string(rune('a'+i%26)) + itoa(i)
		if FinalHoldout(id) {
			test++
			if split.Of(id) == "test" {
				both++
			}
		}
	}
	if share := float64(test) / float64(n); share < 0.09 || share > 0.11 {
		t.Errorf("holdout share %.3f, want ~0.10", share)
	}
	// Independent of the train/test split: about 20% of the holdout is test.
	if share := float64(both) / float64(test); share < 0.16 || share > 0.24 {
		t.Errorf("holdout ∩ test share %.3f, want ~0.20", share)
	}
}

func itoa(i int) string {
	b := []byte{}
	for {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
		if i == 0 {
			return string(b)
		}
	}
}

func TestBanditFavorsMisses(t *testing.T) {
	b := LoadBandit(filepath.Join(t.TempDir(), "none.json"), 3)
	hot := Cell{Genre: "memo", Band: 1, Style: StyleEdit, Gen: "g"}
	cold := Cell{Genre: "blog", Band: 0, Style: StylePlain, Gen: "g"}
	for range 40 {
		b.Observe(hot, true)
		b.Observe(cold, false)
	}
	hits := 0
	for range 1000 {
		if b.Pick([]Cell{cold, hot}, 0.2) == hot {
			hits++
		}
	}
	// 80% exploit (nearly always hot) + 20% uniform (half hot) ≈ 90%.
	if hits < 850 || hits > 950 {
		t.Errorf("hot picked %d/1000, want ~900", hits)
	}
}

func TestGenSharesFloor(t *testing.T) {
	b := LoadBandit(filepath.Join(t.TempDir(), "none.json"), 4)
	for range 200 {
		b.Observe(Cell{Gen: "a"}, true)
		b.Observe(Cell{Gen: "d"}, false)
	}
	s := b.GenShares(map[string]float64{"a": 0.45, "b": 0.25, "c": 0.25, "d": 0.05}, PaidFloor, 2)
	sum := 0.0
	for g, v := range s {
		sum += v
		if v < PaidFloor*0.95 {
			t.Errorf("%s share %.3f under the floor", g, v)
		}
	}
	if sum < 0.999 || sum > 1.001 {
		t.Errorf("shares sum to %.3f", sum)
	}
}

func TestKSAndNaiveBayes(t *testing.T) {
	if d := ksStat([]float64{1, 2, 3}, []float64{1, 2, 3}); d != 0 {
		t.Errorf("identical KS = %v", d)
	}
	if d := ksStat([]float64{1, 2, 3}, []float64{10, 11, 12}); d != 1 {
		t.Errorf("disjoint KS = %v", d)
	}
	nb := newNaiveBayes()
	for range 5 {
		nb.add("memo", "hi team please review the budget by friday thanks regards")
		nb.add("essay", "in conclusion society must consider the argument that education matters")
	}
	if c, p := nb.classify("team please send the budget friday"); c != "memo" || p < 0.9 {
		t.Errorf("classify = %s %.2f", c, p)
	}
}

func TestVerifyDetectsChanges(t *testing.T) {
	root := t.TempDir()
	d := Data{Root: root}
	dir := filepath.Join(root, "corpus_frozen", "blog", "human")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("one"), 0o644)
	files, err := d.collect(nil, filepath.Join(root, "corpus_frozen"))
	if err != nil {
		t.Fatal(err)
	}
	m := &Manifest{Sets: map[string]*FrozenSet{"frozen-sources": {Sealed: true, Files: files}}}
	vs, _ := d.Verify(m, "", false)
	if len(vs) != 1 || !vs[0].OK() {
		t.Fatalf("clean verify: %+v", vs)
	}
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("two"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("new"), 0o644)
	vs, _ = d.Verify(m, "", false)
	if vs[0].Changed != 1 || vs[0].Added != 1 {
		t.Fatalf("want 1 changed, 1 added: %+v", vs)
	}
}

func TestSpendForCountsBatchRequests(t *testing.T) {
	es := []corpus.Entry{
		{Night: "n", Model: "m", Mode: corpus.ModeBatch, Purpose: "batch reservation", Cost: 1},
		{Night: "n", Model: "m", Mode: corpus.ModeBatch, Purpose: "batch correction", Cost: -0.2, N: 10, Usage: corpus.Usage{Input: 1000, Output: 500}},
		{Night: "other", Model: "m", Cost: 5},
	}
	s := SpendFor(es, "n")
	if s.Total < 0.799 || s.Total > 0.801 || s.Calls["m@batch"] != 10 || s.Usage["m@batch"].Output != 500 {
		t.Fatalf("%+v", s)
	}
}

func TestEditsStayOutOfTraining(t *testing.T) {
	d := Data{Root: "/data"}
	g, _ := GenreByName("blog")
	if p := filepath.ToSlash(d.OutputPath(g, "m", "x", StyleEdit)); !strings.Contains(p, "/ai_edited/") {
		t.Errorf("edit output %s should be outside ai/", p)
	}
	if p := filepath.ToSlash(d.OutputPath(g, "m", "x", StylePlain)); !strings.Contains(p, "/ai/") {
		t.Errorf("plain output %s should be in ai/", p)
	}
	p := &Planner{Sources: map[string]map[int][]Source{"blog": {0: {{}}}}}
	for _, c := range p.cellsFor("m") {
		if c.Style == StyleEdit {
			t.Fatal("planner offered an edit cell")
		}
	}
}

func TestCountTraining(t *testing.T) {
	d := Data{Root: t.TempDir()}
	if got := countTraining(d); got != 0 {
		t.Fatalf("empty countTraining = %d, want 0", got)
	}
	for _, id := range []string{"doc-1", "doc-2", "doc-3"} {
		if !Trainable(id) {
			t.Fatalf("fixture %s must be train split", id)
		}
	}
	if Trainable("doc-4") {
		t.Fatal("fixture doc-4 must be test split")
	}

	write := func(p string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("sample text\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	blog, _ := GenreByName("blog")
	techblog, _ := GenreByName("techblog")
	pr, _ := GenreByName("pr")

	// Counted: 4 train-split .txt files under human/ and ai/ across genres.
	write(filepath.Join(d.Dir(blog), "human", "doc-1.txt"))
	write(d.OutputPath(blog, "m", "doc-1", StylePlain))
	write(d.OutputPath(techblog, "m", "doc-2", StylePersona))
	write(filepath.Join(d.Dir(pr), "human", "doc-3.txt"))

	// Excluded: eval-only sets, unpromoted pool, test-split files, and non-.txt metadata.
	write(d.OutputPath(blog, "m", "doc-1", StyleEdit))
	write(filepath.Join(d.Dir(blog), "ai_polished", "m", "doc-1.txt"))
	write(filepath.Join(d.Dir(blog), "ai_humanized", "m", "doc-1.txt"))
	write(filepath.Join(d.Dir(blog), "pool", "doc-1.txt"))
	write(filepath.Join(d.Dir(blog), "human", "doc-4.txt"))
	write(d.OutputPath(blog, "m", "doc-4", StylePlain))
	write(filepath.Join(d.Dir(blog), "human", "doc-1.json"))

	if got := countTraining(d); got != 4 {
		t.Fatalf("countTraining = %d, want 4", got)
	}
}

func TestCorpusPathHelpersIgnoreParentAINames(t *testing.T) {
	populate := func(root string) Data {
		d := Data{Root: root}
		write := func(p string) {
			t.Helper()
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte("sample text\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		blog, _ := GenreByName("blog")
		techblog, _ := GenreByName("techblog")
		pr, _ := GenreByName("pr")

		// 4 trainable files under human/ and ai/<gen>/.
		write(filepath.Join(d.Dir(blog), "human", "doc-1.txt"))
		write(d.OutputPath(blog, "claude-sonnet-5-5", "doc-1", StylePlain))
		write(d.OutputPath(techblog, "gemini-3.8-flash", "doc-2", StylePersona))
		write(filepath.Join(d.Dir(pr), "human", "doc-3.txt"))

		// Excluded eval-only sets, reserve pool, test split, and non-.txt files.
		write(d.OutputPath(blog, "claude-sonnet-5-5", "doc-1", StyleEdit))
		write(filepath.Join(d.Dir(blog), "ai_polished", "claude-sonnet-5-5", "doc-1.txt"))
		write(filepath.Join(d.Dir(blog), "ai_humanized", "claude-sonnet-5-5", "doc-1.txt"))
		write(filepath.Join(d.Dir(blog), "pool", "doc-1.txt"))
		write(filepath.Join(d.Dir(blog), "human", "doc-4.txt"))
		write(d.OutputPath(blog, "claude-sonnet-5-5", "doc-4", StylePlain))
		write(filepath.Join(d.Dir(blog), "human", "doc-1.json"))
		return d
	}

	clean := populate(filepath.Join(t.TempDir(), "clean"))
	nested := populate(filepath.Join(t.TempDir(), "ai", "claude-sonnet-5-5", "human", "checkout"))

	if gotClean, gotNested := countTraining(clean), countTraining(nested); gotClean != 4 || gotNested != 4 {
		t.Errorf("countTraining clean=%d nested(/ai/.../human/)=%d, want both 4", gotClean, gotNested)
	}

	blog, _ := GenreByName("blog")
	humanPath := filepath.Join(nested.Dir(blog), "human", "doc-1.txt")
	claudeAIPath := filepath.Join(nested.Dir(blog), "ai", "claude-sonnet-5-5", "doc-1.txt")
	geminiAIPath := filepath.Join(nested.Dir(blog), "ai", "gemini-3.8-flash", "doc-1.txt")
	poolPath := filepath.Join(nested.Dir(blog), "pool", "doc-1.txt")
	editedPath := filepath.Join(nested.Dir(blog), "ai_edited", "claude-sonnet-5-5", "doc-1~edit.txt")
	polishedPath := filepath.Join(nested.Dir(blog), "ai_polished", "claude-sonnet-5-5", "doc-1.txt")

	for _, tc := range []struct {
		name string
		path string
		want bool
	}{
		{"ai_claude", claudeAIPath, true},
		{"ai_gemini", geminiAIPath, true},
		{"human", humanPath, false},
		{"pool", poolPath, false},
		{"ai_edited", editedPath, false},
		{"ai_polished", polishedPath, false},
	} {
		if got := isAIPath(tc.path); got != tc.want {
			t.Errorf("isAIPath(%s: %q) = %v, want %v", tc.name, tc.path, got, tc.want)
		}
	}

	keepClaude := onlyGens(func(g string) bool { return g == "claude-sonnet-5-5" })
	for _, tc := range []struct {
		name string
		path string
		want bool
	}{
		{"human_kept", humanPath, true},
		{"matching_gen_kept", claudeAIPath, true},
		{"other_gen_dropped", geminiAIPath, false},
		{"pool_dropped", poolPath, false},
		{"ai_edited_dropped", editedPath, false},
		{"ai_polished_dropped", polishedPath, false},
	} {
		if got := keepClaude(tc.path, "doc-1"); got != tc.want {
			t.Errorf("onlyGens(claude)(%s: %q) = %v, want %v", tc.name, tc.path, got, tc.want)
		}
	}
}

// AC3 + AC4: RetrainCandidate with WithLM writes prose.lm.json,
// prose.lm.ngram.json, pr.lm.json, and pr.lm.ngram.json (plus thresholds.json),
// does NOT write prose.json, pr.json, prose.ngram.json, or pr.ngram.json into
// candDir, and engine.FromDir(candDir) loads and scores cleanly both with and
// without AIDETECTOR_SCORER_URL set.
func TestRetrainCandidateWithLMIsolatesLMNgramAndOmitsFallbackCopies(t *testing.T) {
	const modelID = "fake-observer+fake-performer"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model":      modelID,
				"max_tokens": 512,
			})
		case r.Method == http.MethodPost && r.URL.Path == "/score":
			var req struct {
				Texts []string `json:"texts"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
				return
			}
			results := make([]map[string]any, len(req.Texts))
			for i, txt := range req.Texts {
				logPPL, xPPL, bino, curv := 2.80, 3.00, 0.933, 0.35
				if strings.Contains(txt, "comprehensive") || strings.Contains(txt, "orchestration") {
					logPPL, xPPL, bino, curv = 1.45, 2.15, 0.674, 2.80
				}
				results[i] = map[string]any{
					"tokens":     64,
					"log_ppl":    logPPL,
					"x_ppl":      xPPL,
					"binoculars": bino,
					"curvature":  curv,
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model":   modelID,
				"results": results,
			})
		default:
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": "not found"})
		}
	}))
	t.Cleanup(srv.Close)
	t.Setenv("AIDETECTOR_SCORER_URL", srv.URL)

	root := t.TempDir()
	d := Data{Root: root}
	for _, g := range Genres {
		for _, sub := range []string{"human", filepath.Join("ai", "claude-sonnet-5-5")} {
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
		humanProse = "Yesterday morning I carried the wooden ladder out behind the barn, scraped loose paint off the window frame, " +
			"and replaced two cracked panes before noon. The putty knife left linseed oil on my sleeves while sparrows nested under the eaves " +
			"and rain clouds gathered over the ridge. After lunch we stacked split oak by the woodshed, swept the gravel path clean, " +
			"and brewed a fresh pot of strong coffee in the warm kitchen before evening.\n"
		aiProse = "Modern cloud platforms leverage comprehensive observability, deterministic orchestration, and automated governance " +
			"to streamline distributed workloads across hybrid clusters. Furthermore, standardized deployment pipelines strengthen operational " +
			"resilience, eliminate configuration drift, and accelerate sustainable delivery velocity across engineering organizations. " +
			"Teams that systematically align telemetry metrics with architectural guardrails consistently achieve predictable production reliability and scalable governance outcomes.\n"
		humanPR = "Fix a nil pointer dereference in the configuration loader when the file on disk is empty on startup, " +
			"return sensible default values instead of crashing during initialization, and add a unit test covering empty paths and missing directories.\n"
		aiPR = "## Summary\n\nThis pull request introduces comprehensive orchestration safeguards in the configuration loader to ensure " +
			"deterministic startup behavior, consistent validation semantics, and resilient error handling across empty input files and missing directories.\n"
	)

	blog, _ := GenreByName("blog")
	pr, _ := GenreByName("pr")
	for i, id := range trainIDs {
		suffix := fmt.Sprintf(" (%d)\n", i)
		for _, wf := range []struct {
			path string
			body string
		}{
			{filepath.Join(d.Dir(blog), "human", id+".txt"), humanProse + suffix},
			{filepath.Join(d.Dir(blog), "ai", "claude-sonnet-5-5", id+".txt"), aiProse + suffix},
			{filepath.Join(d.Dir(pr), "human", id+".txt"), humanPR + suffix},
			{filepath.Join(d.Dir(pr), "ai", "claude-sonnet-5-5", id+".txt"), aiPR + suffix},
		} {
			if err := os.WriteFile(wf.path, []byte(wf.body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}

	ctx := context.Background()
	candDir := filepath.Join(t.TempDir(), "lm_candidate")
	if err := RetrainCandidate(ctx, d, candDir, FitOptions{
		WithLM:        true,
		KeepCutoffs:   true,
		SignConstrain: true,
	}); err != nil {
		t.Fatalf("RetrainCandidate(WithLM: true): %v", err)
	}

	for _, mustExist := range []string{
		"prose.lm.json",
		"prose.lm.ngram.json",
		"pr.lm.json",
		"pr.lm.ngram.json",
		"thresholds.json",
	} {
		p := filepath.Join(candDir, mustExist)
		if st, err := os.Stat(p); err != nil || st.Size() == 0 {
			t.Errorf("expected non-empty %s in candDir (err=%v)", mustExist, err)
		}
	}
	for _, mustNotExist := range []string{
		"prose.json",
		"pr.json",
		"prose.ngram.json",
		"pr.ngram.json",
	} {
		p := filepath.Join(candDir, mustNotExist)
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("RetrainCandidate(WithLM: true) should not write %s into candDir (err=%v)", mustNotExist, err)
		}
	}

	// 1. With AIDETECTOR_SCORER_URL set, FromDir(candDir) loads the LM models + LM ngram models and scores cleanly.
	eLM, err := engine.FromDir(candDir)
	if err != nil {
		t.Fatalf("engine.FromDir(candDir) with scorer: %v", err)
	}
	resLMProse, err := eLM.Analyze(ctx, detect.Input{Text: aiProse, Kind: detect.KindProse})
	if err != nil {
		t.Fatalf("Analyze(prose) with scorer: %v", err)
	}
	resLMPR, err := eLM.Analyze(ctx, detect.Input{Text: aiPR, Kind: detect.KindPR})
	if err != nil {
		t.Fatalf("Analyze(pr) with scorer: %v", err)
	}
	if resLMProse.Scorer != modelID || resLMPR.Scorer != modelID {
		t.Errorf("Scorer = (%q, %q), want %q for both kinds", resLMProse.Scorer, resLMPR.Scorer, modelID)
	}
	if resLMProse.Label == engine.TooShort || resLMPR.Label == engine.TooShort {
		t.Errorf("unexpected TooShort labels: prose=%q pr=%q", resLMProse.Label, resLMPR.Label)
	}

	// 2. With AIDETECTOR_SCORER_URL unset, FromDir(candDir) overlays onto the embedded fallback models cleanly.
	t.Setenv("AIDETECTOR_SCORER_URL", "")
	eFallback, err := engine.FromDir(candDir)
	if err != nil {
		t.Fatalf("engine.FromDir(candDir) without scorer: %v", err)
	}
	def := engine.Default()
	if eFallback.BuildInfo().Models != def.BuildInfo().Models {
		t.Errorf("without scorer, FromDir(candDir) Models = %q, want embedded Default() %q",
			eFallback.BuildInfo().Models, def.BuildInfo().Models)
	}
	for _, tc := range []struct {
		kind detect.Kind
		text string
	}{
		{detect.KindProse, humanProse},
		{detect.KindPR, humanPR},
	} {
		got, err := eFallback.Analyze(ctx, detect.Input{Text: tc.text, Kind: tc.kind})
		if err != nil {
			t.Fatalf("fallback Analyze(%s): %v", tc.kind, err)
		}
		want, err := def.Analyze(ctx, detect.Input{Text: tc.text, Kind: tc.kind})
		if err != nil {
			t.Fatalf("default Analyze(%s): %v", tc.kind, err)
		}
		if got.Scorer != "" || math.Abs(got.Raw-want.Raw) > 1e-12 || got.Score != want.Score {
			t.Errorf("fallback Analyze(%s) = (Scorer=%q Raw=%.6f Score=%d), want Default (Raw=%.6f Score=%d)",
				tc.kind, got.Scorer, got.Raw, got.Score, want.Raw, want.Score)
		}
	}
}
