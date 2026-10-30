package overnight

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/corpus"
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
	if p := d.OutputPath(g, "m", "x", StyleEdit); !strings.Contains(p, "/ai_edited/") {
		t.Errorf("edit output %s should be outside ai/", p)
	}
	if p := d.OutputPath(g, "m", "x", StylePlain); !strings.Contains(p, "/ai/") {
		t.Errorf("plain output %s should be in ai/", p)
	}
	p := &Planner{Sources: map[string]map[int][]Source{"blog": {0: {{}}}}}
	for _, c := range p.cellsFor("m") {
		if c.Style == StyleEdit {
			t.Fatal("planner offered an edit cell")
		}
	}
}
