package overnight

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/eval"
	"github.com/joeshirey/AIDetector/internal/split"
	"github.com/joeshirey/AIDetector/internal/train"
)

// These measurements are report-only. Nothing here changes labels or
// cutoffs.

// genrePools are the fresh pools that calibrate a per-genre prose cutoff.
var genrePools = map[string]string{"blog": "fresh-blog", "memo": "fresh-memo", "essay": "fresh-essay", "techblog": "tech-pool"}

// GenreCutoff compares the shared prose cutoffs with cutoffs calibrated on
// one genre's pool (same rates, same calibration/check halves).
type GenreCutoff struct {
	Genre        string
	Shared, Own  engine.Thresholds
	FPRShared    [2]float64 // check half, at likely-AI and AI
	FPROwn       [2]float64
	RecallShared float64 // test-split AI of this genre at likely-AI
	RecallOwn    float64
	Human, AI    int
}

// GenreRouting is what happens when the genre has to be detected first.
type GenreRouting struct {
	Accuracy  float64
	Confusion map[string]map[string]int // true -> predicted
	Fallback  float64                   // share routed to the strictest cutoff (classifier unsure)
	// Overall human false-positive rate at likely-AI and AI recall at
	// likely-AI under: shared cutoffs, true-genre cutoffs, and detected-genre
	// cutoffs. The cost of detection errors is detected minus true-genre.
	FPR, Recall map[string]float64
}

type GenreReport struct {
	Cutoffs []GenreCutoff
	Routing GenreRouting
}

type labeled struct {
	genre string
	text  string
	raw   int
	human bool
	calib bool
}

// MeasureGenres calibrates per-genre cutoffs and measures a Naive Bayes
// genre detector's accuracy and the cost of its mistakes.
func MeasureGenres(ctx context.Context, e *engine.Engine, d Data) (*GenreReport, error) {
	var shared engine.Thresholds
	if b, err := engine.EmbeddedFile("thresholds.json"); err == nil {
		var f map[string]engine.Thresholds
		if json.Unmarshal(b, &f) == nil {
			shared = f["prose"]
		}
	}
	pools := d.FreshPools()
	o := train.DefaultThresholdOptions()
	rep := &GenreReport{}
	var items []labeled
	own := map[string]engine.Thresholds{}
	names := sortedKeys(genrePools)
	for _, name := range names {
		g, _ := GenreByName(name)
		hs, err := train.ScoreFiles(ctx, e, []string{pools[genrePools[name]]}, detect.KindProse,
			func(path, id string) bool { return !FinalHoldout(id) })
		if err != nil {
			return nil, err
		}
		var calib, check []int
		for _, s := range hs {
			isCalib := split.Hash(s.ID)%2 == 0
			if isCalib {
				calib = append(calib, s.Raw)
			} else {
				check = append(check, s.Raw)
			}
			items = append(items, labeled{genre: name, text: readText(s.Path), raw: s.Raw, human: true, calib: isCalib})
		}
		t := train.FitCutoffs(calib, check, o.Mixed, o.Likely, o.AI)
		own[name] = t
		ai, err := train.ScoreFiles(ctx, e, []string{filepath.Join(d.Dir(g), "ai")}, detect.KindProse,
			func(path, id string) bool { return split.Of(id) == "test" })
		if err != nil {
			return nil, err
		}
		var aiRaw []int
		for _, s := range ai {
			aiRaw = append(aiRaw, s.Raw)
			items = append(items, labeled{genre: name, text: readText(s.Path), raw: s.Raw})
		}
		rep.Cutoffs = append(rep.Cutoffs, GenreCutoff{Genre: name, Shared: shared, Own: t,
			FPRShared:    [2]float64{train.Share(check, shared.LikelyAI), train.Share(check, shared.AI)},
			FPROwn:       [2]float64{train.Share(check, t.LikelyAI), train.Share(check, t.AI)},
			RecallShared: train.Share(aiRaw, shared.LikelyAI), RecallOwn: train.Share(aiRaw, t.LikelyAI),
			Human: len(hs), AI: len(aiRaw)})
	}

	// Genre detector: trained on train-split human and AI texts of each genre.
	nb := newNaiveBayes()
	for _, name := range names {
		g, _ := GenreByName(name)
		for _, sub := range []string{"human", "ai"} {
			filepath.WalkDir(filepath.Join(d.Dir(g), sub), func(p string, de os.DirEntry, err error) error {
				if err == nil && !de.IsDir() && strings.HasSuffix(p, ".txt") && Trainable(strings.TrimSuffix(filepath.Base(p), ".txt")) {
					nb.add(name, readText(p))
				}
				return nil
			})
		}
	}
	strictest := 0
	for _, t := range own {
		strictest = max(strictest, t.LikelyAI)
	}
	rt := GenreRouting{Confusion: map[string]map[string]int{}, FPR: map[string]float64{}, Recall: map[string]float64{}}
	var correct, fallback, n int
	hit := map[string][2]int{} // scheme -> {flagged, total} for humans
	rec := map[string][2]int{}
	for _, it := range items {
		if it.human && it.calib {
			continue // judge on the check half only
		}
		pred, p := nb.classify(it.text)
		n++
		if rt.Confusion[it.genre] == nil {
			rt.Confusion[it.genre] = map[string]int{}
		}
		rt.Confusion[it.genre][pred]++
		if pred == it.genre {
			correct++
		}
		routed := own[pred].LikelyAI
		if p < 0.9 {
			routed = strictest
			fallback++
		}
		for scheme, cut := range map[string]int{"shared": shared.LikelyAI, "true genre": own[it.genre].LikelyAI, "detected genre": routed} {
			flag := 0
			if it.raw >= cut {
				flag = 1
			}
			if it.human {
				v := hit[scheme]
				hit[scheme] = [2]int{v[0] + flag, v[1] + 1}
			} else {
				v := rec[scheme]
				rec[scheme] = [2]int{v[0] + flag, v[1] + 1}
			}
		}
	}
	if n > 0 {
		rt.Accuracy = float64(correct) / float64(n)
		rt.Fallback = float64(fallback) / float64(n)
	}
	for s, v := range hit {
		rt.FPR[s] = float64(v[0]) / float64(max(v[1], 1))
	}
	for s, v := range rec {
		rt.Recall[s] = float64(v[0]) / float64(max(v[1], 1))
	}
	rep.Routing = rt
	return rep, nil
}

func readText(path string) string {
	b, _ := os.ReadFile(path)
	return string(b)
}

var wordRE = regexp.MustCompile(`[a-z']+`)

// naiveBayes is a multinomial Naive Bayes classifier over lowercase words.
type naiveBayes struct {
	counts map[string]map[string]int
	totals map[string]int
	docs   map[string]int
	vocab  map[string]bool
}

func newNaiveBayes() *naiveBayes {
	return &naiveBayes{counts: map[string]map[string]int{}, totals: map[string]int{}, docs: map[string]int{}, vocab: map[string]bool{}}
}

func (nb *naiveBayes) add(class, text string) {
	if nb.counts[class] == nil {
		nb.counts[class] = map[string]int{}
	}
	nb.docs[class]++
	for _, w := range wordRE.FindAllString(strings.ToLower(text), -1) {
		nb.counts[class][w]++
		nb.totals[class]++
		nb.vocab[w] = true
	}
}

// classify returns the most likely class and its posterior probability.
func (nb *naiveBayes) classify(text string) (string, float64) {
	words := wordRE.FindAllString(strings.ToLower(text), -1)
	totalDocs := 0
	for _, n := range nb.docs {
		totalDocs += n
	}
	classes := sortedKeys(nb.docs)
	logp := make([]float64, len(classes))
	v := float64(len(nb.vocab))
	for i, c := range classes {
		lp := math.Log(float64(nb.docs[c]) / float64(totalDocs))
		denom := float64(nb.totals[c]) + v
		for _, w := range words {
			lp += math.Log((float64(nb.counts[c][w]) + 1) / denom)
		}
		logp[i] = lp
	}
	best := 0
	for i := range logp {
		if logp[i] > logp[best] {
			best = i
		}
	}
	sum := 0.0
	for i := range logp {
		sum += math.Exp(logp[i] - logp[best])
	}
	if len(classes) == 0 {
		return "", 0
	}
	return classes[best], 1 / sum
}

// LabelDist counts labels on the eval-only polished and humanized sets.
type LabelDist struct {
	Set    string // e.g. "blog/ai_polished"
	N      int
	Labels map[string]int
}

func MeasurePolished(ctx context.Context, e *engine.Engine, d Data) ([]LabelDist, error) {
	var out []LabelDist
	for _, g := range Genres {
		if g.Store != "corpus" {
			continue
		}
		for _, set := range []string{"ai_polished", "ai_humanized"} {
			dir := filepath.Join(d.Dir(g), set)
			ld := LabelDist{Set: g.Name + "/" + set, Labels: map[string]int{}}
			err := filepath.WalkDir(dir, func(p string, de os.DirEntry, err error) error {
				if err != nil {
					if os.IsNotExist(err) {
						return filepath.SkipDir
					}
					return err
				}
				if de.IsDir() || !strings.HasSuffix(p, ".txt") {
					return nil
				}
				r, err := e.Analyze(ctx, detect.Input{Text: readText(p), Kind: g.Det})
				if err != nil {
					return err
				}
				ld.N++
				ld.Labels[r.Label]++
				return nil
			})
			if err != nil {
				return nil, err
			}
			if ld.N > 0 {
				out = append(out, ld)
			}
		}
	}
	return out, nil
}

// LOFORow is one leave-one-family-out result: a model trained without a
// family's AI text, tested on that family's test-split text.
type LOFORow struct {
	Family     string
	Kind       string
	Held, Full AR // held-out model vs. production on the same samples
	Err        string
}

var LOFOFamilies = []string{"claude", "gemini", "qwen", "gpt", "gemma"}

// LeaveOneFamilyOut refits each kind without one generator family at a time
// (no LM features, so it runs on CPU) under dir, and scores that family's
// test-split text against the same humans.
func LeaveOneFamilyOut(ctx context.Context, d Data, prod *engine.Engine, dir string) ([]LOFORow, error) {
	var out []LOFORow
	for _, k := range []detect.Kind{detect.KindProse, detect.KindPR} {
		full, err := eval.RunFiltered(ctx, prod, d.TrainDirs(k), k, "test", nil)
		if err != nil {
			return nil, err
		}
		for _, fam := range LOFOFamilies {
			ofFamily := onlyGens(func(g string) bool { return GeneratorFamily(g) == fam })
			fullSub := full.Subset(ofFamily)
			row := LOFORow{Family: fam, Kind: string(k), Full: reportAR(fullSub)}
			if fullSub.AI == 0 {
				row.Err = "no test-split samples"
				out = append(out, row)
				continue
			}
			md := filepath.Join(dir, fam)
			if err := os.MkdirAll(md, 0o755); err != nil {
				return nil, err
			}
			if b, err := engine.EmbeddedFile("thresholds.json"); err == nil {
				os.WriteFile(filepath.Join(md, "thresholds.json"), b, 0o644)
			}
			keep := func(path, id string) bool { return Trainable(id) && !(isAIPath(path) && ofFamily(path, id)) }
			if _, err := train.Fit(ctx, k, d.TrainDirs(k), md, false, keep); err != nil {
				row.Err = err.Error()
				out = append(out, row)
				continue
			}
			held, err := engine.FromDir(md)
			if err != nil {
				return nil, err
			}
			r, err := eval.RunFiltered(ctx, held, d.TrainDirs(k), k, "test", ofFamily)
			if err != nil {
				row.Err = err.Error()
			} else {
				row.Held = reportAR(r)
			}
			out = append(out, row)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Kind < out[j].Kind })
	return out, nil
}

func isAIPath(path string) bool {
	return strings.Contains(path, string(filepath.Separator)+"ai"+string(filepath.Separator))
}

// Flips lists, per fresh pool, the human texts (final holdout excluded)
// that score 60+ under one engine but not the other, with both scores and
// raw scores, to show whether a false-positive rise is a few texts tipping
// over a cutoff or broad drift.
func Flips(ctx context.Context, w io.Writer, d Data, prod, cand *engine.Engine, pools []string) error {
	all := d.FreshPools()
	for _, name := range pools {
		dir, ok := all[strings.TrimSpace(name)]
		if !ok {
			return fmt.Errorf("unknown pool %q", name)
		}
		k := detect.KindProse
		if name == "fresh-pr" {
			k = detect.KindPR
		}
		keep := func(path, id string) bool { return !FinalHoldout(id) }
		a, err := train.ScoreFiles(ctx, prod, []string{dir}, k, keep)
		if err != nil {
			return err
		}
		b, err := train.ScoreFiles(ctx, cand, []string{dir}, k, keep)
		if err != nil {
			return err
		}
		byID := map[string]train.Scored{}
		for _, s := range a {
			byID[s.ID] = s
		}
		var up, down []string
		meanA, meanB := 0.0, 0.0
		for _, s := range b {
			p := byID[s.ID]
			meanA += float64(p.Raw)
			meanB += float64(s.Raw)
			line := fmt.Sprintf("  %-50s prod %3d (raw %3d)  cand %3d (raw %3d)", s.ID, p.Score, p.Raw, s.Score, s.Raw)
			switch {
			case s.Score >= 60 && p.Score < 60:
				up = append(up, line)
			case p.Score >= 60 && s.Score < 60:
				down = append(down, line)
			}
		}
		n := float64(max(len(b), 1))
		fmt.Fprintf(w, "%s: %d texts, mean raw prod %.1f cand %.1f; %d newly 60+, %d no longer 60+\n", name, len(b), meanA/n, meanB/n, len(up), len(down))
		for _, l := range up {
			fmt.Fprintln(w, "  +"+l[1:])
		}
		for _, l := range down {
			fmt.Fprintln(w, "  -"+l[1:])
		}
	}
	return nil
}
