package overnight

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Check is one gate comparison.
type Check struct {
	Name         string  `json:"name"`
	Before       float64 `json:"before"`
	After        float64 `json:"after"`
	Tolerance    float64 `json:"tolerance"`
	HigherBetter bool    `json:"higher_better"`
	// P is the bootstrap p-value for "the candidate is worse than the
	// baseline by more than Tolerance"; Pass is false when that holds
	// after the Holm correction across all checks.
	P    float64 `json:"p"`
	Pass bool    `json:"pass"`
}

// The family-wise error budget, 5% in all, is split between the human
// false-positive checks and the detection checks (AUROC and recall), with
// most of it on false positives because those matter most here. A
// candidate no worse than the baseline fails by chance at most ~5% of the
// time.
const (
	AlphaFalsePositives = 0.04
	AlphaDetection      = 0.01
)

// Gate compares a candidate with the baseline. Each metric's change is
// tested with a paired bootstrap (the same resampled documents, or source
// groups, scored by both models), and the Holm correction across all
// checks keeps the false-rejection rate of a neutral candidate near
// GateAlpha (Joe, 2026-10-29; replaces the fixed one-text tolerance, which
// rejected neutral refits most of the time). The fixtures are a hard check,
// and a metric the baseline has but the candidate lacks fails.
func Gate(prod, cand *Metrics) ([]Check, bool) {
	rng := rand.New(rand.NewPCG(20261029, 1))
	type test struct {
		c    Check
		boot func(*rand.Rand) float64 // resampled worsening (positive = candidate worse)
		hard bool                     // fails outright (missing data)
		fp   bool                     // a human false-positive check (its own Holm family)
	}
	var tests []test
	missing := func(name string) {
		tests = append(tests, test{c: Check{Name: name + " (missing from candidate)", P: 0}, hard: true})
	}

	// Ranking metrics: AUROC and recall at 1% false positives.
	rank := func(prefix, key string, a, b AR, aucTol, recTol float64) {
		pa, pb, groups := pairSamples(a.S, b.S)
		if len(pa) == 0 {
			// No per-sample data: fall back to point comparison, no test.
			tests = append(tests,
				test{c: Check{Name: prefix + " AUROC " + key, Before: a.AUROC, After: b.AUROC, Tolerance: aucTol, HigherBetter: true, P: pointP(a.AUROC-b.AUROC, aucTol)}},
				test{c: Check{Name: prefix + " recall@1%FP " + key, Before: a.Recall1, After: b.Recall1, Tolerance: recTol, HigherBetter: true, P: pointP(a.Recall1-b.Recall1, recTol)}})
			return
		}
		mk := func(name string, before, after, tol float64, which int) test {
			return test{c: Check{Name: name, Before: before, After: after, Tolerance: tol, HigherBetter: true},
				boot: func(r *rand.Rand) float64 {
					idx := resampleGroups(r, groups)
					a1, r1 := aurocRecall(pa, idx)
					a2, r2 := aurocRecall(pb, idx)
					if which == 0 {
						return a1 - a2
					}
					return r1 - r2
				}}
		}
		// Observed values on the paired samples (the same ones resampled).
		all := make([]int, len(pa))
		for i := range all {
			all[i] = i
		}
		aucA, recA := aurocRecall(pa, all)
		aucB, recB := aurocRecall(pb, all)
		tests = append(tests, mk(prefix+" AUROC "+key, aucA, aucB, aucTol, 0))
		if recTol > 0 {
			tests = append(tests, mk(prefix+" recall@1%FP "+key, recA, recB, recTol, 1))
		}
	}
	for _, k := range sortedKeys(prod.Test) {
		c, ok := cand.Test[k]
		if !ok {
			missing("test " + k)
			continue
		}
		rank("test", k, prod.Test[k], c, 0.002, 0.01)
	}
	for _, k := range sortedKeys(prod.Frozen) {
		c, ok := cand.Frozen[k]
		if !ok {
			missing("frozen " + k)
			continue
		}
		rank("frozen", k, prod.Frozen[k], c, 0.01, 0.01)
	}
	for _, k := range sortedKeys(prod.Slices) {
		c, ok := cand.Slices[k]
		if !ok {
			missing("slice " + k)
			continue
		}
		rank("slice", k, prod.Slices[k], c, 0.005, 0)
	}

	// Human false-positive rates on the checking half, resampled by document
	// (a document's segments move together).
	for _, p := range sortedKeys(prod.PoolDocs) {
		cd, ok := cand.PoolDocs[p]
		if !ok {
			missing("pool " + p)
			continue
		}
		pairs := pairDocs(prod.PoolDocs[p], cd)
		if len(pairs) == 0 {
			continue
		}
		rate := func(idx []int, side int, fn func(PoolDoc) (hit, n int)) float64 {
			h, n := 0, 0
			for _, i := range idx {
				a, b := fn(pairs[i][side])
				h, n = h+a, n+b
			}
			if n == 0 {
				return 0
			}
			return float64(h) / float64(n)
		}
		docAt := func(t int) func(PoolDoc) (int, int) {
			return func(d PoolDoc) (int, int) {
				if d.Score >= t {
					return 1, 1
				}
				return 0, 1
			}
		}
		seg60 := func(d PoolDoc) (int, int) {
			h := 0
			for _, s := range d.Segs {
				if s >= 60 {
					h++
				}
			}
			return h, len(d.Segs)
		}
		all := make([]int, len(pairs))
		for i := range all {
			all[i] = i
		}
		add := func(name string, fn func(PoolDoc) (int, int)) {
			units := 0
			for _, pr := range pairs {
				_, k := fn(pr[0])
				units += k
			}
			if units == 0 {
				return // e.g. a pool with no multi-segment documents
			}
			before, after := rate(all, 0, fn), rate(all, 1, fn)
			tests = append(tests, test{c: Check{Name: name, Before: before, After: after}, fp: true,
				boot: func(r *rand.Rand) float64 {
					idx := make([]int, len(pairs))
					for i := range idx {
						idx[i] = r.IntN(len(pairs))
					}
					return rate(idx, 1, fn) - rate(idx, 0, fn)
				}})
		}
		add("doc FPR 60+ "+p, docAt(60))
		add("doc FPR 80+ "+p, docAt(80))
		add("segment FPR 60+ "+p, seg60)
	}

	// p-values: the bootstrap gives the spread of the paired difference;
	// the p-value for "worse by more than the tolerance" is the normal tail
	// beyond the observed difference. (Counting resamples directly can't
	// produce p-values below 1/boots, which with ~60 checks is above the
	// Holm threshold, so nothing could ever fail.)
	const boots = 400
	for i := range tests {
		t := &tests[i]
		if t.hard || t.boot == nil {
			continue
		}
		var sum, sq float64
		for range boots {
			v := t.boot(rng)
			sum += v
			sq += v * v
		}
		mean := sum / boots
		sd := math.Sqrt(max(sq/boots-mean*mean, 0))
		worse := t.c.Before - t.c.After // positive = candidate worse, for higher-is-better metrics
		if !t.c.HigherBetter {
			worse = t.c.After - t.c.Before
		}
		switch {
		case sd == 0 && worse > t.c.Tolerance:
			t.c.P = 0
		case sd == 0:
			t.c.P = 1
		default:
			z := (worse - t.c.Tolerance) / sd
			t.c.P = 0.5 * math.Erfc(z/math.Sqrt2)
		}
	}

	// Holm step-down within each family.
	holm := func(fp bool, alpha float64) {
		var order []int
		for i := range tests {
			if tests[i].fp == fp {
				order = append(order, i)
			}
		}
		sort.SliceStable(order, func(a, b int) bool { return tests[order[a]].c.P < tests[order[b]].c.P })
		for rankIdx, i := range order {
			if tests[i].c.P <= alpha/float64(len(order)-rankIdx) {
				tests[i].c.Pass = false
				continue
			}
			break // Holm stops at the first non-rejection
		}
	}
	for i := range tests {
		tests[i].c.Pass = !tests[i].hard
	}
	holm(true, AlphaFalsePositives)
	holm(false, AlphaDetection)
	var checks []Check
	ok := true
	for _, t := range tests {
		checks = append(checks, t.c)
		ok = ok && t.c.Pass
	}
	fx := cand.Fixtures.Pass() && cand.Fixtures.Missing == 0
	fval := 0.0
	if fx {
		fval = 1
	}
	checks = append(checks, Check{Name: "regression fixtures pass (none missing)", Before: 1, After: fval, HigherBetter: true, Pass: fx})
	return checks, ok && fx
}

// pointP is used when per-sample data is missing: 0 if worse than allowed.
func pointP(worse, tol float64) float64 {
	if worse > tol {
		return 0
	}
	return 1
}

type rankPair struct {
	raw float64
	ai  bool
}

// pairSamples matches baseline and candidate samples by file and returns
// each side's scores in the same order, plus the index lists of each
// source group.
func pairSamples(a, b []RankSample) (pa, pb []rankPair, groups [][]int) {
	byKey := map[string]RankSample{}
	for _, s := range b {
		byKey[s.Key] = s
	}
	gi := map[string]int{}
	for _, s := range a {
		c, ok := byKey[s.Key]
		if !ok {
			continue
		}
		g, ok := gi[s.Group]
		if !ok {
			g = len(groups)
			gi[s.Group] = g
			groups = append(groups, nil)
		}
		groups[g] = append(groups[g], len(pa))
		pa = append(pa, rankPair{s.Raw, s.AI})
		pb = append(pb, rankPair{c.Raw, c.AI})
	}
	return pa, pb, groups
}

func resampleGroups(r *rand.Rand, groups [][]int) []int {
	var idx []int
	for range groups {
		idx = append(idx, groups[r.IntN(len(groups))]...)
	}
	return idx
}

// aurocRecall computes AUROC (Mann-Whitney with tie correction) and
// recall at <=1% false positives over the selected samples.
func aurocRecall(s []rankPair, idx []int) (float64, float64) {
	type v struct {
		raw float64
		ai  bool
	}
	xs := make([]v, len(idx))
	var nh, na int
	for k, i := range idx {
		xs[k] = v{s[i].raw, s[i].ai}
		if s[i].ai {
			na++
		} else {
			nh++
		}
	}
	if nh == 0 || na == 0 {
		return 0, 0
	}
	sort.Slice(xs, func(i, j int) bool { return xs[i].raw < xs[j].raw })
	// Rank sum of AI samples with average ranks for ties.
	var rankSum float64
	for i := 0; i < len(xs); {
		j := i
		for j < len(xs) && xs[j].raw == xs[i].raw {
			j++
		}
		avg := float64(i+j+1) / 2
		for k := i; k < j; k++ {
			if xs[k].ai {
				rankSum += avg
			}
		}
		i = j
	}
	auc := (rankSum - float64(na)*float64(na+1)/2) / (float64(na) * float64(nh))
	// Recall: walk down from the top until more than 1% of humans are above.
	allowed := nh / 100
	humansAbove, aiAbove := 0, 0
	recall := 0.0
	for i := len(xs) - 1; i >= 0; i-- {
		if !xs[i].ai {
			humansAbove++
			if humansAbove > allowed {
				break
			}
		} else {
			aiAbove++
		}
		recall = float64(aiAbove) / float64(na)
	}
	return auc, recall
}

func pairDocs(a, b []PoolDoc) [][2]PoolDoc {
	byID := map[string]PoolDoc{}
	for _, d := range b {
		byID[d.ID] = d
	}
	var out [][2]PoolDoc
	for _, d := range a {
		if c, ok := byID[d.ID]; ok {
			out = append(out, [2]PoolDoc{d, c})
		}
	}
	return out
}
