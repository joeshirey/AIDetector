package overnight

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/fixtures"
)

// Paid generators and their base shares of the paid sample budget. Argon
// trains only from week 2, and only once Joe approves (ArgonTraining).
var PaidBase = map[string]float64{
	"claude:claude-haiku-5-5":  0.45,
	"gemini:gemini-3.8-flash":  0.25,
	"claude:claude-sonnet-5-5": 0.15,
	"gemini:gemini-4-argon":    0.15,
}

const (
	PaidFloor   = 0.05 // minimum share per paid generator
	BanditFloor = 0.20 // uniform exploration share
	MaxAccepts  = 3
)

// Config is one night's settings.
type Config struct {
	Data        Data
	RunRoot     string // runs/overnight (state, bandit, jobs, history, candidates)
	Ledger      string
	ReportDir   string // reports/overnight
	Bucket      string
	FixturesDir string
	FixtureList string

	Until         time.Time          // hard stop; retrain, gate, and report must finish by then
	BudgetUSD     float64            // hard cap for the night
	ModelCaps     map[string]float64 // per-model caps (by model id)
	MaxNew        int                // max paid requests per generator per night
	ArgonTraining bool               // Joe approved Argon for training (week 2+)
	Efforts       map[string]string  // model -> "low" where the effort experiment allows it

	LocalModels  []string // lmstudio specs, e.g. lmstudio:qwen3.6-35b-a3b
	LocalWorkers int
	LocalQuota   int // max local samples per model per night

	MineN     int // human false positives promoted per kind per night
	SourceCap int // per source per night
	Rounds    int // retrain attempts (at most MaxAccepts accepted)
	Analyst   string
	LOFO      bool // run leave-one-family-out (Sundays)
	DryRun    bool // tiny night: few requests, no LOFO

	Log io.Writer
}

// State is resumable progress for one night.
type State struct {
	Night    string            `json:"night"`
	Done     map[string]bool   `json:"done"`
	Limit    float64           `json:"limit"`
	Notes    []string          `json:"notes"`
	Mined    map[string]int    `json:"mined"`  // kind -> promoted
	Misses   map[string][2]int `json:"misses"` // generator -> {missed, scored}
	Batches  BatchSummary      `json:"batches"`
	Local    []LocalStats      `json:"local"`
	Rounds   []Round           `json:"rounds"`
	Accepted int               `json:"accepted"`
}

type BatchSummary struct {
	Collected, Carried int
	Submitted          []string // gen: n requests, $est
	Shares             map[string]float64
}

type Round struct {
	N        int
	Dir      string
	Samples  int // training files when it was fit
	Checks   []Check
	Accepted bool
	Err      string
}

type Loop struct {
	Cfg     Config
	Night   string
	dir     string
	state   *State
	bandit  *Bandit
	prod    *engine.Engine
	prodM   *Metrics
	fixList *fixtures.List
	rng     *rand.Rand
}

func (l *Loop) logf(format string, a ...any) {
	fmt.Fprintf(l.Cfg.Log, time.Now().Format("15:04:05 ")+format+"\n", a...)
}

func (l *Loop) note(format string, a ...any) {
	s := fmt.Sprintf(format, a...)
	l.state.Notes = append(l.state.Notes, s)
	l.logf("%s", s)
}

func (l *Loop) save() error {
	b, _ := json.MarshalIndent(l.state, "", " ")
	return os.WriteFile(filepath.Join(l.dir, "state.json"), b, 0o644)
}

// step runs fn once per night; a finished step is skipped on restart.
func (l *Loop) step(ctx context.Context, name string, fn func(context.Context) error) error {
	if l.state.Done[name] {
		return nil
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	l.logf("== %s", name)
	if err := fn(ctx); err != nil {
		if errors.Is(err, corpus.ErrBudget) {
			l.note("%s stopped at the budget: %v", name, err)
		} else {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	l.state.Done[name] = true
	return l.save()
}

// Run executes the night.
func Run(ctx context.Context, cfg Config) (reportPath string, err error) {
	night := corpus.NightOf(time.Now())
	l := &Loop{Cfg: cfg, Night: night.Format("2006-01-02"), rng: rand.New(rand.NewPCG(stableSeed(night.Format("2006-01-02")), 1))}
	l.dir = filepath.Join(cfg.RunRoot, l.Night)
	if cfg.DryRun {
		l.dir += "-dry-run" // never shares state with the night's full run
	}
	if err := os.MkdirAll(l.dir, 0o755); err != nil {
		return "", err
	}
	l.state = &State{Night: l.Night, Done: map[string]bool{}, Limit: cfg.BudgetUSD, Mined: map[string]int{}, Misses: map[string][2]int{}}
	if b, err := os.ReadFile(filepath.Join(l.dir, "state.json")); err == nil {
		json.Unmarshal(b, l.state)
	}
	saved := l.state.Limit
	// The cap only ever goes down (billing reconciliation can lower it).
	if l.state.Limit == 0 || cfg.BudgetUSD < l.state.Limit {
		l.state.Limit = cfg.BudgetUSD
	}
	corpus.Spend = corpus.NewBudget(l.state.Limit)
	corpus.Spend.ModelCaps = cfg.ModelCaps
	corpus.Spend.Purpose = "overnight"
	if err := corpus.Spend.OpenLedger(cfg.Ledger); err != nil {
		return "", err
	}
	if cfg.DryRun && !l.state.Done["verify frozen sets"] {
		// A dry run's cap is on top of whatever pre-work already spent tonight.
		l.state.Limit = corpus.Spend.Spent + cfg.BudgetUSD
		corpus.Spend.Limit = l.state.Limit
		l.note("dry run: $%.2f cap on top of $%.2f already spent tonight", cfg.BudgetUSD, corpus.Spend.Spent)
	} else if cfg.DryRun {
		l.state.Limit = saved
		corpus.Spend.Limit = saved
	}
	l.bandit = LoadBandit(filepath.Join(cfg.RunRoot, "bandit.json"), stableSeed(l.Night))
	l.prod = engine.Default()
	if cfg.FixtureList != "" {
		if l.fixList, err = fixtures.Load(cfg.FixtureList); err != nil {
			return "", err
		}
	}
	rep := &Report{Night: l.Night, Started: time.Now(), Cfg: cfg}

	// Report whatever happened, even if a step failed.
	defer func() {
		rep.State = l.state
		rep.Finished = time.Now()
		if err != nil {
			rep.Error = err.Error()
		}
		p, werr := l.writeReport(context.WithoutCancel(ctx), rep)
		if werr != nil && err == nil {
			err = werr
		}
		reportPath = p
	}()

	steps := []struct {
		name string
		fn   func(context.Context) error
	}{
		{"verify frozen sets", func(ctx context.Context) error { return l.verifyFrozen(rep) }},
		{"reconcile billing", func(ctx context.Context) error { return l.reconcile(ctx, rep) }},
		{"collect batch jobs", l.collect},
		{"baseline metrics", func(ctx context.Context) error { return l.baseline(ctx, rep) }},
		{"mine human false positives", l.mineHuman},
		{"score new AI text", l.scorePending},
		{"submit paid batches", l.submitPaid},
	}
	for _, s := range steps {
		if err := l.step(ctx, s.name, s.fn); err != nil {
			return "", err
		}
	}
	if rep.FrozenBroken {
		l.note("a sealed frozen set changed; skipping retraining tonight")
	} else if err := l.generateAndRetrain(ctx, rep); err != nil {
		return "", err
	}
	if err := l.measure(ctx, rep); err != nil {
		l.note("measurements failed: %v", err)
	}
	return "", nil
}

func (l *Loop) verifyFrozen(rep *Report) error {
	m, err := l.Cfg.Data.LoadManifest()
	if err != nil {
		return fmt.Errorf("frozen manifest: %w (run `corpus overnight init` first)", err)
	}
	rep.Manifest = m
	if rep.Frozen, err = l.Cfg.Data.Verify(m, l.Cfg.FixturesDir, false); err != nil {
		return err
	}
	for _, v := range rep.Frozen {
		if !v.OK() {
			rep.FrozenBroken = true
		}
	}
	return nil
}

func (l *Loop) reconcile(ctx context.Context, rep *Report) error {
	entries, err := ReadLedger(l.Cfg.Ledger)
	if err != nil {
		return err
	}
	night, _ := time.ParseInLocation("2006-01-02", l.Night, time.Local)
	rep.Billing = Reconcile(ctx, entries, night, l.state.Limit)
	if rep.Billing.Over {
		l.state.Limit = rep.Billing.NewLimit
		corpus.Spend.Limit = rep.Billing.NewLimit
		l.note("FLAG: %s", rep.Billing.Note)
	}
	return nil
}

func (l *Loop) batcher(ctx context.Context) (*corpus.Batcher, *Jobs, error) {
	b, err := corpus.NewBatcher(ctx, l.Cfg.Bucket)
	if err != nil {
		return nil, nil, err
	}
	j, err := LoadJobs(filepath.Join(l.Cfg.RunRoot, "batch-jobs.json"))
	return b, j, err
}

func (l *Loop) pendingPath() string { return filepath.Join(l.Cfg.RunRoot, "pending.jsonl") }

// recordWrite queues a generated file for scoring (bandit feedback).
func (l *Loop) recordWrite(r GenRecord) { QueueForScoring(l.Cfg.RunRoot)(r) }

// QueueForScoring returns a Collector.OnWrite that queues generated files
// in runRoot/pending.jsonl; the next night scores them for the bandit.
func QueueForScoring(runRoot string) func(GenRecord) {
	return func(r GenRecord) {
		f, err := os.OpenFile(filepath.Join(runRoot, "pending.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return
		}
		defer f.Close()
		b, _ := json.Marshal(r)
		f.Write(append(b, '\n'))
	}
}

func (l *Loop) collect(ctx context.Context) error {
	b, jobs, err := l.batcher(ctx)
	if err != nil {
		return err
	}
	c := &Collector{Data: l.Cfg.Data, Batcher: b, Jobs: jobs, OnWrite: l.recordWrite, Log: l.Cfg.Log}
	l.state.Batches.Collected, l.state.Batches.Carried, err = c.Collect(ctx, l.Night)
	return err
}

// baseline measures production's models with their cutoffs refit the same
// way candidates' are, so the gate compares like with like.
func (l *Loop) baseline(ctx context.Context, rep *Report) error {
	dir := filepath.Join(l.dir, "baseline")
	if err := ProductionBaseline(ctx, l.Cfg.Data, dir); err != nil {
		return err
	}
	base, err := engine.FromDir(dir)
	if err != nil {
		return err
	}
	m, err := Measure(ctx, base, l.Cfg.Data, l.fixList, l.Cfg.FixturesDir)
	if err != nil {
		return err
	}
	l.prodM = m
	rep.Prod = m
	return writeJSONFile(filepath.Join(l.dir, "prod-metrics.json"), m)
}

// mineHuman promotes the human pool texts production scores as most
// AI-like into training (hard negatives). Only the main corpus pools are
// promotable, only train-split ids (the test split is frozen), and each
// source contributes at most SourceCap a night.
func (l *Loop) mineHuman(ctx context.Context) error {
	store := &corpus.Store{Root: l.Cfg.Data.Path("corpus")}
	for _, kind := range []string{"blog", "memo", "essay", "pr"} {
		pool, err := store.Pool(kind)
		if err != nil {
			return err
		}
		det := detect.KindProse
		if kind == "pr" {
			det = detect.KindPR
		}
		type cand struct {
			s     corpus.Sample
			score int
		}
		var cands []cand
		for _, s := range pool {
			if !Trainable(s.ID) {
				continue
			}
			r, err := l.prod.Analyze(ctx, detect.Input{Text: s.Text, Kind: det})
			if err != nil || r.Score < 50 {
				continue
			}
			cands = append(cands, cand{s, r.Score})
		}
		slices.SortFunc(cands, func(a, b cand) int { return cmp.Compare(b.score, a.score) })
		perSource := map[string]int{}
		n := 0
		for _, c := range cands {
			if n >= l.Cfg.MineN {
				break
			}
			if perSource[c.s.Source] >= l.Cfg.SourceCap {
				continue
			}
			if err := corpus.FetchDiff(ctx, &c.s); err != nil {
				continue
			}
			if err := store.Promote(c.s); err != nil {
				return err
			}
			perSource[c.s.Source]++
			n++
		}
		l.state.Mined[kind] = n
		l.logf("%s: %d pool texts at 50+ (train split), promoted %d", kind, len(cands), n)
	}
	return nil
}

// scorePending scores queued generations with production and feeds misses
// (AI text under 60) to the bandit. Frozen-track files don't count.
func (l *Loop) scorePending(ctx context.Context) error {
	f, err := os.Open(l.pendingPath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	var recs []GenRecord
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r GenRecord
		if json.Unmarshal(sc.Bytes(), &r) == nil {
			recs = append(recs, r)
		}
	}
	f.Close()
	for _, r := range recs {
		if r.Track != "train" || r.Cell.Style == StyleEdit {
			continue // edits are eval-only; they don't steer the bandit
		}
		g, ok := GenreByName(r.Cell.Genre)
		if !ok {
			continue
		}
		res, err := l.prod.Analyze(ctx, detect.Input{Text: readText(r.Path), Kind: g.Det})
		if err != nil || res.Label == engine.TooShort {
			continue
		}
		miss := res.Score < 60
		l.bandit.Observe(r.Cell, miss)
		v := l.state.Misses[r.Cell.Gen]
		if miss {
			v[0]++
		}
		v[1]++
		l.state.Misses[r.Cell.Gen] = v
	}
	if err := l.bandit.Save(filepath.Join(l.Cfg.RunRoot, "bandit.json")); err != nil {
		return err
	}
	return os.Remove(l.pendingPath())
}

func (l *Loop) planner() (*Planner, error) {
	p, err := NewPlanner(l.Cfg.Data, l.bandit, Genres)
	if err != nil {
		return nil, err
	}
	p.Floor = BanditFloor
	return p, nil
}

func (l *Loop) briefer(ctx context.Context) (*Briefer, error) {
	gs, err := corpus.NewGenerators(ctx, "claude:claude-haiku-5-5")
	if err != nil {
		return nil, err
	}
	return &Briefer{Gen: gs[0], Dir: l.Cfg.Data.Path("briefs", "train")}, nil
}

// submitPaid plans tonight's paid generation and submits one batch job per
// generator. Results arrive by a later night (pipelined); the estimate is
// reserved against tonight now.
func (l *Loop) submitPaid(ctx context.Context) error {
	base := map[string]float64{}
	for spec, share := range PaidBase {
		if strings.Contains(spec, "argon") && !l.Cfg.ArgonTraining {
			continue // eval-only until Joe approves
		}
		s, _ := corpus.ParseSpec(spec)
		if e := l.Cfg.Efforts[s.Model]; e != "" {
			spec += "@" + e
		}
		base[spec] = share
	}
	// GenShares works on generator names (the corpus directory).
	byName := map[string]string{}
	baseNames := map[string]float64{}
	for spec, share := range base {
		s, _ := corpus.ParseSpec(spec)
		byName[s.Name()] = spec
		baseNames[s.Name()] = share
	}
	shares := l.bandit.GenShares(baseNames, PaidFloor, 2)
	l.state.Batches.Shares = shares
	// Keep back room for briefs and the morning analysis.
	pot := corpus.Spend.Headroom("") - 1.0
	if l.Cfg.DryRun {
		pot = corpus.Spend.Headroom("") * 0.6
	}
	if pot <= 0 {
		return corpus.ErrBudget
	}
	p, err := l.planner()
	if err != nil {
		return err
	}
	br, err := l.briefer(ctx)
	if err != nil {
		return err
	}
	b, jobs, err := l.batcher(ctx)
	if err != nil {
		return err
	}
	// Shares are of samples, not dollars: find the sample total whose cost
	// at each generator's typical per-request price fits the pot.
	names := sortedKeys(shares)
	gens := map[string]corpus.Generator{}
	perReq := map[string]float64{}
	costPerSample := 0.0
	for _, name := range names {
		gs, err := corpus.NewGenerators(ctx, byName[name])
		if err != nil {
			return err
		}
		gens[name] = gs[0]
		s := gs[0].Spec()
		perReq[name] = corpus.Spend.Cost(s.Model, corpus.ModeBatch, EstimateUsage(s, strings.Repeat("word ", 700), 450))
		costPerSample += shares[name] * perReq[name]
	}
	total := pot / max(costPerSample, 1e-9)
	for _, name := range names {
		gen := gens[name]
		n := min(int(total*shares[name]), l.Cfg.MaxNew)
		alloc := min(float64(n)*perReq[name]*1.5, corpus.Spend.Headroom(gen.Spec().Model)) // slack for long prompts
		reqs, est := PlanPaid(ctx, p, br, gen, alloc, n, l.rng)
		t, err := SubmitPaid(ctx, b, jobs, gen.Spec(), l.Night, "train", reqs)
		if err != nil {
			l.note("%s: batch not submitted: %v", name, err)
			continue
		}
		if t != nil {
			l.state.Batches.Submitted = append(l.state.Batches.Submitted, fmt.Sprintf("%s: %d requests, est. $%.2f", name, len(reqs), est))
		}
	}
	return nil
}

// reserve is how long retraining, gating, measuring, and the report need.
func (l *Loop) reserve() time.Duration {
	d := 75 * time.Minute
	if l.Cfg.LOFO {
		d += 60 * time.Minute
	}
	if l.Cfg.DryRun {
		d = 20 * time.Minute
	}
	return d
}

// generateAndRetrain alternates local generation slices with retrain-and-
// gate rounds, until the deadline or MaxAccepts accepted rounds. Data from
// rejected rounds stays in the corpus.
func (l *Loop) generateAndRetrain(ctx context.Context, rep *Report) error {
	rounds := max(l.Cfg.Rounds, 1)
	stopAll := l.Cfg.Until.Add(-l.reserve())
	for i := len(l.state.Rounds); i < rounds && l.state.Accepted < MaxAccepts; i++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Each round gets an equal share of the remaining generation time.
		slice := time.Until(stopAll) / time.Duration(rounds-i)
		if slice > 0 {
			l.runLocal(ctx, time.Now().Add(slice))
			if err := l.scorePending(ctx); err != nil {
				return err
			}
		}
		// Without new training data a refit would just repeat the last round.
		if n := len(l.state.Rounds); n > 0 && countTraining(l.Cfg.Data) == l.state.Rounds[n-1].Samples {
			l.note("no new training data since round %d; skipping the remaining rounds", n)
			break
		}
		r := l.retrain(ctx, i+1)
		l.state.Rounds = append(l.state.Rounds, r)
		if r.Accepted {
			l.state.Accepted++
		}
		if err := l.save(); err != nil {
			return err
		}
		// Retraining takes a while; stop before eating the report reserve.
		if time.Now().After(stopAll) {
			break
		}
	}
	return nil
}

// runLocal generates with the local models one at a time (they share this
// machine's GPU, so running them together slows both), splitting the time
// until stopAt evenly among the models that are available.
func (l *Loop) runLocal(ctx context.Context, stopAt time.Time) {
	gemmaReady := false
	if m, err := l.Cfg.Data.LoadManifest(); err == nil {
		_, gemmaReady = m.Sets["gen-"+GemmaModel]
	}
	var ready []string
	for _, spec := range l.Cfg.LocalModels {
		s, err := corpus.ParseSpec(spec)
		if err != nil {
			continue
		}
		if strings.HasPrefix(s.Model, "gemma") && !gemmaReady {
			l.state.Local = append(l.state.Local, LocalStats{Gen: s.Model, Skipped: "frozen Gemma holdout not sealed yet; Gemma stays out of training"})
			continue
		}
		if err := localAvailable(ctx, s); err != nil {
			l.state.Local = append(l.state.Local, LocalStats{Gen: s.Model, Skipped: err.Error()})
			continue
		}
		ready = append(ready, spec)
	}
	for i, spec := range ready {
		if ctx.Err() != nil || !time.Now().Before(stopAt) {
			return
		}
		gens, err := corpus.NewGenerators(ctx, spec)
		if err != nil {
			continue
		}
		p, err := l.planner()
		if err != nil {
			continue
		}
		br, err := l.briefer(ctx)
		if err != nil {
			continue
		}
		until := time.Now().Add(time.Until(stopAt) / time.Duration(len(ready)-i))
		l.logf("local: %s generating until %s (quota %d)", gens[0].Name(), until.Format("15:04"), l.Cfg.LocalQuota)
		st := RunLocal(ctx, p, br, gens[0], l.Cfg.Data, until, l.Cfg.LocalQuota, l.Cfg.LocalWorkers, l.Night, l.recordWrite,
			rand.New(rand.NewPCG(l.rng.Uint64(), 2)))
		l.state.Local = append(l.state.Local, st)
	}
}

// retrain fits a candidate into the run directory and gates it.
func (l *Loop) retrain(ctx context.Context, n int) Round {
	dir := filepath.Join(l.dir, fmt.Sprintf("round-%d", n))
	r := Round{N: n, Dir: dir, Samples: countTraining(l.Cfg.Data)}
	l.logf("round %d: fitting candidate on %d training files", n, r.Samples)
	if err := RetrainCandidate(ctx, l.Cfg.Data, dir, FitOptions{Weighting: "source"}); err != nil {
		r.Err = err.Error()
		return r
	}
	cand, err := engine.FromDir(dir)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	m, err := Measure(ctx, cand, l.Cfg.Data, l.fixList, l.Cfg.FixturesDir)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	writeJSONFile(filepath.Join(dir, "metrics.json"), m)
	r.Checks, r.Accepted = Gate(l.prodM, m)
	l.logf("round %d: %s", n, map[bool]string{true: "ACCEPTED", false: "rejected"}[r.Accepted])
	return r
}

func countTraining(d Data) int {
	n := 0
	for _, g := range Genres {
		filepath.WalkDir(d.Dir(g), func(p string, de os.DirEntry, err error) error {
			if err == nil && !de.IsDir() && strings.HasSuffix(p, ".txt") && Trainable(strings.TrimSuffix(filepath.Base(p), ".txt")) &&
				(strings.Contains(p, "/human/") || strings.Contains(p, "/ai/")) {
				n++
			}
			return nil
		})
	}
	return n
}

func (l *Loop) measure(ctx context.Context, rep *Report) error {
	var err error
	if rep.Genres, err = MeasureGenres(ctx, l.prod, l.Cfg.Data); err != nil {
		return err
	}
	if rep.Polished, err = MeasurePolished(ctx, l.prod, l.Cfg.Data); err != nil {
		return err
	}
	if l.Cfg.LOFO && !l.Cfg.DryRun {
		rep.LOFO, err = LeaveOneFamilyOut(ctx, l.Cfg.Data, l.prod, filepath.Join(l.dir, "lofo"))
	}
	return err
}

func writeJSONFile(path string, v any) error {
	b, _ := json.MarshalIndent(v, "", " ")
	return os.WriteFile(path, b, 0o644)
}
