package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/fixtures"
	"github.com/joeshirey/AIDetector/internal/overnight"
)

const overnightUsage = `corpus overnight: the unattended improvement loop (see README, "Overnight loop")

  init           create the frozen sets and final holdout (once), check the GPT split
  gpt-check      report frozen GPT test-split mirrors and check for split leaks
  frozen-gen     generate a frozen generator set (-gens; -plan prints the estimate only)
  collect        collect finished batch jobs (frozen, effort, or training)
  seal           seal a generated frozen set (-gen NAME) once its jobs are collected
  effort         submit the effort experiment (-n per effort per model; -plan for the estimate)
  effort-report  score the effort experiment and write runs/effort/result.json
  run            run one night (until -until; -budget-usd over 2 needs 'approve')
  gate           measure production, fit a candidate into -run-root/manual, gate it (no spend)
  flips          list fresh-pool texts that cross 60 between production and a candidate (-name DIR)
  holdout        one logged look at the final holdout: shipped vs a candidate (-name DIR)
  add-pool       reseal the fresh-pool checksums after adding a new pool directory
  approve        record Joe's approval for full-budget nights
  verify         verify frozen-set checksums (-holdout includes the final holdout)
  status         list batch jobs
`

func runOvernight(ctx context.Context, args []string) error {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, overnightUsage)
		os.Exit(2)
	}
	cmd := args[0]
	fs := flag.NewFlagSet("overnight "+cmd, flag.ExitOnError)
	dataRoot := fs.String("data", ".", "main checkout holding the corpus directories")
	runRoot := fs.String("run-root", "runs/overnight", "state, bandit, batch jobs, history, and candidate models (relative to -data)")
	ledger := fs.String("ledger", "runs/ledger.jsonl", "usage ledger (relative to -data)")
	reports := fs.String("reports", "reports/overnight", "morning reports (relative to -data)")
	bucket := fs.String("bucket", "dss-model-study-aidetector-batch", "Cloud Storage bucket for batch jobs")
	fixDir := fs.String("fixtures", "internal/engine/testdata/fixtures", "fixture cache (relative to -data)")
	fixList := fs.String("fixture-list", "internal/engine/testdata/fixtures.json", "fixture list (relative to -data)")
	budget := fs.Float64("budget-usd", 2, "hard spending cap for the night (USD)")
	caps := fs.String("model-caps", "claude-haiku-5-5=18,gemini-3.8-flash=10,claude-sonnet-5-5=8,gemini-4-argon=8,claude-opus-5-5=10",
		"per-model caps for the night, model=USD, comma-separated")
	until := fs.String("until", "07:00", "run: local time to be finished by (next morning if earlier than now)")
	maxNew := fs.Int("max-new", 2000, "run: max paid requests per generator per night (safety cap)")
	maxTotal := fs.Int("max-total", 2000, "run: max paid requests per night, split among generators by sample share")
	paid := fs.String("paid", "auto", "run: paid generation: auto (pauses after 3 runs without a candidate worth promoting, resumes if a frozen set slips), on, or off")
	argon := fs.Bool("argon-training", true, "run: let Argon generate training text (approved by Joe 2026-10-29; -argon-training=false makes it eval-only)")
	gemma := fs.String("gemma", overnight.GemmaModel, "Gemma model name (its corpus directory); served by mlx_lm.server")
	local := fs.String("local", "", "run: local generators, comma-separated, e.g. mlx:gemma-4-26b-a4b-it-8bit or lmstudio:qwen3.6-35b-a3b (default: none)")
	workers := fs.Int("workers", 2, "local requests in flight per model")
	quota := fs.Int("local-quota", 600, "run: max local samples per model per night")
	mineN := fs.Int("mine-n", 30, "run: human false positives promoted per kind per night")
	sourceCap := fs.Int("source-cap", 8, "run: max promotions per source per night")
	rounds := fs.Int("rounds", 3, "run: retrain-and-gate rounds (at most 3 accepted)")
	analyst := fs.String("analyst", "claude:claude-sonnet-5-5", "run: model for the morning failure analysis")
	lofo := fs.String("lofo", "auto", "run: leave-one-family-out: auto (Sundays), on, or off")
	gens := fs.String("gens", "", "frozen-gen: generators, provider:model, comma-separated")
	plan := fs.Bool("plan", false, "frozen-gen/effort: print requests and estimated cost without submitting")
	gen := fs.String("gen", "", "seal: generator directory name")
	n := fs.Int("n", 50, "effort: samples per effort level per model")
	holdout := fs.Bool("holdout", false, "verify: include the final holdout")
	excludeSince := fs.String("exclude-since", "", "gate: leave AI generations written after this time (RFC 3339) out of the candidate")
	excludeGen := fs.String("exclude-gen", "", "gate: with -exclude-since, only generations from generator directories starting with this")
	excludeHumanSince := fs.String("exclude-human-since", "", "gate: leave human training files added after this time (RFC 3339; e.g. promotions) out of the candidate")
	candDir := fs.String("candidate", "", "gate: gate an existing candidate directory instead of fitting one")
	weighting := fs.String("weighting", "", "gate: sample weighting for the candidate fit: \"\" or source (each source and its mirrors weigh the same)")
	denseL2 := fs.Bool("dense-l2", false, "gate: fit with experimental dense word-pattern regularization")
	shipped := fs.Bool("shipped", false, "gate: compare against production's shipped cutoffs instead of a recalibrated baseline")
	keepCutoffs := fs.Bool("keep-cutoffs", false, "gate: keep production's label cutoffs instead of recalibrating (diagnostic)")
	outName := fs.String("name", "manual", "gate: candidate directory under -run-root; flips: candidate directory to compare")
	pools := fs.String("pools", "fresh-blog,nontech-medium,tech-pool,fresh-memo", "flips: fresh pools to compare")
	excludeStyle := fs.String("exclude-style", "", "gate: leave generations of these prompt styles (comma-separated: edit, continue, persona) out of the candidate")
	fs.Parse(args[1:])
	overnight.GemmaModel = *gemma
	// No local generators by default (2026-10-29): the released model
	// already catches Qwen (AUROC 1.0) and Gemma (0.998-1.000, before any
	// Gemma training text), and Qwen's samples added nothing when tested.
	// Gemma stays a sealed evaluation set.

	abs := func(p string) string {
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(*dataRoot, p)
	}
	root, err := filepath.Abs(*dataRoot)
	if err != nil {
		return err
	}
	d := overnight.Data{Root: root}
	modelCaps := map[string]float64{}
	for _, kv := range strings.Split(*caps, ",") {
		if k, v, ok := strings.Cut(strings.TrimSpace(kv), "="); ok {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return fmt.Errorf("-model-caps %q: %w", kv, err)
			}
			modelCaps[k] = f
		}
	}
	if err := os.MkdirAll(abs(*runRoot), 0o755); err != nil {
		return err
	}
	// Commands other than run share one budget and the ledger.
	if cmd != "run" {
		corpus.Spend = corpus.NewBudget(*budget)
		corpus.Spend.ModelCaps = modelCaps
		corpus.Spend.Purpose = "overnight " + cmd
		if err := corpus.Spend.OpenLedger(abs(*ledger)); err != nil {
			return err
		}
	}
	night := corpus.NightOf(time.Now()).Format("2006-01-02")
	jobsPath := filepath.Join(abs(*runRoot), "batch-jobs.json")
	logw := os.Stderr

	switch cmd {
	case "init":
		m, err := d.InitFrozen(ctx, abs(*fixDir), logw)
		if err != nil {
			return err
		}
		fmt.Printf("manifest written with %d sets\n", len(m.Sets))
		return printGPT(d)

	case "gpt-check":
		return printGPT(d)

	case "frozen-gen":
		if *gens == "" {
			return errors.New("frozen-gen needs -gens")
		}
		gs, err := corpus.NewGenerators(ctx, *gens)
		if err != nil {
			return err
		}
		bg, err := corpus.NewGenerators(ctx, "claude:claude-haiku-5-5")
		if err != nil {
			return err
		}
		br := &overnight.Briefer{Gen: bg[0], Dir: d.Path("briefs", "frozen")}
		for _, g := range gs {
			reqs, est, err := overnight.FrozenRequests(ctx, d, br, g, rand.New(rand.NewPCG(5, 5)))
			if err != nil {
				return err
			}
			fmt.Printf("%s: %d frozen requests, estimated $%.2f\n", g.Name(), len(reqs), est)
			if *plan || len(reqs) == 0 {
				continue
			}
			if p := g.Spec().Provider; p == "lmstudio" || p == "mlx" {
				st := overnight.RunFrozenLocal(ctx, d, g, reqs, *workers, logw)
				fmt.Printf("%s: %d written, %d failed\n", g.Name(), st.Written, st.Failed)
				continue
			}
			b, jobs, err := batchSetup(ctx, *bucket, jobsPath)
			if err != nil {
				return err
			}
			t, err := overnight.SubmitPaid(ctx, b, jobs, g.Spec(), night, "frozen", reqs)
			if err != nil {
				return err
			}
			fmt.Printf("%s: submitted %s\n", g.Name(), t.Job.Name)
		}
		fmt.Println(corpus.Spend)
		return nil

	case "collect":
		b, jobs, err := batchSetup(ctx, *bucket, jobsPath)
		if err != nil {
			return err
		}
		c := &overnight.Collector{Data: d, Batcher: b, Jobs: jobs, Log: logw, OnWrite: overnight.QueueForScoring(abs(*runRoot))}
		done, carried, err := c.Collect(ctx, night)
		fmt.Printf("collected %d jobs, %d still running\n", done, carried)
		return err

	case "seal":
		if *gen == "" {
			return errors.New("seal needs -gen")
		}
		jobs, err := overnight.LoadJobs(jobsPath)
		if err != nil {
			return err
		}
		for _, t := range jobs.Pending() {
			if t.Purpose == "frozen" && t.Job.Spec.Name() == *gen {
				return fmt.Errorf("%s still has a running frozen job (%s); run collect first", *gen, t.Job.Name)
			}
		}
		m, err := d.LoadManifest()
		if err != nil {
			return err
		}
		if s, ok := m.Sets["gen-"+*gen]; ok && s.Sealed {
			return fmt.Errorf("gen-%s is already sealed", *gen)
		}
		files, err := d.SealGenerated(m, *gen)
		if err != nil {
			return err
		}
		fmt.Printf("sealed gen-%s: %d files\n", *gen, files)
		return nil

	case "effort":
		srcs, err := overnight.EffortSources(d, *n)
		if err != nil {
			return err
		}
		bg, err := corpus.NewGenerators(ctx, "claude:claude-haiku-5-5")
		if err != nil {
			return err
		}
		br := &overnight.Briefer{Gen: bg[0], Dir: d.Path("briefs", "train")}
		var b *corpus.Batcher
		var jobs *overnight.Jobs
		if !*plan {
			if b, jobs, err = batchSetup(ctx, *bucket, jobsPath); err != nil {
				return err
			}
		}
		total := 0.0
		for _, spec := range overnight.EffortModels {
			for _, eff := range []string{"", "@low"} {
				s, _ := corpus.ParseSpec(spec + eff)
				reqs, est, err := overnight.EffortRequests(ctx, srcs, br, s)
				if err != nil {
					return err
				}
				total += est
				fmt.Printf("%s: %d requests, estimated $%.2f\n", s.Name(), len(reqs), est)
				if *plan {
					continue
				}
				if _, err := overnight.SubmitPaid(ctx, b, jobs, s, night, "effort", reqs); err != nil {
					return fmt.Errorf("%s: %w", s.Name(), err)
				}
			}
		}
		fmt.Printf("total estimated $%.2f; %s\n", total, corpus.Spend)
		return nil

	case "effort-report":
		jobs, err := overnight.LoadJobs(jobsPath)
		if err != nil {
			return err
		}
		res, err := overnight.AnalyzeEffort(ctx, d, jobs)
		if err != nil {
			return err
		}
		fmt.Println("| Model | N default/low | Mean score default/low | KS (critical) | Tokens/sample default (in/out/think) | low | $/sample default/low | Low OK |")
		fmt.Println("|---|---|---|---|---|---|---|---|")
		efforts := map[string]string{}
		for _, r := range res {
			fmt.Printf("| %s | %d/%d | %.1f/%.1f | %.3f (%.3f) | %d/%d/%d | %d/%d/%d | %.5f/%.5f | %v |\n", r.Model, r.NDefault, r.NLow,
				r.MeanDefault, r.MeanLow, r.KS, r.KSCritical, r.TokensDefault.Input, r.TokensDefault.Output, r.TokensDefault.Thinking,
				r.TokensLow.Input, r.TokensLow.Output, r.TokensLow.Thinking, r.CostDefault, r.CostLow, r.Match)
			if r.Match {
				efforts[r.Model] = "low"
			}
		}
		b, _ := json.MarshalIndent(map[string]any{"results": res, "efforts": efforts}, "", " ")
		return os.WriteFile(d.Path("runs", "effort", "result.json"), b, 0o644)

	case "gate":
		var fl *fixtures.List
		if fl, err = fixtures.Load(abs(*fixList)); err != nil {
			return err
		}
		start := time.Now()
		// Baseline: production's models with cutoffs refit the same way as
		// the candidate's (cached; delete the directory to redo it).
		base := engine.Default()
		if !*shipped {
			bdir := filepath.Join(abs(*runRoot), "baseline")
			if _, err := os.Stat(filepath.Join(bdir, "thresholds.json")); err != nil {
				if err := overnight.ProductionBaseline(ctx, d, bdir); err != nil {
					return err
				}
			}
			if base, err = engine.FromDir(bdir); err != nil {
				return err
			}
		}
		prod, err := overnight.Measure(ctx, base, d, fl, abs(*fixDir))
		if err != nil {
			return err
		}
		fmt.Printf("baseline measured in %s\n", time.Since(start).Round(time.Second))
		dir := filepath.Join(abs(*runRoot), *outName)
		start = time.Now()
		var rules []func(path, id string) bool
		sep := string(filepath.Separator)
		newerThan := func(path string, t time.Time) bool {
			st, err := os.Stat(path)
			return err == nil && st.ModTime().After(t)
		}
		if *excludeSince != "" {
			since, err := time.Parse(time.RFC3339, *excludeSince)
			if err != nil {
				return fmt.Errorf("-exclude-since: %w", err)
			}
			rules = append(rules, func(path, id string) bool {
				i := strings.Index(path, sep+"ai"+sep)
				return i >= 0 && strings.HasPrefix(path[i+4:], *excludeGen) && newerThan(path, since)
			})
		}
		if *excludeHumanSince != "" {
			since, err := time.Parse(time.RFC3339, *excludeHumanSince)
			if err != nil {
				return fmt.Errorf("-exclude-human-since: %w", err)
			}
			rules = append(rules, func(path, id string) bool {
				return strings.Contains(path, sep+"human"+sep) && newerThan(path, since)
			})
		}
		if *excludeStyle != "" {
			styles := strings.Split(*excludeStyle, ",")
			rules = append(rules, func(path, id string) bool {
				for _, st := range styles {
					if strings.HasSuffix(id, "~"+st) {
						return true
					}
				}
				return false
			})
		}
		exclude := func(path, id string) bool {
			for _, r := range rules {
				if r(path, id) {
					return true
				}
			}
			return false
		}
		if *candDir != "" {
			dir = *candDir
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(abs(*runRoot), dir)
			}
		} else if err := overnight.RetrainCandidate(ctx, d, dir, overnight.FitOptions{Exclude: exclude, KeepCutoffs: *keepCutoffs,
			Weighting: *weighting, DenseL2: *denseL2}); err != nil {
			return err
		}
		fmt.Printf("candidate fit in %s\n", time.Since(start).Round(time.Second))
		cand, err := engine.FromDir(dir)
		if err != nil {
			return err
		}
		cm, err := overnight.Measure(ctx, cand, d, fl, abs(*fixDir))
		if err != nil {
			return err
		}
		if b, err := json.Marshal(cm); err == nil {
			os.WriteFile(filepath.Join(dir, "metrics.json"), b, 0o644)
		}
		checks, ok := overnight.Gate(prod, cm)
		for _, c := range checks {
			fmt.Printf("%-5v %-48s %.4f -> %.4f  p=%.3f\n", c.Pass, c.Name, c.Before, c.After, c.P)
		}
		fmt.Printf("accepted: %v (candidate in %s)\n", ok, dir)
		return nil

	case "flips":
		dir := *outName
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(abs(*runRoot), dir)
		}
		cand, err := engine.FromDir(dir)
		if err != nil {
			return err
		}
		return overnight.Flips(ctx, os.Stdout, d, engine.Default(), cand, strings.Split(*pools, ","))

	case "effort-paired":
		rows, err := overnight.PairedEffort(ctx, d, 3)
		if err != nil {
			return err
		}
		fmt.Println("| Model | Kind | Pairs | Mean raw diff (default−low) | 95% CI | Wilcoxon p | Within ±3 | Words default/low |")
		fmt.Println("|---|---|---:|---:|---|---:|---|---|")
		for _, r := range rows {
			fmt.Printf("| %s | %s | %d | %+.2f | [%+.2f, %+.2f] | %.3f | %v | %.0f/%.0f |\n", r.Model, r.Kind, r.N, r.MeanDiff, r.CILow, r.CIHigh, r.WilcoxonP, r.Equivalent, r.WordsDefault, r.WordsLow)
		}
		return nil

	case "holdout":
		dir := *outName
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(abs(*runRoot), dir)
		}
		cand, err := engine.FromDir(dir)
		if err != nil {
			return err
		}
		// Every look is logged: the holdout is only worth something while
		// it's rarely used.
		logf, err := os.OpenFile(filepath.Join(abs(*runRoot), "holdout-looks.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		b, _ := json.Marshal(map[string]string{"time": time.Now().Format(time.RFC3339), "candidate": dir})
		logf.Write(append(b, '\n'))
		logf.Close()
		ship, err := overnight.ScoreHoldout(ctx, d, engine.Default())
		if err != nil {
			return err
		}
		cr, err := overnight.ScoreHoldout(ctx, d, cand)
		if err != nil {
			return err
		}
		fmt.Println("| Pool | Docs | Shipped 60+ | Candidate 60+ | Shipped 80+ | Candidate 80+ | Segments | Shipped seg 60+ | Candidate seg 60+ |")
		fmt.Println("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
		for i := range ship {
			a, c := ship[i], cr[i]
			fmt.Printf("| %s | %d | %d | %d | %d | %d | %d | %d | %d |\n", a.Pool, a.Docs, a.Doc60, c.Doc60, a.Doc80, c.Doc80, a.Segs, a.Seg60, c.Seg60)
		}
		for i := range ship {
			fmt.Printf("%s flagged: shipped %v; candidate %v\n", ship[i].Pool, ship[i].Flagged, cr[i].Flagged)
		}
		return nil

	case "fixtures":
		dir := *outName
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(abs(*runRoot), dir)
		}
		cand, err := engine.FromDir(dir)
		if err != nil {
			return err
		}
		fl, err := fixtures.Load(abs(*fixList))
		if err != nil {
			return err
		}
		a, err := overnight.FixtureRows(ctx, engine.Default(), fl, abs(*fixDir))
		if err != nil {
			return err
		}
		b, err := overnight.FixtureRows(ctx, cand, fl, abs(*fixDir))
		if err != nil {
			return err
		}
		fmt.Printf("pass: shipped %v, candidate %v (missing %d)\n", a.Pass(), b.Pass(), b.Missing)
		fmt.Println("| Fixture | Set | Shipped doc / max seg / label | Candidate doc / max seg / label |")
		fmt.Println("|---|---|---|---|")
		for i := range a.Rows {
			x, y := a.Rows[i], b.Rows[i]
			set := "known human"
			if x.Monitoring {
				set = "monitoring"
			}
			mark := ""
			if x.Label != y.Label {
				mark = " **changed**"
			}
			fmt.Printf("| %s | %s | %d / %d / %s | %d / %d / %s%s |\n", x.ID, set, x.Doc, x.MaxSeg, x.Label, y.Doc, y.MaxSeg, y.Label, mark)
		}
		return nil

	case "add-pool":
		// Existing pool files must verify before the sets are resealed with
		// the new pool's files.
		m, err := d.LoadManifest()
		if err != nil {
			return err
		}
		vs, err := d.Verify(m, abs(*fixDir), true)
		if err != nil {
			return err
		}
		for _, v := range vs {
			if (v.Set == "fresh-pools" || v.Set == "final-holdout") && (v.Changed > 0 || v.Missing > 0) {
				return fmt.Errorf("%s has changed or missing files; not resealing", v.Set)
			}
		}
		p, h, err := d.ResealPools(m)
		if err != nil {
			return err
		}
		fmt.Printf("fresh-pools %d files, final-holdout %d files\n", p, h)
		return nil

	case "redraw-holdout":
		m, err := d.LoadManifest()
		if err != nil {
			return err
		}
		p, h, err := d.RedrawHoldout(m)
		if err != nil {
			return err
		}
		fmt.Printf("fresh-pools %d files, final-holdout %d files\n", p, h)
		return nil

	case "approve":
		p := filepath.Join(abs(*runRoot), "full-budget-approved")
		if err := os.WriteFile(p, []byte(time.Now().Format(time.RFC3339)+"\n"), 0o644); err != nil {
			return err
		}
		fmt.Println("full-budget nights approved:", p)
		return nil

	case "verify":
		m, err := d.LoadManifest()
		if err != nil {
			return err
		}
		vs, err := d.Verify(m, abs(*fixDir), *holdout)
		if err != nil {
			return err
		}
		bad := 0
		for _, v := range vs {
			status := "ok"
			if !v.OK() {
				status = fmt.Sprintf("CHANGED (%d changed, %d missing, %d added)", v.Changed, v.Missing, v.Added)
				bad++
			}
			fmt.Printf("%-24s %6d files  %s\n", v.Set, v.Files, status)
		}
		if bad > 0 {
			return fmt.Errorf("%d frozen sets changed", bad)
		}
		return nil

	case "status":
		jobs, err := overnight.LoadJobs(jobsPath)
		if err != nil {
			return err
		}
		for _, t := range jobs.List {
			fmt.Printf("%s  %-8s %-28s %4d req  est $%.2f  actual $%.2f  %s  collected=%v %s\n", t.Job.Night, t.Purpose, t.Job.Spec.Name(),
				t.Job.Requests, t.Job.Estimate, t.Actual, t.Job.State, t.Collected, t.Note)
		}
		return nil

	case "run":
		if *budget > 2 {
			if _, err := os.Stat(filepath.Join(abs(*runRoot), "full-budget-approved")); err != nil {
				return fmt.Errorf("-budget-usd %.0f needs Joe's approval first (corpus overnight approve); dry runs use -budget-usd 2", *budget)
			}
		}
		stop, err := parseUntil(*until)
		if err != nil {
			return err
		}
		efforts := map[string]string{}
		if b, err := os.ReadFile(d.Path("runs", "effort", "result.json")); err == nil {
			var r struct{ Efforts map[string]string }
			if json.Unmarshal(b, &r) == nil {
				efforts = r.Efforts
			}
		}
		doLOFO := *lofo == "on" || (*lofo == "auto" && corpus.NightOf(time.Now()).Weekday() == time.Sunday)
		var locals []string
		for _, s := range strings.Split(*local, ",") {
			if s = strings.TrimSpace(s); s != "" {
				locals = append(locals, s)
			}
		}
		dry := *budget <= 2
		cfg := overnight.Config{
			Data: d, RunRoot: abs(*runRoot), Ledger: abs(*ledger), ReportDir: abs(*reports), Bucket: *bucket,
			FixturesDir: abs(*fixDir), FixtureList: abs(*fixList),
			Until: stop, BudgetUSD: *budget, ModelCaps: modelCaps, MaxNew: *maxNew, MaxTotal: *maxTotal, PaidMode: *paid, ArgonTraining: *argon, Efforts: efforts,
			LocalModels: locals, LocalWorkers: *workers, LocalQuota: *quota,
			MineN: *mineN, SourceCap: *sourceCap, Rounds: *rounds, Analyst: *analyst, LOFO: doLOFO, DryRun: dry,
			Log: logw,
		}
		if dry {
			cfg.MaxNew = min(cfg.MaxNew, 20)
			cfg.MaxTotal = min(cfg.MaxTotal, 80)
			cfg.LocalQuota = min(cfg.LocalQuota, 20)
			cfg.MineN = min(cfg.MineN, 5)
			cfg.Rounds = 1
		}
		ctx, cancel := context.WithDeadline(ctx, stop)
		defer cancel()
		p, err := overnight.Run(ctx, cfg)
		if p != "" {
			fmt.Println("report:", p)
		}
		return err
	}
	fmt.Fprint(os.Stderr, overnightUsage)
	return fmt.Errorf("unknown overnight command %q", cmd)
}

func batchSetup(ctx context.Context, bucket, jobsPath string) (*corpus.Batcher, *overnight.Jobs, error) {
	b, err := corpus.NewBatcher(ctx, bucket)
	if err != nil {
		return nil, nil, err
	}
	j, err := overnight.LoadJobs(jobsPath)
	return b, j, err
}

func printGPT(d overnight.Data) error {
	g, err := d.CheckGPTSplit()
	if err != nil {
		return err
	}
	total := 0
	for genre, n := range g.TestMirrors {
		fmt.Printf("frozen GPT test-split mirrors, %s: %d\n", genre, n)
		total += n
	}
	fmt.Printf("frozen GPT test-split mirrors, total: %d\n", total)
	if len(g.Leaks) == 0 {
		fmt.Println("no source text or PR diff appears on both sides of the split")
	}
	for _, l := range g.Leaks {
		fmt.Println("LEAK:", l)
	}
	return nil
}

// parseUntil turns "07:00" into the next such local time.
func parseUntil(s string) (time.Time, error) {
	t, err := time.ParseInLocation("15:04", s, time.Local)
	if err != nil {
		return time.Time{}, fmt.Errorf("-until %q: want HH:MM", s)
	}
	now := time.Now()
	stop := time.Date(now.Year(), now.Month(), now.Day(), t.Hour(), t.Minute(), 0, 0, time.Local)
	if !stop.After(now) {
		stop = stop.AddDate(0, 0, 1)
	}
	return stop, nil
}
