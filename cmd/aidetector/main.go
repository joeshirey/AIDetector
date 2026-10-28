// Command aidetector scores text for AI authorship from the command line.
//
//	aidetector [detect] [-kind K] [-json] [file|-]
//	aidetector eval [-kind K] [-split train|test] [-heuristic] [-json] DIR...
//	aidetector fit  -kind K DIR...
//	aidetector dist [-kind K] DIR...   score distribution of a folder (e.g. the human pool)
//	aidetector thresholds -kind K DIR...  set label cutoffs from human-only text
//	aidetector setup [-client all|claude|codex|antigravity] [-dry-run]  register the MCP server and install the skill
//	aidetector fixtures [-fetch-only]  fetch and score the regression fixtures (run after every refit)
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"text/tabwriter"

	"github.com/joeshirey/AIDetector/internal/calibrate"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/ngram"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/eval"
	"github.com/joeshirey/AIDetector/internal/fixtures"
	"github.com/joeshirey/AIDetector/internal/setup"
	"github.com/joeshirey/AIDetector/internal/split"
)

func main() {
	args := os.Args[1:]
	cmd := "detect"
	if len(args) > 0 {
		switch args[0] {
		case "detect", "eval", "fit", "dist", "thresholds", "setup", "fixtures":
			cmd, args = args[0], args[1:]
		case "-h", "-help", "--help", "help":
			usage()
			return
		}
	}
	var err error
	switch cmd {
	case "detect":
		err = runDetect(args)
	case "eval":
		err = runEval(args)
	case "fit":
		err = runFit(args)
	case "dist":
		err = runDist(args)
	case "thresholds":
		err = runThresholds(args)
	case "setup":
		err = runSetup(args)
	case "fixtures":
		err = runFixtures(args)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "aidetector:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage:
  aidetector setup [-client all|claude|codex|antigravity] [-dry-run] [-skill-only] [-scorer URL]
  aidetector fixtures [-fetch-only]   fetch (cached, gitignored) and score the regression fixtures
  aidetector [detect] [-kind auto|prose|pr] [-json] [file|-]
  aidetector eval [-kind K] [-split train|test] [-heuristic] [-json] DIR...   (DIR/human/*, DIR/ai/*)
  aidetector fit -kind prose|pr DIR...   (writes internal/engine/models/KIND.json)
  aidetector dist [-kind K] DIR...   score distribution for every .txt under DIR (pool false-positive rate, variant sets)
  aidetector thresholds -kind K DIR...   set label cutoffs from human-only text (writes internal/engine/models/thresholds.json)
`)
}

func runDetect(args []string) error {
	fs := flag.NewFlagSet("detect", flag.ExitOnError)
	kind := fs.String("kind", "auto", "auto, prose, or pr")
	asJSON := fs.Bool("json", false, "print full JSON result")
	fs.Parse(args)

	var r io.Reader = os.Stdin
	if p := fs.Arg(0); p != "" && p != "-" {
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		defer f.Close()
		r = f
	}
	b, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	res, err := engine.Default().Analyze(context.Background(), detect.Input{Text: string(b), Kind: detect.ParseKind(*kind)})
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(res)
	}
	fmt.Printf("Score: %d/100 (%s, %s confidence)\nKind: %s, %d words\n\n", res.Score, res.Label, res.Confidence, res.Kind, res.WordCount)
	fmt.Println("Why:")
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	for _, r := range res.Reasons {
		typical := ""
		if r.TypicalH != nil && r.Signal != "word_patterns" {
			typical = fmt.Sprintf("(typical human %s, AI %s)", fmtNum(*r.TypicalH), fmtNum(*r.TypicalAI))
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\n", arrow(r.Direction), r.Impact, r.Detail, typical)
	}
	tw.Flush()
	if len(res.Segments) > 0 {
		fmt.Printf("\nSegments (%.0f%% at 60+):\n", *res.AIFraction*100)
		for _, s := range res.Segments {
			fmt.Printf("  [%3d] %s\n", s.Score, s.Preview)
		}
	}
	for _, n := range res.Notes {
		fmt.Println("\nNote:", n)
	}
	return nil
}

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	kind := fs.String("kind", "auto", "auto, prose, or pr")
	asJSON := fs.Bool("json", false, "print per-sample JSON")
	splitName := fs.String("split", "", "only score the train or test split")
	heuristic := fs.Bool("heuristic", false, "ignore fitted models; use hand-set weights")
	fs.Parse(args)
	if fs.NArg() == 0 {
		usage()
		os.Exit(2)
	}
	e := engine.Default()
	if *heuristic {
		e = engine.Heuristic()
	}
	r, err := eval.Run(context.Background(), e, fs.Args(), detect.ParseKind(*kind), *splitName)
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(r)
	}
	fmt.Printf("samples: %d human, %d ai\n", r.Human, r.AI)
	fmt.Printf("AUROC:   %.3f\n", r.AUROC)
	fmt.Printf("@50:     catches %.0f%% of AI, flags %.0f%% of human\n", r.TPRAt50*100, r.FPRAt50*100)
	fmt.Printf("@%d:     catches %.0f%% of AI at <=1%% false positives\n", r.Threshold1, r.TPRAtThresh*100)
	fmt.Println("\nworst misses:")
	for _, s := range r.Samples {
		if !s.IsAI && s.Score >= 50 {
			fmt.Printf("  human scored %3d  %s\n", s.Score, s.Path)
		}
	}
	for i := len(r.Samples) - 1; i >= 0; i-- {
		if s := r.Samples[i]; s.IsAI && s.Score < 50 {
			fmt.Printf("  ai    scored %3d  %s\n", s.Score, s.Path)
		}
	}
	return nil
}

func runFit(args []string) error {
	fs := flag.NewFlagSet("fit", flag.ExitOnError)
	kind := fs.String("kind", "", "prose or pr")
	out := fs.String("out", "internal/engine/models", "output directory")
	withLM := fs.Bool("lm", false, "include token-probability features from the local scorer; writes KIND.lm.json")
	fs.Parse(args)
	k := detect.ParseKind(*kind)
	if k == detect.KindAuto || fs.NArg() == 0 {
		usage()
		os.Exit(2)
	}
	// Features come from the heuristic engine so fitting never depends on a
	// previously fitted model.
	he := engine.Heuristic()
	if *withLM && he.Scorer() == nil {
		return fmt.Errorf("-lm needs a running scorer at $AIDETECTOR_SCORER_URL")
	}
	r, err := eval.Run(context.Background(), he, fs.Args(), k, "train")
	if err != nil {
		return err
	}
	if !*withLM {
		for _, s := range r.Samples {
			for n := range s.Feats {
				if strings.HasPrefix(n, "lm_") {
					delete(s.Feats, n)
				}
			}
		}
	}
	docs := make([]string, len(r.Samples))
	labels := make([]bool, len(r.Samples))
	for i, s := range r.Samples {
		b, err := os.ReadFile(s.Path)
		if err != nil {
			return err
		}
		docs[i], labels[i] = string(b), s.IsAI
	}

	// Word-pattern scores for the calibration step come from 5-fold
	// out-of-fold models, so the calibrator sees how the classifier does on
	// text it wasn't trained on rather than its overconfident training fit.
	const folds = 5
	oof := make([]float64, len(docs))
	for f := range folds {
		var trDocs []string
		var trLabels []bool
		for i, s := range r.Samples {
			if fold(s.ID, folds) != f {
				trDocs, trLabels = append(trDocs, docs[i]), append(trLabels, labels[i])
			}
		}
		fm := ngram.Train(string(k), trDocs, trLabels, ngram.DefaultTrainOptions())
		for i, s := range r.Samples {
			if fold(s.ID, folds) == f {
				oof[i], _ = fm.Logit(docs[i])
			}
		}
	}
	words := ngram.Train(string(k), docs, labels, ngram.DefaultTrainOptions())

	rows := make([]calibrate.Row, len(r.Samples))
	for i, s := range r.Samples {
		feats := map[string]float64{"word_patterns": oof[i]}
		for n, v := range s.Feats {
			feats[n] = v
		}
		rows[i] = calibrate.Row{Feats: feats, Words: s.Words, IsAI: s.IsAI}
	}
	m := calibrate.Fit(string(k), rows)
	path := *out + "/" + string(k) + ".json"
	if *withLM {
		path = *out + "/" + string(k) + ".lm.json"
	}
	if err := writeJSON(path, m); err != nil {
		return err
	}
	if err := writeJSON(*out+"/"+string(k)+".ngram.json", words); err != nil {
		return err
	}
	fmt.Printf("word-pattern model: %d terms\n", len(words.Weights))
	fmt.Printf("fit %s on %d train samples (%d human, %d ai) -> %s\n", k, len(rows), r.Human, r.AI, path)
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "FEATURE\tWEIGHT\tIF MISSING")
	for _, f := range m.Features {
		fmt.Fprintf(tw, "%s\t%+.2f\t%+.2f\n", f.Name, f.Weight, m.Missing[f.Name])
	}
	tw.Flush()
	fmt.Println("\nrebuild, then check held-out accuracy with: aidetector eval -split test", strings.Join(fs.Args(), " "))
	return nil
}

func runDist(args []string) error {
	fs := flag.NewFlagSet("dist", flag.ExitOnError)
	kind := fs.String("kind", "auto", "auto, prose, or pr")
	worst := fs.Int("worst", 5, "list this many highest-scoring files")
	segs := fs.Bool("segments", false, "also report the share of 200-word segments at 40/60/80")
	dump := fs.String("dump", "", "write per-text raw and segment scores as JSON lines to this file (for analysis)")
	fs.Parse(args)
	if fs.NArg() == 0 {
		usage()
		os.Exit(2)
	}
	e := engine.Default()
	all, err := scoreFiles(e, fs.Args(), detect.ParseKind(*kind))
	if err != nil {
		return err
	}
	at := func(t int) float64 {
		n := 0
		for _, s := range all {
			if s.score >= t {
				n++
			}
		}
		return 100 * float64(n) / float64(len(all))
	}
	fmt.Printf("%d texts  median %d  |  scored 40+: %.1f%%  60+: %.1f%%  80+: %.1f%%\n",
		len(all), all[len(all)/2].score, at(40), at(60), at(80))
	if *dump != "" {
		f, err := os.Create(*dump)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(f)
		for _, s := range all {
			enc.Encode(map[string]any{"id": s.id, "path": s.path, "raw": s.raw, "score": s.score, "seg_raw": s.segRaw, "seg": s.segs, "seg_preview": s.segPreview})
		}
		f.Close()
	}
	if *segs {
		var n, s40, s60, s80, docsWith int
		for _, s := range all {
			if len(s.segs) > 0 {
				docsWith++
			}
			for _, v := range s.segs {
				n++
				if v >= 40 {
					s40++
				}
				if v >= 60 {
					s60++
				}
				if v >= 80 {
					s80++
				}
			}
		}
		if n > 0 {
			p := func(k int) float64 { return 100 * float64(k) / float64(n) }
			fmt.Printf("  segments: %d from %d texts  |  40+: %.1f%%  60+: %.1f%%  80+: %.1f%%\n", n, docsWith, p(s40), p(s60), p(s80))
		}
	}
	labels := map[string]int{}
	for _, s := range all {
		labels[s.label]++
	}
	var parts []string
	for _, l := range []string{"human", "likely human", "inconclusive", "likely AI", "AI"} {
		parts = append(parts, fmt.Sprintf("%s %.0f%%", l, 100*float64(labels[l])/float64(len(all))))
	}
	fmt.Println("  labels:", strings.Join(parts, ", "))
	for _, s := range all[:min(*worst, len(all))] {
		fmt.Printf("  %3d  %s\n", s.score, s.path)
	}
	return nil
}

type scored struct {
	path, id, label string
	score, raw      int
	segs, segRaw    []int
	segPreview      []string
}

// scoreFiles scores every .txt in dirs (and one level of subdirectories),
// highest score first.
func scoreFiles(e *engine.Engine, dirs []string, kind detect.Kind) ([]scored, error) {
	var all []scored
	for _, dir := range dirs {
		paths, err := filepath.Glob(filepath.Join(dir, "*.txt"))
		if err != nil {
			return nil, err
		}
		sub, _ := filepath.Glob(filepath.Join(dir, "*", "*.txt"))
		for _, p := range append(paths, sub...) {
			b, err := os.ReadFile(p)
			if err != nil {
				return nil, err
			}
			res, err := e.Analyze(context.Background(), detect.Input{Text: string(b), Kind: kind})
			if err != nil {
				return nil, err
			}
			id := strings.TrimSuffix(filepath.Base(p), ".txt")
			sc := scored{path: p, id: id, label: res.Label, score: res.Score, raw: res.RawScore}
			for _, s := range res.Segments {
				sc.segs = append(sc.segs, s.Score)
				sc.segRaw = append(sc.segRaw, s.RawScore)
				sc.segPreview = append(sc.segPreview, s.Preview)
			}
			all = append(all, sc)
		}
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no .txt files found")
	}
	slices.SortFunc(all, func(a, b scored) int { return b.score - a.score })
	return all, nil
}

// runThresholds sets a kind's label cutoffs from human-only text. Half the
// texts (by id hash) set the cutoffs; the other half measures how often
// human text actually reaches each label, so the reported rates are honest.
func runThresholds(args []string) error {
	fs := flag.NewFlagSet("thresholds", flag.ExitOnError)
	kind := fs.String("kind", "", "prose or pr")
	out := fs.String("out", "internal/engine/models/thresholds.json", "thresholds file to update")
	mixed := fs.Float64("mixed", 0.05, "share of human text allowed at or above the inconclusive cutoff")
	likely := fs.Float64("likely", 0.02, "share of human text allowed at or above the likely-AI cutoff")
	ai := fs.Float64("ai", 0.005, "share of human text allowed at or above the AI cutoff")
	only := fs.String("only", "both", "calibrate documents, segments, or both (segments need far more text, so they can use a wider pool)")
	fs.Parse(args)
	k := detect.ParseKind(*kind)
	if k == detect.KindAuto || fs.NArg() == 0 {
		usage()
		os.Exit(2)
	}
	e := engine.Default()
	all, err := scoreFiles(e, fs.Args(), k)
	if err != nil {
		return err
	}
	key := e.ThresholdKey(k)
	// Cutoffs are in raw-score units; the shared scale is derived from them.
	// Documents and their 200-word segments are calibrated separately, each
	// split by document so a document's segments stay on one side.
	var calib, check, segCalib, segCheck, segDocRaw []int
	for _, s := range all {
		if split.Hash(s.id)%2 == 0 {
			calib = append(calib, s.raw)
			segCalib = append(segCalib, s.segRaw...)
			if len(s.segRaw) > 0 {
				segDocRaw = append(segDocRaw, s.raw) // documents that have segments
			}
		} else {
			check = append(check, s.raw)
			segCheck = append(segCheck, s.segRaw...)
		}
	}
	share := func(scores []int, t int) float64 {
		if len(scores) == 0 {
			return 0
		}
		n := 0
		for _, s := range scores {
			if s >= t {
				n++
			}
		}
		return float64(n) / float64(len(scores))
	}
	// fit picks, on the calibration scores, the lowest cutoffs that at most
	// the allowed shares reach, and measures them on the check scores.
	fit := func(calib, check []int, mixed, likely, ai float64) engine.Thresholds {
		cutoff := func(allowed float64, floor int) int {
			for t := floor; t <= 100; t++ {
				if share(calib, t) <= allowed {
					return t
				}
			}
			return 100
		}
		t := engine.Thresholds{}
		t.Mixed = cutoff(mixed, 1)
		t.LikelyAI = cutoff(likely, t.Mixed+1)
		t.AI = cutoff(ai, t.LikelyAI+1)
		t.HumanAtMixed = share(check, t.Mixed)
		t.HumanAtLikelyAI = share(check, t.LikelyAI)
		t.HumanAtAI = share(check, t.AI)
		t.CalibratedOn = len(calib)
		return t
	}
	t := fit(calib, check, *mixed, *likely, *ai)

	file := map[string]engine.Thresholds{}
	if b, err := os.ReadFile(*out); err == nil {
		if err := json.Unmarshal(b, &file); err != nil {
			return err
		}
	}
	if *only != "segments" { // -only segments keeps the existing document cutoffs
		file[key] = t
	}
	print := func(key string, t engine.Thresholds, nCheck int, unit string) {
		fmt.Printf("%s cutoffs from %d human %s (checked on %d held out):\n", key, t.CalibratedOn, unit, nCheck)
		fmt.Printf("  inconclusive >= %3d held-out human at or above: %.1f%%\n", t.Mixed, 100*t.HumanAtMixed)
		fmt.Printf("  likely AI    >= %3d held-out human at or above: %.1f%%\n", t.LikelyAI, 100*t.HumanAtLikelyAI)
		fmt.Printf("  AI           >= %3d held-out human at or above: %.1f%%\n", t.AI, 100*t.HumanAtAI)
	}
	if *only != "segments" {
		print(key, t, len(check), "texts")
	}
	if len(segCalib) > 0 && *only != "documents" {
		// Segments are matched to their own documents: at each level, a
		// human segment should reach the cutoff about as often as the
		// documents it came from reach the document cutoff. (A fixed 2%
		// would overshoot: memos supply much of the document tail but are
		// too short to have segments.)
		doc := file[key]
		floor := func(v float64) float64 { return max(v, 0.0025) }
		st := fit(segCalib, segCheck, floor(share(segDocRaw, doc.Mixed)), floor(share(segDocRaw, doc.LikelyAI)), floor(share(segDocRaw, doc.AI)))
		file[key+".segment"] = st
		print(key+".segment", st, len(segCheck), "segments")
	}
	return writeJSON(*out, file)
}

// runSetup registers aidetector-mcp with agent clients and installs the
// ai-text-detection skill for each.
func runSetup(args []string) error {
	fs := flag.NewFlagSet("setup", flag.ExitOnError)
	client := fs.String("client", "all", "all, claude, codex, or antigravity (comma-separated)")
	mcpPath := fs.String("mcp", "", "path to aidetector-mcp (default: next to this binary)")
	dryRun := fs.Bool("dry-run", false, "print what would change without changing anything")
	skillOnly := fs.Bool("skill-only", false, "install the skill without registering the MCP server")
	scorer := fs.String("scorer", "", "optional token-probability scorer URL to pass to the server")
	fs.Parse(args)

	o := setup.Options{Skill: true, Server: !*skillOnly, DryRun: *dryRun, Out: os.Stdout}
	if o.Server {
		p := *mcpPath
		if p == "" {
			var err error
			if p, err = setup.DefaultMCPPath(); err != nil {
				return err
			}
		}
		abs, err := filepath.Abs(p)
		if err != nil {
			return err
		}
		o.MCPPath = abs
		if *scorer != "" {
			o.Env = []string{"AIDETECTOR_SCORER_URL=" + *scorer}
		}
	}
	var clients []setup.Client
	if *client == "all" {
		clients = setup.Clients
	} else {
		for _, name := range strings.Split(*client, ",") {
			c, ok := setup.Find(strings.TrimSpace(name))
			if !ok {
				return fmt.Errorf("unknown client %q (want claude, codex, or antigravity)", name)
			}
			clients = append(clients, c)
		}
	}
	for _, c := range clients {
		if err := setup.Install(c, o); err != nil {
			return err
		}
	}
	if !*dryRun {
		fmt.Println("Restart your agent clients to pick up the changes.")
	}
	return nil
}

// runFixtures scores the regression fixtures. Known-human fixtures fail the
// check if more than one scores 60+ or any segment scores 80+; monitoring
// fixtures are reported only.
func runFixtures(args []string) error {
	fs := flag.NewFlagSet("fixtures", flag.ExitOnError)
	list := fs.String("list", "internal/engine/testdata/fixtures.json", "fixture list")
	dir := fs.String("dir", "internal/engine/testdata/fixtures", "cache directory (gitignored)")
	fetchOnly := fs.Bool("fetch-only", false, "download missing fixtures without scoring")
	fs.Parse(args)
	l, err := fixtures.Load(*list)
	if err != nil {
		return err
	}
	all := append(append([]fixtures.Fixture{}, l.KnownHuman...), l.Monitoring...)
	if err := fixtures.Fetch(context.Background(), *dir, all, os.Stderr); err != nil {
		return err
	}
	if *fetchOnly {
		return nil
	}
	e := engine.Default()
	score := func(path string) (engine.Result, bool) {
		b, err := os.ReadFile(path)
		if err != nil {
			return engine.Result{}, false
		}
		r, err := e.Analyze(context.Background(), detect.Input{Text: string(b), Kind: detect.KindProse})
		return r, err == nil
	}
	maxSeg := func(r engine.Result) int {
		m := -1
		for _, s := range r.Segments {
			m = max(m, s.Score)
		}
		return m
	}
	row := func(f fixtures.Fixture) (doc, seg int, ok bool) {
		txt, md := fixtures.Paths(*dir, f)
		r, ok := score(txt)
		if !ok {
			fmt.Printf("  %-32s missing (run with network access)\n", f.ID)
			return 0, 0, false
		}
		segs := "no segments"
		if len(r.Segments) > 0 {
			segs = fmt.Sprintf("max segment %d of %d", maxSeg(r), len(r.Segments))
		}
		extra := ""
		if rm, ok := score(md); ok {
			extra = fmt.Sprintf("  | markdown %d", rm.Score)
		}
		if f.Reported > 0 {
			extra += fmt.Sprintf("  | reported %d", f.Reported)
		}
		fmt.Printf("  %-32s %3d %-30s %s%s\n", f.ID, r.Score, r.Label, segs, extra)
		return r.Score, maxSeg(r), true
	}
	fmt.Println("Known human (paste-style text):")
	over60, segFails := 0, 0
	for _, f := range l.KnownHuman {
		doc, seg, ok := row(f)
		if !ok {
			continue
		}
		if doc >= 60 {
			over60++
		}
		if seg >= 80 {
			segFails++
		}
	}
	fmt.Println("Monitoring (report only; never train or calibrate on these):")
	for _, f := range l.Monitoring {
		row(f)
	}
	if over60 > 1 || segFails > 0 {
		return fmt.Errorf("fixture check failed: %d known-human documents at 60+ (max 1), %d with a segment at 80+ (max 0)", over60, segFails)
	}
	fmt.Println("fixture check passed")
	return nil
}

// fold assigns a sample to one of n cross-validation folds by id, so a human
// text and its AI mirror always land in the same fold.
func fold(id string, n int) int {
	return int(split.Hash(id) / 5 % uint32(n))
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func arrow(direction string) string {
	switch direction {
	case "AI":
		return "→ AI"
	case "human":
		return "→ human"
	case "info":
		return "ℹ info"
	}
	return "⚠ warning"
}

func fmtNum(v float64) string {
	if math.Abs(v) >= 10 || v == math.Trunc(v) {
		return fmt.Sprintf("%.0f", v)
	}
	return fmt.Sprintf("%.2f", v)
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func truncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n-1]) + "…"
	}
	return s
}
