// Command aidetector scores text for AI authorship from the command line.
//
//	aidetector [detect] [-kind K] [-json] [file|-]
//	aidetector commits [-repo DIR] [-n N] [range]
//	aidetector eval [-kind K] [-split train|test] [-heuristic] [-json] DIR...
//	aidetector fit  -kind K DIR...
//	aidetector dist [-kind K] DIR...   score distribution of a folder (e.g. the human pool)
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
	"github.com/joeshirey/AIDetector/internal/mcpserver"
	"github.com/joeshirey/AIDetector/internal/split"
)

func main() {
	args := os.Args[1:]
	cmd := "detect"
	if len(args) > 0 {
		switch args[0] {
		case "detect", "commits", "eval", "fit", "dist":
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
	case "commits":
		err = runCommits(args)
	case "eval":
		err = runEval(args)
	case "fit":
		err = runFit(args)
	case "dist":
		err = runDist(args)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "aidetector:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage:
  aidetector [detect] [-kind auto|prose|commit|pr] [-json] [file|-]
  aidetector commits [-repo DIR] [-n N] [-json] [range]
  aidetector eval [-kind K] [-split train|test] [-heuristic] [-json] DIR...   (DIR/human/*, DIR/ai/*)
  aidetector fit -kind prose|commit|pr DIR...   (writes internal/engine/models/KIND.json)
  aidetector dist [-kind K] DIR...   score distribution for every .txt under DIR (pool false-positive rate, variant sets)
`)
}

func runDetect(args []string) error {
	fs := flag.NewFlagSet("detect", flag.ExitOnError)
	kind := fs.String("kind", "auto", "auto, prose, commit, or pr")
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
		fmt.Printf("\nSegments (%.0f%% scored 50+):\n", *res.AIFraction*100)
		for _, s := range res.Segments {
			fmt.Printf("  [%3d] %s\n", s.Score, s.Preview)
		}
	}
	for _, n := range res.Notes {
		fmt.Println("\nNote:", n)
	}
	return nil
}

func runCommits(args []string) error {
	fs := flag.NewFlagSet("commits", flag.ExitOnError)
	repo := fs.String("repo", ".", "git repository")
	n := fs.Int("n", 20, "max commits")
	asJSON := fs.Bool("json", false, "print JSON")
	fs.Parse(args)
	res, err := mcpserver.ScoreCommits(context.Background(), engine.Default(), *repo, fs.Arg(0), *n)
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(res)
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "SCORE\tSHA\tSUBJECT\tTOP SIGNAL")
	for _, c := range res.Commits {
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\n", c.Score, c.SHA[:min(8, len(c.SHA))], truncate(c.Subject, 60), c.TopSignal)
	}
	tw.Flush()
	fmt.Printf("\n%d commits, mean score %.0f, %d scored 60+\n", len(res.Commits), res.MeanScore, res.Flagged)
	return nil
}

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	kind := fs.String("kind", "auto", "auto, prose, commit, or pr")
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
	kind := fs.String("kind", "", "prose, commit, or pr")
	out := fs.String("out", "internal/engine/models", "output directory")
	fs.Parse(args)
	k := detect.ParseKind(*kind)
	if k == detect.KindAuto || fs.NArg() == 0 {
		usage()
		os.Exit(2)
	}
	// Features come from the heuristic engine so fitting never depends on a
	// previously fitted model.
	r, err := eval.Run(context.Background(), engine.Heuristic(), fs.Args(), k, "train")
	if err != nil {
		return err
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
	kind := fs.String("kind", "auto", "auto, prose, commit, or pr")
	worst := fs.Int("worst", 5, "list this many highest-scoring files")
	fs.Parse(args)
	if fs.NArg() == 0 {
		usage()
		os.Exit(2)
	}
	e := engine.Default()
	type scored struct {
		path  string
		score int
	}
	var all []scored
	for _, dir := range fs.Args() {
		paths, err := filepath.Glob(filepath.Join(dir, "*.txt"))
		if err != nil {
			return err
		}
		sub, _ := filepath.Glob(filepath.Join(dir, "*", "*.txt"))
		for _, p := range append(paths, sub...) {
			b, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			res, err := e.Analyze(context.Background(), detect.Input{Text: string(b), Kind: detect.ParseKind(*kind)})
			if err != nil {
				return err
			}
			all = append(all, scored{p, res.Score})
		}
	}
	if len(all) == 0 {
		return fmt.Errorf("no .txt files found")
	}
	slices.SortFunc(all, func(a, b scored) int { return b.score - a.score })
	at := func(t int) float64 {
		n := 0
		for _, s := range all {
			if s.score >= t {
				n++
			}
		}
		return 100 * float64(n) / float64(len(all))
	}
	fmt.Printf("%d texts  median %d  |  scored 50+: %.1f%%  70+: %.1f%%  80+: %.1f%%  90+: %.1f%%\n",
		len(all), all[len(all)/2].score, at(50), at(70), at(80), at(90))
	for _, s := range all[:min(*worst, len(all))] {
		fmt.Printf("  %3d  %s\n", s.score, s.path)
	}
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
