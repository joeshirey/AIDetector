// Command aidetector scores text for AI authorship from the command line.
//
//	aidetector [detect] [-kind K] [-json] [file|-]
//	aidetector commits [-repo DIR] [-n N] [range]
//	aidetector eval [-kind K] [-split train|test] [-heuristic] [-json] DIR...
//	aidetector fit  -kind K DIR...
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
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
		case "detect", "commits", "eval", "fit":
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
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "SIGNAL\tAI PROB\tWEIGHT\tDETAIL")
	for _, s := range res.Signals {
		fmt.Fprintf(tw, "%s\t%.2f\t%.1f\t%s\n", s.Name, s.AIProb, s.Weight, s.Detail)
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
