// Command corpus builds the labeled human/AI corpus used to evaluate and
// calibrate the detector.
//
//	corpus harvest [-kind all|commit|pr|blog|memo] [-n N]
//	corpus mirror  [-kind ...] [-gens provider:model,...] [-j 8] [-limit N]
//	corpus renormalize [-kind ...]
//	corpus stats
//	corpus ping    [-gens ...]
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/joeshirey/AIDetector/internal/corpus"
)

var kinds = []string{"commit", "pr", "blog", "memo"}

var defaultRepos = []string{
	"golang/go", "rails/rails", "django/django", "hashicorp/terraform", "prometheus/prometheus",
	"rust-lang/cargo", "BurntSushi/ripgrep", "junegunn/fzf", "home-assistant/core", "grafana/grafana",
	"facebook/react", "vercel/next.js", "sveltejs/svelte", "pallets/flask", "docker/compose",
	"apache/airflow", "microsoft/vscode", "neovim/neovim", "tokio-rs/tokio", "kubernetes/kubernetes",
}

const defaultGens = "claude:claude-opus-5-5,claude:claude-sonnet-5-5,claude:claude-haiku-5-5," +
	"gemini:gemini-3.8-flash,gemini:gemini-3.1-pro-preview"

func main() {
	log.SetFlags(log.Ltime)
	if len(os.Args) < 2 {
		usage()
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	root := fs.String("out", "corpus", "corpus directory")
	kind := fs.String("kind", "all", "all, commit, pr, blog, or memo")
	n := fs.Int("n", 0, "harvest: samples per kind (default: commit 400, pr 240, blog 250, memo 250)")
	repos := fs.String("repos", strings.Join(defaultRepos, ","), "harvest: GitHub repos for commits and PRs")
	gens := fs.String("gens", defaultGens, "mirror/ping: generators as provider:model, comma-separated")
	workers := fs.Int("j", 8, "mirror: concurrent requests")
	limit := fs.Int("limit", 0, "mirror: max new mirrors per kind this run (0 = all)")
	fs.Parse(os.Args[2:])

	store := &corpus.Store{Root: *root}
	selected := kinds
	if *kind != "all" {
		selected = strings.Split(*kind, ",")
	}

	var err error
	switch os.Args[1] {
	case "harvest":
		err = harvest(ctx, store, selected, *n, strings.Split(*repos, ","))
	case "mirror":
		var gs []corpus.Generator
		if gs, err = corpus.NewGenerators(ctx, *gens); err == nil {
			for _, k := range selected {
				if err = store.Mirror(ctx, corpus.MirrorOptions{Kind: k, Generators: gs, Workers: *workers, Limit: *limit}); err != nil {
					break
				}
			}
		}
	case "stats":
		err = stats(*root)
	case "renormalize":
		for _, k := range selected {
			n, rerr := store.Renormalize(k)
			if rerr != nil {
				err = rerr
				break
			}
			log.Printf("%s: renormalized %d files", k, n)
		}
	case "ping":
		err = runPing(ctx, *gens)
	default:
		usage()
	}
	if err != nil {
		log.Fatal(err)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: corpus harvest|mirror|renormalize|stats|ping [flags]  (corpus <cmd> -h for flags)")
	os.Exit(2)
}

func harvest(ctx context.Context, store *corpus.Store, kinds []string, n int, repos []string) error {
	defaults := map[string]int{"commit": 400, "pr": 240, "blog": 250, "memo": 250}
	for _, k := range kinds {
		want := n
		if want == 0 {
			want = defaults[k]
		}
		seen := func(id string) bool { return store.Has(k, id) }
		var samples []corpus.Sample
		var err error
		switch k {
		case "commit", "pr":
			per := (want + len(repos) - 1) / len(repos)
			for _, repo := range repos {
				var got []corpus.Sample
				if k == "commit" {
					got, err = corpus.HarvestCommits(ctx, repo, corpus.Cutoff, per, seen)
				} else {
					got, err = corpus.HarvestPRs(ctx, repo, corpus.Cutoff, per, seen)
				}
				if err != nil {
					log.Printf("%s %s: %v", k, repo, err) // keep going with other repos
				}
				log.Printf("%s %s: %d", k, repo, len(got))
				if err := save(store, got); err != nil {
					return err
				}
				samples = append(samples, got...)
			}
		case "blog":
			if samples, err = corpus.HarvestMedium(ctx, corpus.Cutoff, want, seen); err == nil {
				err = save(store, samples)
			}
		case "memo":
			if samples, err = corpus.HarvestEnron(ctx, want, seen); err == nil {
				err = save(store, samples)
			}
		default:
			return fmt.Errorf("unknown kind %q", k)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", k, err)
		}
		log.Printf("%s: harvested %d new samples", k, len(samples))
	}
	return nil
}

func save(store *corpus.Store, samples []corpus.Sample) error {
	for _, s := range samples {
		if err := store.SaveHuman(s); err != nil {
			return err
		}
	}
	return nil
}

func stats(root string) error {
	fmt.Printf("%-8s %7s  %s\n", "KIND", "HUMAN", "AI BY GENERATOR")
	for _, k := range kinds {
		human, _ := filepath.Glob(filepath.Join(root, k, "human", "*.txt"))
		gens, _ := filepath.Glob(filepath.Join(root, k, "ai", "*"))
		var parts []string
		for _, g := range gens {
			files, _ := filepath.Glob(filepath.Join(g, "*.txt"))
			parts = append(parts, fmt.Sprintf("%s=%d", filepath.Base(g), len(files)))
		}
		fmt.Printf("%-8s %7d  %s\n", k, len(human), strings.Join(parts, " "))
	}
	return nil
}
