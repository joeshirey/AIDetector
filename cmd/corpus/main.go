// Command corpus builds the labeled human/AI corpus used to evaluate and
// calibrate the detector.
//
//	corpus harvest [-kind all|commit|pr|blog|memo|essay] [-n N] [-pool]
//	corpus promote [-kind ...] -n N [-hard 0.5]
//	corpus mirror  [-kind ...] [-gens provider:model,...] [-j 8] [-limit N]
//	corpus variant -type polished|humanized [-kind ...] [-n 60]
//	corpus renormalize [-kind ...]
//	corpus stats
//	corpus ping    [-gens ...]
package main

import (
	"cmp"
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
)

var kinds = []string{"commit", "pr", "blog", "memo", "essay"}

var defaultRepos = []string{
	"golang/go", "rails/rails", "django/django", "hashicorp/terraform", "prometheus/prometheus",
	"rust-lang/cargo", "BurntSushi/ripgrep", "junegunn/fzf", "home-assistant/core", "grafana/grafana",
	"facebook/react", "vercel/next.js", "sveltejs/svelte", "pallets/flask", "docker/compose",
	"apache/airflow", "microsoft/vscode", "neovim/neovim", "tokio-rs/tokio", "kubernetes/kubernetes",
	"pytorch/pytorch", "ansible/ansible", "elastic/elasticsearch", "rust-lang/rust", "nodejs/node",
	"laravel/framework", "spring-projects/spring-boot", "godotengine/godot", "mastodon/mastodon", "denoland/deno",
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
	kind := fs.String("kind", "all", "all, or a comma-separated list of commit, pr, blog, memo, essay")
	n := fs.Int("n", 0, "harvest/promote/variant: samples per kind")
	pool := fs.Bool("pool", false, "harvest: collect into the human-only reserve pool")
	hard := fs.Float64("hard", 0.5, "promote: share of promotions taken from the highest-scoring (hard-negative) pool texts")
	variant := fs.String("type", "", "variant: polished or humanized")
	exclude := fs.String("exclude", "", "harvest: skip samples already in this other corpus directory (for fresh, unbiased pools)")
	phase := fs.Int("phase", 3, "harvest -pool: which every-7th Medium row to sample (0 is the training harvest)")
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
		var other *corpus.Store
		if *exclude != "" {
			other = &corpus.Store{Root: *exclude}
		}
		err = harvest(ctx, store, other, selected, *n, *pool, *phase, strings.Split(*repos, ","))
	case "promote":
		err = promote(ctx, store, selected, *n, *hard)
	case "variant":
		err = variants(ctx, store, selected, *variant, *gens, *workers, max(*n, 1))
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
	fmt.Fprintln(os.Stderr, "usage: corpus harvest|promote|mirror|variant|renormalize|stats|ping [flags]  (corpus <cmd> -h for flags)")
	os.Exit(2)
}

func harvest(ctx context.Context, store, exclude *corpus.Store, kinds []string, n int, pool bool, mediumPhase int, repos []string) error {
	defaults := map[string]int{"commit": 400, "pr": 240, "blog": 250, "memo": 250, "essay": 0}
	if pool {
		defaults = map[string]int{"commit": 1500, "pr": 600, "blog": 1500, "memo": 1500, "essay": 2000}
	}
	save := func(samples []corpus.Sample) error {
		for _, s := range samples {
			var err error
			if pool {
				err = store.SavePool(s)
			} else {
				err = store.SaveHuman(s)
			}
			if err != nil {
				return err
			}
		}
		return nil
	}
	for _, k := range kinds {
		want := n
		if want == 0 {
			want = defaults[k]
		}
		seen := func(id string) bool { return store.Has(k, id) || (exclude != nil && exclude.Has(k, id)) }
		var samples []corpus.Sample
		var err error
		switch k {
		case "commit", "pr":
			per := (want + len(repos) - 1) / len(repos)
			for _, repo := range repos {
				var got []corpus.Sample
				if k == "commit" {
					got, err = corpus.HarvestCommits(ctx, repo, corpus.Cutoff, per, pool, seen)
				} else {
					got, err = corpus.HarvestPRs(ctx, repo, corpus.Cutoff, per, pool, seen)
				}
				if err != nil {
					log.Printf("%s %s: %v", k, repo, err) // keep going with other repos
				}
				log.Printf("%s %s: %d", k, repo, len(got))
				if err := save(got); err != nil {
					return err
				}
				samples = append(samples, got...)
			}
		case "blog":
			phase, medium := 0, want
			if pool {
				// Most of the blog pool from Medium, plus every eligible Go blog post.
				var gob []corpus.Sample
				if gob, err = corpus.HarvestGoBlog(ctx, corpus.Cutoff, want, seen); err != nil {
					return fmt.Errorf("goblog: %w", err)
				}
				log.Printf("blog goblog: %d", len(gob))
				if err = save(gob); err != nil {
					return err
				}
				samples = append(samples, gob...)
				phase, medium = mediumPhase, max(0, want-len(gob))
			}
			var med []corpus.Sample
			if med, err = corpus.HarvestMedium(ctx, corpus.Cutoff, medium, phase, seen); err == nil {
				err = save(med)
				samples = append(samples, med...)
			}
		case "memo":
			if samples, err = corpus.HarvestEnron(ctx, want, seen); err == nil {
				err = save(samples)
			}
		case "essay":
			if samples, err = corpus.HarvestLearnerEssays(ctx, want, seen); err == nil {
				err = save(samples)
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

// promote moves n pool texts per kind into the training corpus. A `hard`
// share are the human texts the current model scores as most AI-like (hard
// negatives, as in active-learning detector training); the rest are random so the
// corpus stays representative.
func promote(ctx context.Context, store *corpus.Store, kinds []string, n int, hard float64) error {
	e := engine.Default()
	for _, k := range kinds {
		pool, err := store.Pool(k)
		if err != nil {
			return err
		}
		if len(pool) == 0 || n == 0 {
			continue
		}
		type scored struct {
			s     corpus.Sample
			score int
		}
		all := make([]scored, len(pool))
		for i, s := range pool {
			res, err := e.Analyze(ctx, detect.Input{Text: s.Text, Kind: detectKind(k)})
			if err != nil {
				return err
			}
			all[i] = scored{s, res.Score}
		}
		slices.SortFunc(all, func(a, b scored) int { return cmp.Compare(b.score, a.score) })
		nHard := int(float64(n) * hard)
		picked := slices.Clone(all[:min(nHard, len(all))])
		rest := all[len(picked):]
		rng := rand.New(rand.NewPCG(7, 7))
		rng.Shuffle(len(rest), func(i, j int) { rest[i], rest[j] = rest[j], rest[i] })
		picked = append(picked, rest[:min(n-len(picked), len(rest))]...)

		flagged := 0
		for _, a := range all {
			if a.score >= 50 {
				flagged++
			}
		}
		promoted := 0
		for _, p := range picked {
			if err := corpus.FetchDiff(ctx, &p.s); err != nil {
				log.Printf("%s %s: %v (skipped)", k, p.s.ID, err)
				continue
			}
			if err := store.Promote(p.s); err != nil {
				return err
			}
			promoted++
		}
		log.Printf("%s: pool %d (%d scored 50+, hardest %d), promoted %d (%d hard negatives)",
			k, len(all), flagged, all[0].score, promoted, min(nHard, promoted))
	}
	return nil
}

func detectKind(k string) detect.Kind {
	switch k {
	case "commit":
		return detect.KindCommit
	case "pr":
		return detect.KindPR
	}
	return detect.KindProse
}

func variants(ctx context.Context, store *corpus.Store, kinds []string, typ, gens string, workers, n int) error {
	gs, err := corpus.NewGenerators(ctx, gens)
	if err != nil {
		return err
	}
	o := corpus.VariantOptions{Type: typ, Generators: gs, Workers: workers, N: n}
	if typ == corpus.VariantHumanized {
		if o.Humanizer, err = humanizerSkill(ctx, store.Root); err != nil {
			return err
		}
	}
	for _, k := range kinds {
		o.Kind = k
		if err := store.Variants(ctx, o); err != nil {
			return err
		}
	}
	return nil
}

// humanizerSkill fetches github.com/blader/humanizer's SKILL.md (MIT) once
// and caches it in the corpus directory.
func humanizerSkill(ctx context.Context, root string) (string, error) {
	path := filepath.Join(root, "_cache", "humanizer-SKILL.md")
	if b, err := os.ReadFile(path); err == nil {
		return string(b), nil
	}
	out, err := exec.CommandContext(ctx, "gh", "api", "repos/blader/humanizer/contents/SKILL.md", "-H", "Accept: application/vnd.github.raw").Output()
	if err != nil {
		return "", fmt.Errorf("fetching humanizer skill: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	return string(out), os.WriteFile(path, out, 0o644)
}

func stats(root string) error {
	fmt.Printf("%-8s %6s %6s  %s\n", "KIND", "POOL", "HUMAN", "AI BY GENERATOR")
	for _, k := range kinds {
		human, _ := filepath.Glob(filepath.Join(root, k, "human", "*.txt"))
		pool, _ := filepath.Glob(filepath.Join(root, k, "pool", "*.txt"))
		gens, _ := filepath.Glob(filepath.Join(root, k, "ai", "*"))
		var parts []string
		for _, g := range gens {
			files, _ := filepath.Glob(filepath.Join(g, "*.txt"))
			parts = append(parts, fmt.Sprintf("%s=%d", filepath.Base(g), len(files)))
		}
		fmt.Printf("%-8s %6d %6d  %s\n", k, len(pool), len(human), strings.Join(parts, " "))
	}
	return nil
}
