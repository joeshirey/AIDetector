package corpus

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/joeshirey/AIDetector/internal/split"
)

// Eval-only variant sets. They live outside human/ and ai/, so `fit` and
// `eval` never read them; they exist to measure how the detector handles
// AI-assisted and evasion-rewritten text.
//
//	<root>/<kind>/ai_polished/<model>/<id>.txt   human test text, polished by a model
//	<root>/<kind>/ai_humanized/<model>/<id>.txt  AI test mirror, rewritten with the humanizer skill
const (
	VariantPolished  = "polished"
	VariantHumanized = "humanized"
)

var polishPrompts = []string{
	"Polish this draft for clarity and flow. Keep my meaning. Return only the revised text.\n\n{text}",
	"Can you clean up the grammar and wording of this? Just return the edited version.\n\n{text}",
	"Edit this to read more professionally. Output only the edited text.\n\n{text}",
}

type VariantOptions struct {
	Kind       string
	Type       string // VariantPolished or VariantHumanized
	Generators []Generator
	Workers    int
	N          int    // samples to produce
	Humanizer  string // humanizer SKILL.md text, for VariantHumanized
}

type variantJob struct {
	id, text, from string
}

// Variants writes up to N eval-only variants from the test split.
func (s *Store) Variants(ctx context.Context, o VariantOptions) error {
	jobs, err := s.variantSources(o)
	if err != nil {
		return err
	}
	outDir := "ai_" + o.Type
	var todo []variantJob
	for _, j := range jobs {
		if m, _ := filepath.Glob(s.dir(o.Kind, outDir, "*", j.id+".txt")); len(m) == 0 && len(todo) < o.N {
			todo = append(todo, j)
		}
	}
	log.Printf("%s %s: %d to generate", o.Kind, o.Type, len(todo))

	ch := make(chan variantJob)
	var wg sync.WaitGroup
	for range max(1, o.Workers) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range ch {
				gen := o.Generators[hashN(j.id+o.Type, len(o.Generators))]
				var prompt string
				if o.Type == VariantHumanized {
					// Deliberately bypasses checkPrompts: this set exists to
					// test evasion and is never trained on.
					prompt = o.Humanizer + "\n\n---\n\nApply the skill above in embedded mode to the text below. Return only the final rewritten text.\n\n<text>\n" + j.text + "\n</text>"
				} else {
					prompt = fill(polishPrompts[hashN(j.id, len(polishPrompts))], map[string]string{"text": j.text})
				}
				out, err := gen.Generate(ctx, prompt)
				e := ManifestEntry{ID: j.id, Generator: gen.Name(), Split: "test", At: time.Now().UTC()}
				if err == nil {
					out = clean(o.Kind, out, "")
					if len(strings.Fields(out)) < 2 {
						err = fmt.Errorf("empty output")
					}
				}
				if err == nil {
					err = writeFile(s.dir(o.Kind, outDir, gen.Name(), j.id+".txt"), out)
				}
				if err != nil {
					e.Error = err.Error()
					log.Printf("%s %s %s: %v", o.Type, gen.Name(), j.id, err)
				}
				s.logTo(o.Kind, "variants.jsonl", struct {
					ManifestEntry
					Type string `json:"type"`
					From string `json:"from,omitempty"`
				}{e, o.Type, j.from})
			}
		}()
	}
	for _, j := range todo {
		select {
		case ch <- j:
		case <-ctx.Done():
		}
	}
	close(ch)
	wg.Wait()
	return ctx.Err()
}

func (s *Store) variantSources(o VariantOptions) ([]variantJob, error) {
	var jobs []variantJob
	switch o.Type {
	case VariantPolished:
		srcs, err := s.Sources(o.Kind)
		if err != nil {
			return nil, err
		}
		for _, sm := range srcs {
			if split.Of(sm.ID) == "test" {
				jobs = append(jobs, variantJob{id: sm.ID, text: sm.Text, from: "human"})
			}
		}
	case VariantHumanized:
		if o.Humanizer == "" {
			return nil, fmt.Errorf("humanized variants need the humanizer skill text")
		}
		paths, _ := filepath.Glob(s.dir(o.Kind, "ai", "*", "*.txt"))
		for _, p := range paths {
			id := strings.TrimSuffix(filepath.Base(p), ".txt")
			if split.Of(id) != "test" {
				continue
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return nil, err
			}
			jobs = append(jobs, variantJob{id: id, text: strings.TrimSpace(string(b)), from: filepath.Base(filepath.Dir(p))})
		}
	default:
		return nil, fmt.Errorf("unknown variant type %q", o.Type)
	}
	slices.SortFunc(jobs, func(a, b variantJob) int { return strings.Compare(a.id, b.id) })
	return jobs, nil
}
