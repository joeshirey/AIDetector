package corpus

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/joeshirey/AIDetector/internal/split"
	"github.com/joeshirey/AIDetector/internal/text"
)

// Prompt templates, phrased the varied ways people actually ask. {placeholders}
// are filled from the sample's context.
//
// These must ask for the artifact plainly. Instructions to sound human, avoid
// AI tells, or "humanize" would make the AI side unrepresentative of normal
// use; adversarial variants belong in a separate set that is never trained on.
// checkPrompts enforces this.
var writePrompts = map[string][]string{
	"pr": {
		"Write a pull request description for this change.\n\nTitle: {title}\n\nDiff:\n{diff}\n\nOutput only the PR description in Markdown (no title).",
		"Here's the diff for a PR titled \"{title}\" in {repo}. Write the PR body I should use. Only output the body.\n\n{diff}",
		"Draft a GitHub pull request description for the changes below. Explain what changed and why, and how it was tested if that's apparent.\n\nTitle: {title}\n\n{diff}\n\nReturn just the description.",
	},
	"blog": {
		"Write a blog post titled \"{title}\". Aim for about {words} words.\n\nBrief:\n{brief}\n\nOutput only the post.",
		"I need a Medium article for my blog. Title: \"{title}\". Tags: {tags}. About {words} words. Here's what it should cover:\n{brief}\n\nJust give me the article text.",
		"Write a ~{words}-word post for a tech/business blog based on this outline. Title: {title}\n\n{brief}\n\nReturn the post only, no commentary.",
	},
	"essay": {
		"Write an essay of about {words} words responding to this prompt.\n\nBrief:\n{brief}\n\nOutput only the essay.",
		"I need to write an essay for class. Here's what it should argue:\n{brief}\n\nAbout {words} words. Just give me the essay.",
		"Write a persuasive essay of about {words} words based on these notes.\n\n{brief}\n\nReturn only the essay.",
	},
	"memo": {
		"Write a work email. Subject: {subject}. About {words} words.\n\n{brief}\n\nOutput only the email body.",
		"Draft an internal email for me based on these notes. Subject: {subject}. Keep it to roughly {words} words.\n\n{brief}\n\nJust the body text please.",
	},
}

// Prose mirrors are written from a brief rather than the original, so the AI
// text covers the same content without copying its wording.
var briefPrompts = map[string]string{
	"blog":  "Read this blog post and write a short writing brief for it: the topic, the intended audience, the main points in order (terse bullets), and the tone. Don't quote the post.\n\n<post>\n{text}\n</post>",
	"essay": "Read this student essay and write a brief for it: the question or prompt it responds to, the writer's position, and the main points in order. Don't quote it.\n\n<essay>\n{text}\n</essay>",
	"memo":  "Read this work email and write a brief for it: who it's addressed to (names and roles), its purpose, the key facts and requests in order, and the tone. Don't quote it.\n\n<email>\n{text}\n</email>",
}

var evasionRE = regexp.MustCompile(`(?i)humaniz|sounds? (?:more )?human|like a human|natural-sounding|undetect|ai detect|avoid (?:ai|em dash|dashes)|signs of ai|ai tells|written by (?:a person|a human)`)

func checkPrompts() error {
	all := []string{}
	for _, ps := range writePrompts {
		all = append(all, ps...)
	}
	for _, p := range briefPrompts {
		all = append(all, p)
	}
	for _, p := range all {
		if m := evasionRE.FindString(p); m != "" {
			return fmt.Errorf("generation prompt contains detector-evasion wording %q; mirrors must reflect normal use", m)
		}
	}
	return nil
}

var placeholderRE = regexp.MustCompile(`\{(\w+)\}`)

func fill(tmpl string, vars map[string]string) string {
	return placeholderRE.ReplaceAllStringFunc(tmpl, func(m string) string {
		if v, ok := vars[m[1:len(m)-1]]; ok {
			return v
		}
		return m
	})
}

type MirrorOptions struct {
	Kind       string
	Generators []Generator
	Workers    int
	Limit      int  // max new mirrors this run; 0 = all
	Add        bool // add a mirror from the (single) generator to samples that lack one from it, even if they have others
}

// Mirror writes one AI counterpart for every human sample that lacks one.
// It is safe to re-run: existing mirrors are skipped.
func (s *Store) Mirror(ctx context.Context, o MirrorOptions) error {
	if err := checkPrompts(); err != nil {
		return err
	}
	sources, err := s.Sources(o.Kind)
	if err != nil {
		return err
	}

	// Hashed order, so a limited run samples evenly instead of taking the
	// ids (and repos) that sort first.
	slices.SortFunc(sources, func(a, b Sample) int { return cmp.Compare(hashN(a.ID, 1<<30), hashN(b.ID, 1<<30)) })
	var todo []Sample
	for _, sm := range sources {
		has := s.HasAI(o.Kind, sm.ID)
		if o.Add { // only mirrors from these generators count
			has = false
			for _, g := range o.Generators {
				if _, err := os.Stat(s.AIPath(o.Kind, g.Name(), sm.ID)); err == nil {
					has = true
				}
			}
		}
		if !has && (o.Limit == 0 || len(todo) < o.Limit) {
			todo = append(todo, sm)
		}
	}
	log.Printf("%s: %d sources, %d to mirror", o.Kind, len(sources), len(todo))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan Sample)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done, failed := 0, 0
	for range max(1, o.Workers) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for sm := range jobs {
				gen := o.Generators[hashN(sm.ID, len(o.Generators))]
				variant, err := s.mirrorOne(ctx, gen, sm)
				e := ManifestEntry{ID: sm.ID, Generator: gen.Name(), Variant: variant, Split: split.Of(sm.ID), At: time.Now().UTC()}
				mu.Lock()
				if errors.Is(err, ErrBudget) {
					log.Printf("stopping: %v", err)
					cancel() // no more paid calls; unsent jobs are dropped
					mu.Unlock()
					continue
				}
				if err != nil {
					failed++
					e.Error = err.Error()
					log.Printf("%s %s: %v", gen.Name(), sm.ID, err)
				} else {
					done++
				}
				if (done+failed)%25 == 0 {
					log.Printf("%s: %d/%d done, %d failed; %v", o.Kind, done, len(todo), failed, Spend)
				}
				mu.Unlock()
				s.Log(o.Kind, e)
			}
		}()
	}
	for _, sm := range todo {
		select {
		case jobs <- sm:
		case <-ctx.Done():
		}
	}
	close(jobs)
	wg.Wait()
	log.Printf("%s: finished, %d written, %d failed; %v", o.Kind, done, failed, Spend)
	if err := Spend.Check(""); err != nil {
		return err
	}
	return nil
}

func (s *Store) mirrorOne(ctx context.Context, gen Generator, sm Sample) (int, error) {
	templates := writePrompts[sm.Kind]
	variant := hashN(sm.ID+gen.Name(), len(templates))
	vars := map[string]string{"text": sm.Text}
	for k, v := range sm.Context {
		vars[k] = v
	}
	if bp, ok := briefPrompts[sm.Kind]; ok {
		brief, err := gen.Generate(ctx, fill(bp, vars))
		if err != nil {
			return variant, fmt.Errorf("brief: %w", err)
		}
		vars["brief"] = strings.TrimSpace(brief)
	}
	out, err := gen.Generate(ctx, fill(templates[variant], vars))
	if err != nil {
		return variant, err
	}
	out = clean(sm.Kind, out, sm.Context["title"])
	if len(strings.Fields(out)) < 2 {
		return variant, errors.New("empty output")
	}
	return variant, writeFile(s.AIPath(sm.Kind, gen.Name(), sm.ID), out)
}

var (
	fenceWrapRE = regexp.MustCompile("(?s)^```[a-z]*\n(.*?)\n```$")
	preambleRE  = regexp.MustCompile(`(?i)^(here'?s|here is|sure|certainly)[^\n]*:\s*\n`)
	subjectRE   = regexp.MustCompile(`(?i)^subject:[^\n]*\n+`)
)

// clean strips wrapper text that isn't part of the artifact itself: code
// fences around the whole answer, "Here's your PR description:" preambles, a
// leading Subject: line on emails, and a repeated title heading on posts.
func clean(kind, out, title string) string {
	out = strings.TrimSpace(out)
	out = strings.TrimSpace(preambleRE.ReplaceAllString(out, ""))
	if m := fenceWrapRE.FindStringSubmatch(out); m != nil {
		out = strings.TrimSpace(m[1])
	}
	switch kind {
	case "pr":
		out = text.Normalize(kind, NormalizePR(out))
	case "memo":
		out = text.Normalize(kind, stripMarkdown(subjectRE.ReplaceAllString(out, "")))
	case "blog", "essay":
		// Human texts don't include their title, so drop a leading heading.
		if first, rest, ok := strings.Cut(out, "\n"); ok && strings.HasPrefix(first, "#") {
			out = rest
		}
		out = stripMarkdown(out)
	}
	return out
}

var (
	mdHeadingRE  = regexp.MustCompile(`(?m)^#{1,6}\s+`)
	mdEmphasisRE = regexp.MustCompile(`\*\*([^*\n]+)\*\*|__([^_\n]+)__|\*([^*\n]+)\*`)
	mdBulletRE   = regexp.MustCompile(`(?m)^[ \t]*[-*+]\s+`)
	mdRuleRE     = regexp.MustCompile(`(?m)^[ \t]*(-{3,}|\*{3,})\s*$\n?`)
)

// stripMarkdown renders Markdown to plain text the way the human prose
// sources (scraped Medium posts, AESLC emails) appear, so the detector learns
// from wording rather than from formatting the human side never has.
func stripMarkdown(s string) string {
	s = mdRuleRE.ReplaceAllString(s, "")
	s = mdHeadingRE.ReplaceAllString(s, "")
	s = mdBulletRE.ReplaceAllString(s, "")
	s = mdEmphasisRE.ReplaceAllString(s, "$1$2$3")
	return strings.TrimSpace(s)
}

// CheckPrompt returns an error if a generation prompt asks for evasion
// (sounding human, avoiding AI tells). Training text must reflect normal use.
func CheckPrompt(p string) error {
	if m := evasionRE.FindString(p); m != "" {
		return fmt.Errorf("generation prompt contains detector-evasion wording %q; mirrors must reflect normal use", m)
	}
	return nil
}

// Clean strips wrapper text from a generated artifact (see clean).
func Clean(kind, out, title string) string { return clean(kind, out, title) }

// Fill substitutes {placeholders} in a prompt template.
func Fill(tmpl string, vars map[string]string) string { return fill(tmpl, vars) }

// BriefPrompt is the brief-writing prompt for a prose kind (blog, memo,
// essay), or "" if the kind doesn't use briefs.
func BriefPrompt(kind string) string { return briefPrompts[kind] }
