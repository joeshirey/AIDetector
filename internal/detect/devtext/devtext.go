// Package devtext looks for tells specific to commit messages and pull
// request descriptions, which are usually too short for statistics alone.
package devtext

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/text"
)

const name = "devtext"

type Detector struct{}

func New() *Detector { return &Detector{} }

func (*Detector) Name() string { return name }

var (
	// Tools that sign their work. Checked on every kind, since people paste PR
	// bodies and commit messages into other documents too.
	aiTrailerRE = regexp.MustCompile(`(?mi)^co-authored-by:.*\b(claude|copilot|chatgpt|openai|gpt-?\d|gemini|cursor|devin|codex|aider|noreply@anthropic\.com)\b.*` +
		`|generated (with|by) \[?(claude|copilot|chatgpt|cursor|codex|gemini)` +
		`|🤖 generated with`)

	prSectionRE   = regexp.MustCompile(`(?mi)^#{1,3}\s*(summary|changes|what changed|test plan|testing|motivation|key changes|overview)\b`)
	checkboxRE    = regexp.MustCompile(`(?m)^[ \t]*[-*]\s+\[[ xX]\]\s`)
	bulletRE      = regexp.MustCompile(`(?m)^[ \t]*[-*]\s+(.+)$`)
	emojiHeadRE   = regexp.MustCompile(`(?m)^#{1,3}\s*[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]`)
	imperativeRE  = regexp.MustCompile(`^(Add|Update|Remove|Refactor|Fix|Improve|Enhance|Implement|Introduce|Ensure|Replace|Rename|Move|Extract|Simplify|Clean|Support|Handle|Use|Make|Allow|Prevent|Avoid|Bump|Document|Adjust|Consolidate|Streamline)\b`)
	lowerSubjRE   = regexp.MustCompile(`^[a-z]`)
	wipSubjectRE  = regexp.MustCompile(`(?i)^(wip|fixup!|squash!|oops|typo|tmp|temp|asdf|more|stuff|fix|fixes|update|changes)\b\W*$`)
	conventionRE  = regexp.MustCompile(`^(feat|fix|chore|docs|refactor|test|perf|build|ci|style|revert)(\([^)]*\))?!?:\s`)
	trailingDotRE = regexp.MustCompile(`\.$`)
)

func (*Detector) Detect(_ context.Context, in detect.Input) ([]detect.Signal, error) {
	var out []detect.Signal
	if m := aiTrailerRE.FindString(in.Text); m != "" {
		out = append(out, detect.Signal{
			Name: "ai_attribution", Detector: name, Value: 1, AIProb: 0.99, Weight: 3, Decisive: true,
			Detail: fmt.Sprintf("explicit AI tool attribution: %q", strings.TrimSpace(m)),
		})
	}
	switch in.Kind {
	case detect.KindPR:
		out = append(out, prSignals(in.Text)...)
	case detect.KindCommit:
		out = append(out, commitSignals(in.Text)...)
	}
	for i := range out {
		out[i].Detector = name
	}
	return out, nil
}

func prSignals(t string) []detect.Signal {
	var out []detect.Signal
	sections := len(prSectionRE.FindAllString(t, -1))
	checks := len(checkboxRE.FindAllString(t, -1))
	lower := strings.ToLower(t)
	// "## Summary" + "## Test plan" with a checklist is the house style of
	// several coding agents. Human PRs following a repo template look similar,
	// so this is evidence, not proof.
	switch {
	case strings.Contains(lower, "summary") && strings.Contains(lower, "test plan") && checks > 0:
		out = append(out, detect.Signal{Name: "agent_pr_template", Value: 1, AIProb: 0.85, Weight: 1.2,
			Detail: "Summary + Test plan sections with a checklist (common coding-agent template)"})
	case sections >= 2:
		out = append(out, detect.Signal{Name: "agent_pr_template", Value: float64(sections), AIProb: 0.65, Weight: 0.6,
			Detail: fmt.Sprintf("%d templated section headings", sections)})
	}
	if emojiHeadRE.MatchString(t) {
		out = append(out, detect.Signal{Name: "emoji_headings", Value: 1, AIProb: 0.8, Weight: 0.7,
			Detail: "section headings start with emoji"})
	}
	if s := bulletUniformity(t); s != nil {
		out = append(out, *s)
	}
	return out
}

func commitSignals(t string) []detect.Signal {
	t = strings.TrimSpace(t)
	subject, body, _ := strings.Cut(t, "\n")
	subject = strings.TrimSpace(subject)
	words := len(text.Words(t))
	var out []detect.Signal

	// Terse or sloppy subjects are a human habit. Agents always write a full,
	// tidy subject.
	switch {
	case wipSubjectRE.MatchString(subject) || len(subject) < 12:
		out = append(out, detect.Signal{Name: "terse_subject", Value: float64(len(subject)), AIProb: 0.15, Weight: 1.0,
			Detail: fmt.Sprintf("terse subject %q", subject)})
	case lowerSubjRE.MatchString(subject) && !conventionRE.MatchString(subject),
		trailingDotRE.MatchString(subject) || len(subject) > 72:
		out = append(out, detect.Signal{Name: "loose_subject", Value: float64(len(subject)), AIProb: 0.35, Weight: 0.5,
			Detail: "subject breaks common conventions (case, trailing period, or length)"})
	}

	body = strings.TrimSpace(aiTrailerRE.ReplaceAllString(body, ""))
	if body == "" {
		if words <= 12 {
			out = append(out, detect.Signal{Name: "single_line", Value: float64(words), AIProb: 0.3, Weight: 0.4,
				Detail: "short single-line commit"})
		}
		return out
	}

	// Agents like a body made of "- Verb the thing" bullets.
	bullets := bulletRE.FindAllStringSubmatch(body, -1)
	if len(bullets) >= 3 {
		imperative := 0
		for _, b := range bullets {
			if imperativeRE.MatchString(strings.TrimSpace(b[1])) {
				imperative++
			}
		}
		ratio := float64(imperative) / float64(len(bullets))
		p := 0.5 + 0.35*ratio
		out = append(out, detect.Signal{Name: "bulleted_change_list", Value: ratio, AIProb: p, Weight: 0.8,
			Detail: fmt.Sprintf("body is a %d-item change list, %d starting with a stock verb", len(bullets), imperative)})
	}
	if s := bulletUniformity(body); s != nil {
		out = append(out, *s)
	}
	return out
}

// bulletUniformity: agent-written bullet lists have suspiciously even lengths.
func bulletUniformity(t string) *detect.Signal {
	bullets := bulletRE.FindAllStringSubmatch(t, -1)
	if len(bullets) < 4 {
		return nil
	}
	lens := make([]float64, len(bullets))
	for i, b := range bullets {
		lens[i] = float64(len(text.Words(b[1])))
	}
	mean, std := text.MeanStd(lens)
	if mean == 0 {
		return nil
	}
	cv := std / mean
	p := 0.8
	switch {
	case cv > 0.6:
		p = 0.3
	case cv > 0.4:
		p = 0.5
	case cv > 0.25:
		p = 0.65
	}
	return &detect.Signal{Name: "bullet_uniformity", Value: cv, AIProb: p, Weight: 0.5,
		Detail: fmt.Sprintf("bullet length variation %.2f across %d bullets", cv, len(bullets))}
}
