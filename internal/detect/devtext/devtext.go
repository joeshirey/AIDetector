// Package devtext looks for tells specific to pull request descriptions,
// plus AI tool attributions that can appear in any text.
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
	// bodies into other documents too.
	aiTrailerRE = regexp.MustCompile(`(?mi)^co-authored-by:.*\b(claude|copilot|chatgpt|openai|gpt-?\d|gemini|cursor|devin|codex|aider|noreply@anthropic\.com)\b.*` +
		`|generated (with|by) \[?(claude|copilot|chatgpt|cursor|codex|gemini)` +
		`|🤖 generated with`)

	prSectionRE = regexp.MustCompile(`(?mi)^#{1,3}\s*(summary|changes|what changed|test plan|testing|motivation|key changes|overview)\b`)
	checkboxRE  = regexp.MustCompile(`(?m)^[ \t]*[-*]\s+\[[ xX]\]\s`)
	bulletRE    = regexp.MustCompile(`(?m)^[ \t]*[-*]\s+(.+)$`)
	emojiHeadRE = regexp.MustCompile(`(?m)^#{1,3}\s*[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]`)
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
