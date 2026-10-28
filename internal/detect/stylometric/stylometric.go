// Package stylometric scores text using surface-level writing statistics:
// rhythm, vocabulary, punctuation, and formatting habits common in LLM output.
//
// Every threshold here is a hand-set starting point. They're meant to be
// replaced by weights fitted on the eval set (see internal/eval).
package stylometric

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strings"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/text"
)

const name = "stylometric"

type Detector struct{}

func New() *Detector { return &Detector{} }

func (*Detector) Name() string { return name }

func (*Detector) Detect(_ context.Context, in detect.Input) ([]detect.Signal, error) {
	body := text.StripCode(in.Text)
	words := text.Words(body)
	if len(words) < 15 {
		return nil, nil
	}
	sentences := text.Sentences(body)
	per1k := 1000 / float64(len(words))

	var out []detect.Signal
	add := func(s *detect.Signal) {
		if s != nil {
			s.Detector = name
			out = append(out, *s)
		}
	}
	add(aiVocabulary(words, per1k))
	add(aiPhrases(body, per1k))
	add(burstiness(sentences))
	add(emDashes(body, per1k))
	add(boldLeadBullets(body))
	add(triplets(sentences))
	if in.Kind == detect.KindProse {
		add(paragraphUniformity(body))
	}
	return out, nil
}

// logistic maps x to 0..1, centered on mid. A negative scale flips direction.
func logistic(x, mid, scale float64) float64 {
	return 1 / (1 + math.Exp(-(x-mid)/scale))
}

// presence is for tell-tale features: finding them is strong evidence of AI,
// but not finding them is only weak evidence of a human, since plenty of AI
// text avoids any given tell.
func presence(x, mid, scale float64) float64 {
	return 0.4 + 0.6*logistic(x, mid, scale)
}

// burstiness: humans mix short and long sentences; models are more even.
func burstiness(sentences []string) *detect.Signal {
	if len(sentences) < 5 {
		return nil
	}
	lens := make([]float64, len(sentences))
	for i, s := range sentences {
		lens[i] = float64(len(text.Words(s)))
	}
	mean, std := text.MeanStd(lens)
	if mean == 0 {
		return nil
	}
	cv := std / mean
	return &detect.Signal{
		Name:   "sentence_burstiness",
		Value:  cv,
		AIProb: logistic(cv, 0.45, -0.08),
		Weight: 1.0,
		Detail: fmt.Sprintf("sentence length variation %.2f across %d sentences (lower reads more machine-like)", cv, len(sentences)),
	}
}

// aiStems are word stems LLMs use far more often than people do. Matching on
// stems catches inflections (delve, delves, delving).
var aiStems = []string{
	"delv", "tapestr", "testament", "intricat", "pivotal", "underscor", "realm",
	"multifaceted", "seamless", "foster", "embark", "showcas", "vibrant",
	"bustling", "meticulous", "paramount", "holistic", "synerg", "unwavering",
	"invaluable", "commendable", "noteworthy", "unparalleled", "transformative",
	"groundbreaking", "landscape", "empower", "streamlin", "bolster", "nuanc",
	"interplay", "enhanc", "comprehensive", "crucial", "furthermore", "moreover",
	"additionally", "notably", "leverag", "utiliz", "facilitat", "insightful",
	"resonat", "captivat", "beacon", "symphony", "labyrinth", "elevat",
	"robust", "ensur",
}

func aiVocabulary(words []string, per1k float64) *detect.Signal {
	hits := map[string]int{}
	total := 0
	for _, w := range words {
		lw := strings.ToLower(w)
		for _, stem := range aiStems {
			if strings.HasPrefix(lw, stem) && len(lw)-len(stem) <= 4 {
				hits[lw]++
				total++
				break
			}
		}
	}
	rate := float64(total) * per1k
	return &detect.Signal{
		Name:   "ai_vocabulary",
		Value:  rate,
		AIProb: presence(rate, 6, 2.5),
		Weight: 1.2,
		Detail: strings.TrimSpace(fmt.Sprintf("%.1f AI-favored words per 1k %s", rate, topHits(hits))),
	}
}

var phraseREs = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bit'?s (worth|important) (noting|to note|mentioning)`),
	regexp.MustCompile(`(?i)\bin today'?s (fast-paced|digital|ever-changing|rapidly evolving)`),
	regexp.MustCompile(`(?i)\bplay(s|ing)? an? (crucial|pivotal|vital|key|critical) role`),
	regexp.MustCompile(`(?i)\b(a|stands as a|serves as a) testament to`),
	regexp.MustCompile(`(?i)\b(not|isn'?t) (just|only|merely)\b[^.\n]{1,80}\bbut\b`),
	regexp.MustCompile(`(?i)\b(it'?s|this is|that'?s) not [^.\n]{1,40}[—–] ?(it'?s|this is|that'?s)`),
	regexp.MustCompile(`(?i)\blet'?s (dive|delve)\b|\bdive (deep(er)? )?into\b`),
	regexp.MustCompile(`(?i)\b(in conclusion|to sum up|in summary),`),
	regexp.MustCompile(`(?i)\bwhether you'?re an? `),
	regexp.MustCompile(`(?i)\bnavigat(e|ing) the (complexities|intricacies|challenges)`),
	regexp.MustCompile(`(?i)\b(ever-)?evolving (landscape|world)`),
	regexp.MustCompile(`(?i)\bat its core\b`),
	regexp.MustCompile(`(?i)\bhere'?s (the thing|why|what|how)\b`),
	regexp.MustCompile(`(?i)\b(game[- ]changer|cutting[- ]edge|treasure trove)\b`),
	regexp.MustCompile(`(?i)\bi hope this (email|message|note) finds you well`),
	regexp.MustCompile(`(?i)\b(feel free to|don'?t hesitate to) (reach out|ask|contact)`),
	regexp.MustCompile(`(?i)\bkey takeaways?\b`),
	regexp.MustCompile(`(?i)\bmore than just\b`),
	regexp.MustCompile(`(?i)\bthe (result|bottom line|kicker|catch)\?`),
}

func aiPhrases(body string, per1k float64) *detect.Signal {
	total := 0
	var found []string
	for _, re := range phraseREs {
		for _, m := range re.FindAllString(body, -1) {
			total++
			if len(found) < 5 {
				found = append(found, fmt.Sprintf("%q", strings.ToLower(m)))
			}
		}
	}
	rate := float64(total) * per1k
	detail := fmt.Sprintf("%.1f stock LLM phrases per 1k words", rate)
	if len(found) > 0 {
		detail += " (" + strings.Join(found, ", ") + ")"
	}
	return &detect.Signal{
		Name:   "ai_phrases",
		Value:  rate,
		AIProb: presence(rate, 1.5, 0.8),
		Weight: 1.0,
		Detail: detail,
	}
}

func emDashes(body string, per1k float64) *detect.Signal {
	n := strings.Count(body, "—") + strings.Count(body, " -- ")
	rate := float64(n) * per1k
	return &detect.Signal{
		Name:   "em_dashes",
		Value:  rate,
		AIProb: presence(rate, 5, 2),
		Weight: 0.6,
		Detail: fmt.Sprintf("%.1f em dashes per 1k words", rate),
	}
}

var (
	bulletLineRE = regexp.MustCompile(`(?m)^[ \t]*(?:[-*+]|\d+[.)])\s+`)
	boldLeadRE   = regexp.MustCompile(`(?m)^[ \t]*(?:[-*+]|\d+[.)])\s+\*\*[^*\n]+\*\*`)
)

// boldLeadBullets: "- **Thing:** explanation" lists are a strong LLM formatting habit.
func boldLeadBullets(body string) *detect.Signal {
	bullets := len(bulletLineRE.FindAllString(body, -1))
	if bullets < 3 {
		return nil
	}
	bold := len(boldLeadRE.FindAllString(body, -1))
	ratio := float64(bold) / float64(bullets)
	return &detect.Signal{
		Name:   "bold_lead_bullets",
		Value:  ratio,
		AIProb: presence(ratio, 0.4, 0.12),
		Weight: 0.8,
		Detail: fmt.Sprintf("%d of %d list items start with a bolded label", bold, bullets),
	}
}

var tripletRE = regexp.MustCompile(`(?i)\b[\w-]+(?: [\w-]+)?, [\w-]+(?: [\w-]+)?,? (?:and|or) [\w-]+`)

// triplets: models reach for lists of three ("fast, reliable, and secure").
func triplets(sentences []string) *detect.Signal {
	if len(sentences) < 5 {
		return nil
	}
	n := 0
	for _, s := range sentences {
		if tripletRE.MatchString(s) {
			n++
		}
	}
	ratio := float64(n) / float64(len(sentences))
	return &detect.Signal{
		Name:   "rule_of_three",
		Value:  ratio,
		AIProb: presence(ratio, 0.18, 0.07),
		Weight: 0.5,
		Detail: fmt.Sprintf("%d of %d sentences contain a three-item list", n, len(sentences)),
	}
}

// paragraphUniformity: models write paragraphs of strikingly similar length.
func paragraphUniformity(body string) *detect.Signal {
	paras := text.Paragraphs(body)
	var lens []float64
	for _, p := range paras {
		if n := len(text.Words(p)); n >= 8 { // ignore headings and one-liners
			lens = append(lens, float64(n))
		}
	}
	if len(lens) < 4 {
		return nil
	}
	mean, std := text.MeanStd(lens)
	cv := std / mean
	return &detect.Signal{
		Name:   "paragraph_uniformity",
		Value:  cv,
		AIProb: logistic(cv, 0.35, -0.1),
		Weight: 0.5,
		Detail: fmt.Sprintf("paragraph length variation %.2f across %d paragraphs", cv, len(lens)),
	}
}

func topHits(hits map[string]int) string {
	if len(hits) == 0 {
		return ""
	}
	var ws []string
	for w := range hits {
		ws = append(ws, w)
	}
	// Most frequent first, alphabetical for ties, so output is stable.
	slices.SortFunc(ws, func(a, b string) int {
		if c := cmp.Compare(hits[b], hits[a]); c != 0 {
			return c
		}
		return strings.Compare(a, b)
	})
	if len(ws) > 6 {
		ws = ws[:6]
	}
	return "(" + strings.Join(ws, ", ") + ")"
}
