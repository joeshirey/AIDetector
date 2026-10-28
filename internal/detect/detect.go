// Package detect defines the shared types every detector implements.
package detect

import (
	"context"
	"regexp"
	"strings"
)

// Kind is the type of text being analyzed. Detectors use it to decide which
// signals apply, and the engine uses it to pick length expectations.
type Kind string

const (
	KindAuto  Kind = "auto"
	KindProse Kind = "prose" // blog posts, memos, reports, emails, essays
	KindPR    Kind = "pr"    // pull request descriptions
)

// ParseKind normalizes user input, defaulting to KindAuto.
func ParseKind(s string) Kind {
	switch Kind(strings.ToLower(strings.TrimSpace(s))) {
	case KindProse, "blog", "memo", "report":
		return KindProse
	case KindPR, "pull_request", "pull-request":
		return KindPR
	default:
		return KindAuto
	}
}

// Input is a single piece of text to analyze.
type Input struct {
	Text string
	Kind Kind
}

// Signal is one piece of evidence produced by a detector.
type Signal struct {
	Name     string  `json:"name"`
	Detector string  `json:"detector"`
	Value    float64 `json:"value"`              // raw feature value
	AIProb   float64 `json:"ai_prob"`            // 0..1, how strongly this signal points to AI
	Weight   float64 `json:"weight"`             // relative trust in this signal
	Decisive bool    `json:"decisive,omitempty"` // a near-certain tell, e.g. an AI co-author trailer
	Detail   string  `json:"detail,omitempty"`
}

// Detector produces signals for a piece of text. Implementations should
// return no signals (rather than a neutral one) when they don't apply or
// lack enough text to say anything.
type Detector interface {
	Name() string
	Detect(ctx context.Context, in Input) ([]Signal, error)
}

var prHeadingRE = regexp.MustCompile(`(?mi)^#{1,3}\s*(summary|changes|what changed|test plan|testing|motivation|description|why|how)\b`)

// GuessKind infers the kind of text when the caller didn't specify one:
// PR-style section headings mean a PR description, anything else is prose.
func GuessKind(text string) Kind {
	if prHeadingRE.MatchString(strings.TrimSpace(text)) {
		return KindPR
	}
	return KindProse
}
