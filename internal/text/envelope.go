package text

import (
	"regexp"
	"strings"
)

var (
	urlRE      = regexp.MustCompile(`(?i)\bhttps?://\S+|\bwww\.\S+`)
	issueRefRE = regexp.MustCompile(`(?i)\b(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?|refs?|see|related to|part of)\s*:?\s*(?:[\w.-]+/[\w.-]+)?#\d+\b|(?:[\w.-]+/[\w.-]+)?#\d+\b`)
	mentionRE  = regexp.MustCompile(`(?:^|\s)@[\w-]+`)
	spacesRE   = regexp.MustCompile(`[ \t]{2,}`)
	orphanRE   = regexp.MustCompile(`(?m)^[ \t]*[.,;:]+[ \t]*`)
	emptyPunct = regexp.MustCompile(`\(\s*\)|\[\s*\]`)
	signoffRE  = regexp.MustCompile(`(?i)\b(?:best regards|kind regards|warm regards|many thanks|regards|best|thanks|thank you|cheers|sincerely|talk soon)\s*[,!.]?`)
	greetingRE = regexp.MustCompile(`(?i)^(?:hi|hello|hey|dear|good (?:morning|afternoon|evening)|greetings)\b`)
)

// Normalize applies the corpus normalization for a kind ("commit", "pr",
// "memo", or a detect.Kind such as "prose"). Training and detection both use
// it, so the word model sees the same view of a text in both places.
func Normalize(kind, s string) string {
	switch kind {
	case "commit", "pr":
		return StripLinks(s)
	case "memo", "prose":
		// Fenced code isn't prose; the other detectors already skip it.
		return StripEnvelope(StripLinks(StripCode(s)))
	}
	return s
}

// StripLinks removes URLs, issue references, and @mentions. Human commits
// and PRs link to issues and people that an AI writing from a diff never
// sees, so leaving them in would teach "has a link = human".
func StripLinks(s string) string {
	s = urlRE.ReplaceAllString(s, "")
	s = issueRefRE.ReplaceAllString(s, "")
	s = mentionRE.ReplaceAllString(s, "")
	s = emptyPunct.ReplaceAllString(s, "")
	s = spacesRE.ReplaceAllString(s, " ")
	s = orphanRE.ReplaceAllString(s, "")
	var lines []string
	for _, l := range strings.Split(s, "\n") {
		lines = append(lines, strings.TrimRight(l, " \t"))
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// StripEnvelope removes an email's greeting line and closing sign-off.
// Salutations vary by era and company more than by author (the human memo
// corpus is from 2001), so they're excluded from what the detector learns.
func StripEnvelope(s string) string {
	s = strings.TrimSpace(s)
	if first, rest, ok := strings.Cut(s, "\n"); ok {
		f := strings.TrimSpace(first)
		short := len(strings.Fields(f)) <= 6 && !strings.ContainsAny(strings.TrimRight(f, ",:"), ".?!")
		if greetingRE.MatchString(f) || (short && strings.HasSuffix(f, ",")) || (short && strings.TrimSpace(rest) != "" && !strings.HasSuffix(f, ".")) {
			s = strings.TrimSpace(rest)
		}
	}
	// A sign-off counts only near the end: what follows it must be a name or
	// title, not more prose.
	locs := signoffRE.FindAllStringIndex(s, -1)
	for i := len(locs) - 1; i >= 0; i-- {
		tail := s[locs[i][1]:]
		if len(strings.Fields(tail)) <= 12 && !strings.ContainsAny(strings.TrimSpace(tail), ".?!") {
			s = strings.TrimSpace(s[:locs[i][0]])
			break
		}
		if len(strings.Fields(tail)) > 12 {
			break
		}
	}
	return s
}
