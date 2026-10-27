// Package text provides tokenization helpers shared by detectors.
package text

import (
	"math"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

var (
	wordRE     = regexp.MustCompile(`[\p{L}\p{N}]+(?:['’]\p{L}+)*`)
	fenceRE    = regexp.MustCompile("(?s)```.*?```")
	paraRE     = regexp.MustCompile(`\n\s*\n`)
	bulletRE   = regexp.MustCompile(`^\s*(?:[-*+]|\d+[.)])\s+`)
	headingRE  = regexp.MustCompile(`^\s*#{1,6}\s`)
	boundaryRE = regexp.MustCompile(`[.!?]+["'”’)\]]*\s+`)
)

var abbreviations = map[string]bool{
	"e.g.": true, "i.e.": true, "etc.": true, "vs.": true, "mr.": true, "mrs.": true,
	"ms.": true, "dr.": true, "st.": true, "inc.": true, "approx.": true, "u.s.": true,
}

// Words returns the word tokens in s.
func Words(s string) []string { return wordRE.FindAllString(s, -1) }

// StripCode removes fenced code blocks, which would distort prose statistics.
func StripCode(s string) string { return fenceRE.ReplaceAllString(s, "") }

// Paragraphs splits s on blank lines, dropping empty paragraphs.
func Paragraphs(s string) []string {
	var out []string
	for _, p := range paraRE.Split(s, -1) {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// Sentences splits s into sentences. Headings are skipped and each list item
// counts as its own sentence, since that's how a reader experiences them.
func Sentences(s string) []string {
	var out []string
	for _, para := range Paragraphs(s) {
		var prose []string
		flush := func() {
			if len(prose) > 0 {
				out = append(out, splitProse(strings.Join(prose, " "))...)
				prose = nil
			}
		}
		for _, line := range strings.Split(para, "\n") {
			switch {
			case headingRE.MatchString(line):
				flush()
			case bulletRE.MatchString(line):
				flush()
				prose = append(prose, bulletRE.ReplaceAllString(line, ""))
				flush()
			default:
				prose = append(prose, strings.TrimSpace(line))
			}
		}
		flush()
	}
	return out
}

func splitProse(s string) []string {
	var out []string
	start := 0
	for _, loc := range boundaryRE.FindAllStringIndex(s, -1) {
		// Only cut when the next sentence looks like it starts properly.
		if loc[1] < len(s) {
			r, _ := utf8.DecodeRuneInString(s[loc[1]:])
			if !unicode.IsUpper(r) && !unicode.IsDigit(r) && !strings.ContainsRune(`"'“‘(`, r) {
				continue
			}
		}
		fields := strings.Fields(s[start : loc[0]+1])
		if len(fields) > 0 && abbreviations[strings.ToLower(fields[len(fields)-1])] {
			continue
		}
		if seg := strings.TrimSpace(s[start:loc[1]]); seg != "" {
			out = append(out, seg)
		}
		start = loc[1]
	}
	if seg := strings.TrimSpace(s[start:]); seg != "" {
		out = append(out, seg)
	}
	return out
}

// Windows groups consecutive paragraphs into chunks of at least target words,
// so long documents can be scored piece by piece.
func Windows(s string, target int) []string {
	var out []string
	var cur []string
	n := 0
	for _, p := range Paragraphs(s) {
		cur = append(cur, p)
		n += len(Words(p))
		if n >= target {
			out = append(out, strings.Join(cur, "\n\n"))
			cur, n = nil, 0
		}
	}
	if len(cur) > 0 {
		// Fold a short tail into the previous window rather than scoring a fragment.
		if len(out) > 0 && n < target/2 {
			out[len(out)-1] += "\n\n" + strings.Join(cur, "\n\n")
		} else {
			out = append(out, strings.Join(cur, "\n\n"))
		}
	}
	return out
}

// MeanStd returns the mean and population standard deviation of xs.
func MeanStd(xs []float64) (mean, std float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	for _, x := range xs {
		std += (x - mean) * (x - mean)
	}
	std /= float64(len(xs))
	return mean, math.Sqrt(std)
}
