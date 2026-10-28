package text

import (
	"regexp"
	"strings"
	"unicode"
)

// Exclusions counts the words PrepareProse removed, by kind of content.
type Exclusions struct {
	Code        int `json:"code"`
	Tables      int `json:"tables"`
	FrontMatter int `json:"front_matter"`
	Labels      int `json:"labels"` // headings, captions, and other short label lines
}

func (x Exclusions) Total() int { return x.Code + x.Tables + x.FrontMatter + x.Labels }

var (
	fenceLineRE   = regexp.MustCompile("^\\s*(```|~~~)")
	htmlTableRE   = regexp.MustCompile(`(?is)<table\b.*?</table>`)
	yamlKeyRE     = regexp.MustCompile(`^\s*[A-Za-z_][\w.-]*:(\s|$)`)
	codeStartRE   = regexp.MustCompile(`^\s*(\$ |#!|//|/\*|\*/|import |from \S+ import|package |func |def |class \w+[:({]|const |var |let |return\b|if \(|for \(|while \(|public |private |<\?|<!--|<[a-zA-Z/][^>]*>\s*$|[{}\[\]]\s*$|\w+\s*=\s*\S+;?$|[\w.]+\([^)]*\);?$|npm |go |pip |docker |kubectl |gcloud |git |curl |cd |export |sudo )`)
	endPunctRE    = regexp.MustCompile(`[.!?:;"'”’)\]]\s*$`)
	sentencePunct = regexp.MustCompile(`[\p{L}\p{N})"'”’][.!?](\s|$)`) // a lone " ." (as in "docker build .") isn't a sentence end
	listMarkRE    = regexp.MustCompile(`^\s*([-*+]|\d+[.)])\s+`)
	headingMarkRE = regexp.MustCompile(`^\s*#{1,6}\s`)

	// Sentences glued together when line breaks were lost in a paste:
	// "why.In 1938" -> "why. In 1938". The left side must be a lowercase
	// word of 2+ letters, so "e.g.The", "U.S.Army", "v1.2", and
	// "cloudbuild.yaml" never match.
	gluedRE = regexp.MustCompile(`(\S*?[a-z]{2,})([.!?])([A-Z][a-z]+)(\S*)`)
	// A heading glued to a sentence that starts with "A" or "I":
	// "Why It ExistA clay pot" -> "Why It Exist A clay pot".
	gluedAIRE = regexp.MustCompile(`\b([A-Z]?[a-z]{3,})([AI]) ([a-z])`)
	dottedRE  = regexp.MustCompile(`^[A-Za-z_]\w*(\.[A-Za-z_]\w*){2,}`)
)

// PrepareProse removes content that isn't prose (code, tables, YAML front
// matter, label lines) and repairs sentences glued together by lost line
// breaks. It is idempotent, and text.Normalize applies it for prose, so
// training and detection see the same text.
func PrepareProse(s string) (string, Exclusions) {
	var x Exclusions
	s = htmlTableRE.ReplaceAllStringFunc(s, func(m string) string {
		x.Tables += len(Words(m))
		return "\n\n"
	})
	lines := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
	keep := make([]bool, len(lines))
	for i := range keep {
		keep[i] = true
	}
	drop := func(from, to int, counter *int) {
		for j := from; j < to; j++ {
			if keep[j] {
				*counter += len(Words(lines[j]))
				keep[j] = false
			}
		}
	}

	for i := 0; i < len(lines); i++ {
		l := lines[i]
		switch {
		case fenceLineRE.MatchString(l): // fenced code, through the closing fence
			j := i + 1
			for j < len(lines) && !fenceLineRE.MatchString(lines[j]) {
				j++
			}
			drop(i, min(j+1, len(lines)), &x.Code)
			i = j
		case strings.TrimSpace(l) == "---" && isFrontMatter(lines, i): // --- key: value ... ---
			j := i + 1
			for strings.TrimSpace(lines[j]) != "---" {
				j++
			}
			drop(i, j+1, &x.FrontMatter)
			i = j
		case strings.HasPrefix(strings.TrimSpace(l), "|") && strings.Count(l, "|") >= 2:
			j := i
			for j < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[j]), "|") {
				j++
			}
			drop(i, j, &x.Tables)
			i = j - 1
		}
	}
	// Runs of code-like lines: indented blocks or pasted code without fences.
	for i := 0; i < len(lines); {
		if !keep[i] || !isCodeLine(lines[i]) {
			i++
			continue
		}
		j := i
		for j < len(lines) && keep[j] && (isCodeLine(lines[j]) || strings.TrimSpace(lines[j]) == "") {
			j++
		}
		code := 0
		for k := i; k < j; k++ {
			if strings.TrimSpace(lines[k]) != "" {
				code++
			}
		}
		if code >= 2 || (code == 1 && isStrongCode(lines[i])) {
			drop(i, j, &x.Code)
		}
		i = j
	}
	for i, l := range lines {
		if !keep[i] {
			continue
		}
		t := strings.TrimSpace(l)
		switch {
		case t == "":
		case isFlattenedTable(t):
			drop(i, i+1, &x.Tables)
		case isLabel(t):
			drop(i, i+1, &x.Labels)
		}
	}

	var out []string
	for i, l := range lines {
		if keep[i] {
			out = append(out, repairGlued(l))
		} else if len(out) > 0 && out[len(out)-1] != "" {
			out = append(out, "") // keep paragraph boundaries where content was cut
		}
	}
	return strings.TrimSpace(strings.Join(out, "\n")), x
}

func isFrontMatter(lines []string, i int) bool {
	keys := 0
	for j := i + 1; j < len(lines) && j < i+40; j++ {
		t := strings.TrimSpace(lines[j])
		if t == "---" {
			return keys >= 1
		}
		if yamlKeyRE.MatchString(lines[j]) {
			keys++
		} else if t != "" && !strings.HasPrefix(lines[j], " ") && !strings.HasPrefix(t, "-") {
			return false
		}
	}
	return false
}

// isCodeLine reports whether a line looks like code rather than prose.
func isCodeLine(l string) bool {
	if strings.TrimSpace(l) == "" {
		return false
	}
	if strings.HasPrefix(l, "    ") || strings.HasPrefix(l, "\t") {
		if !listMarkRE.MatchString(l) {
			return true
		}
	}
	t := strings.TrimSpace(l)
	if codeStartRE.MatchString(t) && !sentencePunct.MatchString(t) {
		return true
	}
	if yamlKeyRE.MatchString(t) && len(strings.Fields(t)) <= 6 && !sentencePunct.MatchString(t) {
		return true
	}
	return symbolRatio(t) > 0.18
}

// isStrongCode is a single line that's unmistakably code.
func isStrongCode(l string) bool {
	t := strings.TrimSpace(l)
	return symbolRatio(t) > 0.3 || strings.HasSuffix(t, ";") || strings.HasPrefix(t, "$ ")
}

func symbolRatio(t string) float64 {
	var sym, n int
	for _, r := range t {
		if unicode.IsSpace(r) {
			continue
		}
		n++
		if strings.ContainsRune("{}()[]<>=;$/\\|_`*&^%#@~+", r) {
			sym++
		}
	}
	if n == 0 {
		return 0
	}
	return float64(sym) / float64(n)
}

// isLabel: headings, captions, and other short lines with no sentence
// punctuation. List items are labels only when they're this short too.
func isLabel(t string) bool {
	if headingMarkRE.MatchString(t) {
		return true
	}
	t = listMarkRE.ReplaceAllString(t, "")
	return len(Words(t)) < 4 && !endPunctRE.MatchString(t)
}

// isFlattenedTable catches a table pasted as one run-on line: long, almost
// no sentence punctuation, and mostly capitalized words or numbers.
func isFlattenedTable(t string) bool {
	ws := strings.Fields(t)
	if len(ws) < 12 || len(sentencePunct.FindAllStringIndex(t, -1)) > len(ws)/25 {
		return false
	}
	caps := 0
	for _, w := range ws {
		r := []rune(strings.TrimLeft(w, `"'(“`))
		if len(r) > 0 && (unicode.IsUpper(r[0]) || unicode.IsDigit(r[0])) {
			caps++
		}
	}
	return float64(caps)/float64(len(ws)) >= 0.5
}

// repairGlued inserts the missing space in "word.Word" unless the token is a
// URL, an email address, or a dotted identifier like Microsoft.BizTalk.Pipeline.
func repairGlued(l string) string {
	l = gluedRE.ReplaceAllStringFunc(l, func(m string) string {
		if strings.Contains(m, "://") || strings.HasPrefix(m, "www.") || strings.Contains(m, "@") ||
			dottedRE.MatchString(m) || strings.ContainsAny(m, "()/=_") {
			return m
		}
		p := gluedRE.FindStringSubmatch(m)
		return p[1] + p[2] + " " + p[3] + p[4]
	})
	return gluedAIRE.ReplaceAllString(l, "$1 $2 $3")
}
