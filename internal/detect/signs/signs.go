// Package signs measures the patterns catalogued in Wikipedia's "Signs of AI
// writing" (WikiProject AI Cleanup) and the humanizer skill built on it
// (github.com/blader/humanizer). Section numbers (§N) refer to humanizer v3.1.
//
// Each signal reports a raw rate. The hand-set AIProb values only matter for
// the heuristic engine; fitted models learn their own weights from Value.
package signs

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/text"
)

const name = "signs"

type Detector struct{}

func New() *Detector { return &Detector{} }

func (*Detector) Name() string { return name }

// phraseSet is a named group of patterns counted per 1,000 words.
type phraseSet struct {
	name, detail string
	mid          float64 // rate per 1k words at which AIProb crosses 0.5
	re           *regexp.Regexp
}

func words(ws ...string) string { return `(?i)\b(?:` + strings.Join(ws, "|") + `)\b` }

var phraseSets = []phraseSet{
	{"negative_parallelism", "not-X-but-Y contrasts (§1)", 2, regexp.MustCompile(`(?i)\b(?:not|isn'?t|aren'?t|wasn'?t)\s+(?:just|only|merely|simply)\b|\b(?:it'?s|this is|that'?s|they'?re)\s+not\s+[^.;\n]{1,50}[,;—–]\s*(?:it'?s|this is|that'?s|they'?re)\b|\bdoes(?:n'?t| not) mean\b[^.]{0,80}\.\s+it means\b|\brather than\b`)},
	{"stock_closers", "one-line closers and dramatic fragments (§2)", 0.5, regexp.MustCompile(words(`that'?s the real (?:win|story|lesson)`, `that distinction matters`, `read that again`, `let that sink in`, `the message was clear`, `and that'?s the point`, `that changes everything`, `this is where it gets interesting`, `this shows the importance of`) + `|\b\w+\.\s\w+\.\s\w+\.\s\w+\.`)},
	{"deep_sayings", "sayings that sound deep (§3)", 1, regexp.MustCompile(words(`the real question is`, `at its core`, `what really matters`, `the heart of the matter`, `the deeper issue`, `the (?:language|currency|architecture) of`, `becomes a trap`, `fundamentally`))},
	{"staged_runup", "staged run-ups before the point (§4)", 0.7, regexp.MustCompile(words(`let'?s (?:dive|delve|explore|break (?:this|it) down|unpack)`, `here'?s what you need to know`, `without further ado`, `here'?s the thing`, `the thing is`, `let'?s be honest`, `real talk`, `quick note`) + `|(?m)^(?:Honestly|Look|So)\?\s`)},
	{"arguing_with_no_one", "rebutting objections nobody raised (§5)", 0.7, regexp.MustCompile(words(`this isn'?t (?:mainly |really |just )?about`, `i'?m not saying`, `to be clear`, `don'?t get me wrong`, `this is not to say`, `one might be tempted`, `a tempting approach`, `you might think`, `it would be easy to (?:just )?`))},
	{"stacked_qualifiers", "stacked qualifiers (§9)", 0.7, regexp.MustCompile(words(`could potentially`, `might arguably`, `may potentially`, `it'?s also possible`, `to be fair`, `in some cases it may`))},
	{"vocab_2023", "2023–mid-2024 AI vocabulary (Wikipedia)", 4, regexp.MustCompile(words(`additionally`, `boasts?`, `bolstered`, `crucial`, `delv(?:e|es|ed|ing)`, `enduring`, `garner(?:s|ed)?`, `intricate`, `intricacies`, `interplay`, `landscape`, `meticulous(?:ly)?`, `pivotal`, `tapestry`, `testament`, `vibrant`, `underscor(?:e|es|ed|ing)`))},
	{"vocab_2024", "mid-2024–mid-2026 AI vocabulary (Wikipedia)", 3, regexp.MustCompile(words(`align(?:s|ed)? with`, `enhanc(?:e|es|ed|ing)`, `foster(?:s|ed|ing)?`, `highlight(?:s|ed|ing)`, `showcas(?:e|es|ed|ing)`, `emphasiz(?:e|es|ed|ing)`, `seamless(?:ly)?`, `robust`, `comprehensive`))},
	{"vocab_2026", "mid-2026+ AI vocabulary (Wikipedia)", 4, regexp.MustCompile(words(`challenges?`, `clearer`, `dependable`, `echo(?:ed|es)`, `leverag(?:e|es|ed|ing)`, `matters`, `multifaceted`, `practical`, `prioritiz(?:e|es|ed|ing)`, `quietly`, `steady`, `universally`))},
	{"inflated_significance", "inflated significance (§13)", 0.7, regexp.MustCompile(words(`stands as a`, `marking a (?:pivotal|crucial|significant)`, `plays? an? (?:key|crucial|pivotal|vital) role`, `setting the stage`, `(?:enduring|lasting) legacy`, `indelible mark`, `reflects a broader`, `the future looks bright`, `exciting times ahead`, `a step in the right direction`, `continues to thrive`, `evolving landscape`))},
	{"vague_association", "vague connection or association (§14)", 1.5, regexp.MustCompile(words(`(?:is|are|was|were) associated with`, `in association with`, `in connection with`, `(?:is|are|was|were) (?:linked|tied|connected) to`))},
	{"ing_riders", "shallow -ing riders after a comma (§15)", 2, regexp.MustCompile(`(?i),\s(?:highlighting|underscoring|emphasizing|ensuring|reflecting|symbolizing|contributing to|cultivating|fostering|encompassing|showcasing|demonstrating|making it)\b`)},
	{"sales_language", "sales language (§16)", 1, regexp.MustCompile(words(`nestled`, `in the heart of`, `breathtaking`, `must-visit`, `stunning`, `renowned`, `diverse array`, `commitment to`, `rich (?:history|heritage|tradition|culture)`, `exemplifies`))},
	{"copula_avoidance", "avoiding is/are/has (§18)", 1.5, regexp.MustCompile(words(`serves as`, `stands as`, `functions as`, `operates as`, `boasts`, `acts as a`))},
	{"knowledge_disclaimers", "knowledge-limit disclaimers and corrective framings (§23, mid-2026)", 0.5, regexp.MustCompile(words(`based on (?:the )?available information`, `not (?:publicly available|widely documented|extensively documented)`, `while specific details`, `does not establish`, `should not be confused with`, `should be (?:treated|read|understood) as [^.]{1,40} rather than`))},
	// From "Forensics for AI Generated Text" (persistent 2026 syntactic artifacts).
	{"signposts", "mechanical signpost openers (Furthermore, Moreover…)", 1.5, regexp.MustCompile(`(?im)(?:^|[.!?]\s+)(?:furthermore|moreover|additionally|consequently|notably|importantly|ultimately|overall|in summary|in conclusion|to summarize|it(?: i|')s worth noting that|it(?: i|')s important to note that),`)},
	{"corporate_verbs", "corporate Latinate verbs (utilize, facilitate…)", 2, regexp.MustCompile(words(`utiliz(?:e|es|ed|ing)`, `facilitat(?:e|es|ed|ing)`, `streamlin(?:e|es|ed|ing)`, `empower(?:s|ed|ing)?`, `harness(?:es|ed|ing)?`, `revolutioniz(?:e|es|ed|ing)`, `elevat(?:e|es|ed|ing)`, `optimiz(?:e|es|ed|ing)`, `spearhead(?:s|ed|ing)?`))},
	{"chatbot_phrases", "chatbot wrapper phrases (§22)", 0.5, regexp.MustCompile(words(`i hope this helps`, `great question`, `certainly! here`, `would you like me to`, `let me know if you'?d like me to`, `happy to help`))},
	{"human_markers", "plain constructions humans use more (Wikipedia: Syntax)", 6, regexp.MustCompile(words(`there (?:is|are|was|were) an?`, `it has an?`, `very`, `perhaps`, `tends? to`, `in order to`, `the fact that`, `all of the`, `a part of`, `as a result of`, `one of the (?:best|worst|most)`, `was the first`, `is the only`, `kind of`, `sort of`, `pretty much`))},
}

// residueRE matches text only a chatbot produces (§22). One hit is decisive.
// Friendlier wrappers ("I hope this helps", "great question") also occur in
// human blogs and essays, so they're an ordinary feature (chatbot_phrases).
var residueRE = regexp.MustCompile(words(`as an ai(?: language model| assistant)`, `as of my (?:last )?(?:knowledge|training) (?:cutoff|update)`, `up to my last training`, `i (?:can'?t|cannot) browse the (?:internet|web)`, `i don'?t have (?:real-time|browsing) (?:access|capabilities)`))

func (*Detector) Detect(_ context.Context, in detect.Input) ([]detect.Signal, error) {
	body := text.StripCode(in.Text)
	ws := text.Words(body)
	var out []detect.Signal
	add := func(s detect.Signal) {
		s.Detector = name
		out = append(out, s)
	}
	// Checked at any length: one hit is decisive, unless the phrase is quoted
	// (text discussing "as an AI language model" isn't using it).
	if m := unquotedMatch(residueRE, body); m != "" {
		add(detect.Signal{Name: "chatbot_residue", Value: 1, AIProb: 0.99, Weight: 3, Decisive: true,
			Detail: fmt.Sprintf("chatbot wrapper text: %q (§22)", m)})
	}
	if len(ws) < 15 {
		return out, nil
	}
	per1k := 1000 / float64(len(ws))
	for _, ps := range phraseSets {
		hits := ps.re.FindAllString(body, -1)
		rate := float64(len(hits)) * per1k
		p := presence(rate, ps.mid)
		if ps.name == "human_markers" {
			p = 1 - p // more of these means more human
		}
		add(detect.Signal{Name: ps.name, Value: rate, AIProb: p, Weight: 0.5,
			Detail: fmt.Sprintf("%.1f per 1k words: %s%s", rate, ps.detail, examples(hits))})
	}

	// Commits are too short for rhythm statistics.
	if in.Kind == detect.KindCommit {
		return out, nil
	}
	sentences := text.Sentences(body)
	if s, ok := sentenceBand(sentences); ok {
		add(s)
	}
	if s, ok := repeatedOpenings(sentences); ok {
		add(s)
	}
	if s, ok := oneLineParagraphs(body); ok {
		add(s)
	}
	if s, ok := punctuationVariety(body); ok {
		add(s)
	}
	if s, ok := lexicalDiversity(ws); ok {
		add(s)
	}
	if len(ws) >= 100 {
		add(yulesK(ws))
		add(mtld(ws))
	}
	return out, nil
}

func lowerAll(ws []string) []string {
	out := make([]string, len(ws))
	for i, w := range ws {
		out[i] = strings.ToLower(w)
	}
	return out
}

// yulesK measures vocabulary repetition independent of length (Yule 1944);
// higher means more repetitive.
func yulesK(ws []string) detect.Signal {
	freq := map[string]int{}
	for _, w := range lowerAll(ws) {
		freq[w]++
	}
	spectrum := map[int]int{}
	for _, f := range freq {
		spectrum[f]++
	}
	var s2 float64
	for m, v := range spectrum {
		s2 += float64(m*m) * float64(v)
	}
	n := float64(len(ws))
	k := 1e4 * (s2 - n) / (n * n)
	return detect.Signal{Name: "yules_k", Value: k, AIProb: 0.5, Weight: 0.2,
		Detail: fmt.Sprintf("Yule's K %.0f (vocabulary repetition; higher is more repetitive)", k)}
}

// mtld is the Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010):
// the mean run length that keeps the type-token ratio above 0.72.
func mtld(ws []string) detect.Signal {
	lower := lowerAll(ws)
	pass := func(seq []string) float64 {
		const threshold = 0.72
		var factors float64
		types := map[string]bool{}
		count := 0
		for _, w := range seq {
			count++
			types[w] = true
			if float64(len(types))/float64(count) <= threshold {
				factors++
				clear(types)
				count = 0
			}
		}
		if count > 0 {
			factors += (1 - float64(len(types))/float64(count)) / (1 - threshold)
		}
		if factors == 0 {
			return float64(len(seq))
		}
		return float64(len(seq)) / factors
	}
	rev := slices.Clone(lower)
	slices.Reverse(rev)
	v := (pass(lower) + pass(rev)) / 2
	return detect.Signal{Name: "mtld", Value: v, AIProb: 0.5, Weight: 0.2,
		Detail: fmt.Sprintf("MTLD %.0f (lexical diversity; higher is more varied)", v)}
}

// unquotedMatch returns the first match of re that isn't immediately inside
// quotes or backticks.
func unquotedMatch(re *regexp.Regexp, s string) string {
	for _, loc := range re.FindAllStringIndex(s, -1) {
		before := strings.TrimRight(s[:loc[0]], " ")
		if before != "" && strings.ContainsAny(before[len(before)-1:], "\"'`") ||
			strings.HasSuffix(before, "“") || strings.HasSuffix(before, "‘") {
			continue
		}
		return s[loc[0]:loc[1]]
	}
	return ""
}

// presence: finding a pattern is evidence of AI, not finding it is weak
// evidence of a human (see stylometric.presence).
func presence(rate, mid float64) float64 {
	return 0.4 + 0.6/(1+math.Exp(-(rate-mid)/(mid/2+0.1)))
}

func examples(hits []string) string {
	if len(hits) == 0 {
		return ""
	}
	seen := map[string]bool{}
	var ex []string
	for _, h := range hits {
		h = strings.ToLower(strings.Join(strings.Fields(h), " "))
		if !seen[h] && len(ex) < 4 {
			seen[h] = true
			ex = append(ex, fmt.Sprintf("%q", h))
		}
	}
	return " (" + strings.Join(ex, ", ") + ")"
}

// sentenceBand: current models keep sentences in a narrow 8–20 word band
// (Wikipedia, mid-2026 era notes).
func sentenceBand(sentences []string) (detect.Signal, bool) {
	if len(sentences) < 5 {
		return detect.Signal{}, false
	}
	in := 0
	for _, s := range sentences {
		if n := len(text.Words(s)); n >= 8 && n <= 20 {
			in++
		}
	}
	frac := float64(in) / float64(len(sentences))
	return detect.Signal{Name: "sentence_band", Value: frac, AIProb: 0.3 + 0.5*frac, Weight: 0.5,
		Detail: fmt.Sprintf("%.0f%% of sentences are 8–20 words", frac*100)}, true
}

// repeatedOpenings: consecutive sentences starting with the same word (§7).
func repeatedOpenings(sentences []string) (detect.Signal, bool) {
	if len(sentences) < 6 {
		return detect.Signal{}, false
	}
	repeats := 0
	prev := ""
	for _, s := range sentences {
		w := text.Words(s)
		first := ""
		if len(w) > 0 {
			first = strings.ToLower(w[0])
		}
		if first != "" && first == prev {
			repeats++
		}
		prev = first
	}
	frac := float64(repeats) / float64(len(sentences)-1)
	return detect.Signal{Name: "repeated_openings", Value: frac, AIProb: 0.45 + frac, Weight: 0.3,
		Detail: fmt.Sprintf("%.0f%% of sentences start with the same word as the one before (§7)", frac*100)}, true
}

// oneLineParagraphs: single-sentence paragraphs used as punchy closers (§2).
func oneLineParagraphs(body string) (detect.Signal, bool) {
	paras := text.Paragraphs(body)
	if len(paras) < 4 {
		return detect.Signal{}, false
	}
	single := 0
	for _, p := range paras {
		if len(text.Sentences(p)) == 1 && len(text.Words(p)) <= 15 && !strings.HasPrefix(strings.TrimSpace(p), "#") {
			single++
		}
	}
	frac := float64(single) / float64(len(paras))
	return detect.Signal{Name: "one_line_paragraphs", Value: frac, AIProb: 0.4 + 0.4*frac, Weight: 0.3,
		Detail: fmt.Sprintf("%d of %d paragraphs are a single short sentence (§2)", single, len(paras))}, true
}

// punctuationVariety: newer models avoid dashes, semicolons, and
// parentheses, leaning on commas and periods more than people do
// (Wikipedia, em-dash section, September 2026 on).
func punctuationVariety(body string) (detect.Signal, bool) {
	var plain, varied int
	for _, r := range body {
		switch {
		case r == ',' || r == '.':
			plain++
		case strings.ContainsRune(";:()—–!?", r):
			varied++
		case unicode.IsPunct(r):
		}
	}
	if plain+varied < 15 {
		return detect.Signal{}, false
	}
	frac := float64(varied) / float64(plain+varied)
	return detect.Signal{Name: "punctuation_variety", Value: frac, AIProb: 0.6 - frac, Weight: 0.4,
		Detail: fmt.Sprintf("%.0f%% of sentence punctuation is something other than a comma or period", frac*100)}, true
}

// lexicalDiversity: moving-average type-token ratio over 100-word windows.
// Older models' repetition penalties produce "elegant variation".
func lexicalDiversity(ws []string) (detect.Signal, bool) {
	const window = 100
	if len(ws) < window {
		return detect.Signal{}, false
	}
	lower := make([]string, len(ws))
	for i, w := range ws {
		lower[i] = strings.ToLower(w)
	}
	var sum float64
	n := 0
	for start := 0; start+window <= len(lower); start += 25 {
		seen := map[string]bool{}
		for _, w := range lower[start : start+window] {
			seen[w] = true
		}
		sum += float64(len(seen)) / window
		n++
	}
	mattr := sum / float64(n)
	return detect.Signal{Name: "lexical_diversity", Value: mattr, AIProb: 0.5, Weight: 0.2,
		Detail: fmt.Sprintf("type-token ratio %.2f per 100-word window", mattr)}, true
}
