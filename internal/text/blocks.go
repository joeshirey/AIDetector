package text

import (
	"regexp"
	"strings"
)

// Block is an embedded or quoted passage that may not be the author's own
// writing: a pasted Markdown document (like an agent-generated SKILL.md) or
// a long quotation.
type Block struct {
	What    string // "embedded Markdown block" or "quoted passage"
	Words   int
	Preview string
	Without string // the full text with this block removed
}

var (
	mdLineRE = regexp.MustCompile(`^\s*(#{1,6}\s|>\s?|(-{3,}|\*{3,})\s*$|([-*+]|\d+[.)])\s+(\*\*|\[|` + "`" + `))|\*\*[^*]+\*\*|\]\(|` + "`[^`]+`")
	quoteRE  = regexp.MustCompile(`^\s*>`)
)

// EmbeddedBlocks finds up to two candidate blocks, largest first. A Markdown
// block is a run of at least three paragraphs dense with Markdown syntax,
// totaling 60+ words; a quoted passage is 40+ words of ">"-quoted lines.
func EmbeddedBlocks(s string) []Block {
	paras := regexp.MustCompile(`\n\s*\n`).Split(strings.TrimSpace(s), -1)
	var blocks []Block
	add := func(what string, from, to int) {
		body := strings.Join(paras[from:to], "\n\n")
		words := len(Words(body))
		if (what == "quoted passage" && words < 40) || (what != "quoted passage" && words < 60) {
			return
		}
		rest := append(append([]string{}, paras[:from]...), paras[to:]...)
		blocks = append(blocks, Block{What: what, Words: words, Preview: preview(body), Without: strings.Join(rest, "\n\n")})
	}
	for i := 0; i < len(paras); {
		if !mdParagraph(paras[i]) {
			i++
			continue
		}
		j := i
		for j < len(paras) && mdParagraph(paras[j]) {
			j++
		}
		if j-i >= 3 {
			add("embedded Markdown block", i, j)
		}
		i = j
	}
	for i := 0; i < len(paras); {
		if !allQuoted(paras[i]) {
			i++
			continue
		}
		j := i
		for j < len(paras) && allQuoted(paras[j]) {
			j++
		}
		add("quoted passage", i, j)
		i = j
	}
	// Largest first; a block covering most of the text isn't "embedded".
	total := len(Words(s))
	var out []Block
	for len(blocks) > 0 && len(out) < 2 {
		best := 0
		for k := range blocks {
			if blocks[k].Words > blocks[best].Words {
				best = k
			}
		}
		if blocks[best].Words*10 < total*7 {
			out = append(out, blocks[best])
		}
		blocks = append(blocks[:best], blocks[best+1:]...)
	}
	return out
}

// mdParagraph: most lines carry Markdown syntax, or the paragraph is front
// matter, a fence, or a rule.
func mdParagraph(p string) bool {
	lines := strings.Split(strings.TrimSpace(p), "\n")
	md := 0
	for _, l := range lines {
		if mdLineRE.MatchString(l) || fenceLineRE.MatchString(l) || yamlKeyRE.MatchString(l) {
			md++
		}
	}
	return md*2 >= len(lines)
}

func allQuoted(p string) bool {
	for _, l := range strings.Split(strings.TrimSpace(p), "\n") {
		if !quoteRE.MatchString(l) {
			return false
		}
	}
	return true
}

func preview(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 60 {
		return string(r[:60]) + "…"
	}
	return s
}
