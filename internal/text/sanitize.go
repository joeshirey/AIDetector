package text

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Tampering counts characters that are invisible or disguised. They're used
// to evade detectors, and they also break detectors on innocent human text,
// so they're removed before scoring and reported separately.
type Tampering struct {
	ZeroWidth  int `json:"zero_width"`
	Homoglyphs int `json:"homoglyphs"`
}

func (t Tampering) Any() bool { return t.ZeroWidth > 0 || t.Homoglyphs > 0 }

var (
	zeroWidthRE = regexp.MustCompile(`[\x{200B}\x{200C}\x{200D}\x{2060}\x{FEFF}\x{00AD}\x{180E}]`)
	mixedWordRE = regexp.MustCompile(`[\p{L}]+`)
)

// Cyrillic and Greek letters that render like Latin ones.
var homoglyphs = map[rune]rune{
	'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j', 'ѕ': 's', 'һ': 'h', 'ԁ': 'd',
	'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'І': 'I', 'Ј': 'J', 'Ѕ': 'S',
	'α': 'a', 'ο': 'o', 'ρ': 'p', 'ν': 'v', 'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Η': 'H', 'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Χ': 'X', 'Ζ': 'Z',
}

// Sanitize removes zero-width characters, maps look-alike letters inside
// otherwise-Latin words back to Latin, and applies NFKC normalization
// (fullwidth letters, ligatures, non-breaking spaces).
func Sanitize(s string) (string, Tampering) {
	var t Tampering
	t.ZeroWidth = len(zeroWidthRE.FindAllStringIndex(s, -1))
	s = zeroWidthRE.ReplaceAllString(s, "")
	s = mixedWordRE.ReplaceAllStringFunc(s, func(w string) string {
		latin, lookalike := false, false
		for _, r := range w {
			if _, ok := homoglyphs[r]; ok {
				lookalike = true
			} else if unicode.In(r, unicode.Latin) {
				latin = true
			}
		}
		if !latin || !lookalike { // leave genuine Cyrillic or Greek words alone
			return w
		}
		return strings.Map(func(r rune) rune {
			if l, ok := homoglyphs[r]; ok {
				t.Homoglyphs++
				return l
			}
			return r
		}, w)
	})
	return norm.NFKC.String(s), t
}
