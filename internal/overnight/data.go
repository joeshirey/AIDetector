// Package overnight is the unattended improvement loop: each night it mines
// the detector's mistakes, generates targeted training text, retrains a
// candidate locally, gates it against frozen evaluation sets, and writes a
// report. It never commits, pushes, promotes models, or changes labels or
// cutoffs; Joe reviews the report and promotes by hand.
package overnight

import (
	"crypto/sha256"
	"encoding/binary"
	"path/filepath"
	"strings"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/split"
)

// Data locates the corpus directories (in the main checkout; the loop runs
// from a separate worktree).
type Data struct{ Root string }

func (d Data) Path(parts ...string) string {
	return filepath.Join(append([]string{d.Root}, parts...)...)
}

// Genre is a training genre: four prose genres and PR descriptions.
type Genre struct {
	Name  string      // blog, memo, essay, techblog, pr
	Store string      // corpus root holding it
	Kind  string      // corpus kind directory under Store
	Det   detect.Kind // detector kind
}

var Genres = []Genre{
	{"blog", "corpus", "blog", detect.KindProse},
	{"memo", "corpus", "memo", detect.KindProse},
	{"essay", "corpus", "essay", detect.KindProse},
	{"techblog", "corpus_tech", "blog", detect.KindProse},
	{"pr", "corpus", "pr", detect.KindPR},
}

func GenreByName(n string) (Genre, bool) {
	for _, g := range Genres {
		if g.Name == n {
			return g, true
		}
	}
	return Genre{}, false
}

func (d Data) Store(g Genre) *corpus.Store { return &corpus.Store{Root: d.Path(g.Store)} }

// Dir is the genre's corpus directory (with human/ and ai/).
func (d Data) Dir(g Genre) string { return d.Path(g.Store, g.Kind) }

// TrainDirs returns the training directories for a detector kind.
func (d Data) TrainDirs(k detect.Kind) []string {
	var out []string
	for _, g := range Genres {
		if g.Det == k {
			out = append(out, d.Dir(g))
		}
	}
	return out
}

// FreshPools are the human-only false-positive pools: frozen, never trained
// on, never promoted from. The final holdout is carved out of each.
func (d Data) FreshPools() map[string]string {
	return map[string]string{
		"fresh-blog":     d.Path("corpus_fresh", "blog", "pool"),
		"fresh-memo":     d.Path("corpus_fresh", "memo", "pool"),
		"fresh-essay":    d.Path("corpus_fresh", "essay", "pool"),
		"fresh-pr":       d.Path("corpus_fresh", "pr", "pool"),
		"pr-pool":        d.Path("corpus_prpool", "pr", "pool"), // 99 more repos (2026-10-29), for PR cutoffs
		"tech-pool":      d.Path("corpus_tech", "blog", "pool"),
		"nontech-medium": d.Path("corpus_nontech", "blog", "pool"),
	}
}

// PoolGenre maps a fresh pool to the genre whose cutoffs it measures.
var PoolGenre = map[string]string{
	"fresh-blog": "blog", "fresh-memo": "memo", "fresh-essay": "essay", "fresh-pr": "pr", "pr-pool": "pr",
	"tech-pool": "techblog", "nontech-medium": "blog",
}

// ModernPools are known-human texts written after ChatGPT's launch (so
// never trained on), scored to see whether modern human writing, which has
// absorbed some LLM phrasing, drifts toward false positives. Report-only.
func (d Data) ModernPools() map[string]string {
	return map[string]string{
		"fowler": d.Path("corpus_modern", "blog", "pool"), // martinfowler.com, 2023 on, by Fowler alone
	}
}

// PoolKind is the detector kind a fresh pool is scored as.
func PoolKind(name string) detect.Kind {
	if PoolGenre[name] == "pr" {
		return detect.KindPR
	}
	return detect.KindProse
}

// FinalHoldout reports whether a fresh-pool text belongs to the final
// holdout: 10% of each pool, chosen by a hash independent of the train/test
// and calibration splits. The loop never reads these files; only Joe or a
// manual run checks them.
func FinalHoldout(id string) bool {
	h := sha256.Sum256([]byte("final-holdout:" + split.Base(id)))
	return binary.BigEndian.Uint32(h[:4])%10 == 0
}

// Trainable reports whether a training-corpus sample may be used for
// training: it must be train split. (Test-split files are frozen.)
func Trainable(id string) bool { return split.Of(id) == "train" }

// GeneratorFamily groups generator directories into model families.
func GeneratorFamily(gen string) string {
	g := strings.SplitN(gen, "@", 2)[0]
	switch {
	case strings.HasPrefix(g, "claude"):
		return "claude"
	case strings.HasPrefix(g, "gemini"):
		return "gemini"
	case strings.HasPrefix(g, "gpt"):
		return "gpt"
	case strings.HasPrefix(g, "qwen"):
		return "qwen"
	case strings.HasPrefix(g, "gemma"):
		return "gemma"
	}
	return "other"
}
