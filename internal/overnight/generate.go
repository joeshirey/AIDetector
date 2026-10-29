package overnight

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/split"
)

// Style is how a generation prompt asks for text.
type Style string

const (
	StylePlain    Style = "plain"    // write from a brief (or a diff, for PRs)
	StylePersona  Style = "persona"  // brief plus a persona or audience
	StyleEdit     Style = "edit"     // edit a human draft (AI-assisted)
	StyleContinue Style = "continue" // continue a human opening
)

var Styles = []Style{StylePlain, StylePersona, StyleEdit, StyleContinue}

// TrainStyles are the styles the training track generates. Edits of human
// drafts are mostly human text, so they're eval-only like the polished set
// (Joe, 2026-10-29): they raised human false positives when trained on.
var TrainStyles = []Style{StylePlain, StylePersona, StyleContinue}

// Bands are output length bands in words. PR descriptions are much shorter
// than prose, so they use their own bands.
var (
	ProseBands = [][2]int{{150, 300}, {300, 600}, {600, 1200}}
	PRBands    = [][2]int{{25, 150}, {150, 300}, {300, 1200}}
)

func bandsFor(g Genre) [][2]int {
	if g.Name == "pr" {
		return PRBands
	}
	return ProseBands
}

// BandOf returns the band index for a word count, or -1.
func BandOf(g Genre, words int) int {
	for i, b := range bandsFor(g) {
		if words >= b[0] && words < b[1] {
			return i
		}
	}
	return -1
}

// Cell is one bandit arm: genre x length band x prompt style x generator.
type Cell struct {
	Genre string `json:"genre"`
	Band  int    `json:"band"`
	Style Style  `json:"style"`
	Gen   string `json:"gen"`
}

func (c Cell) Key() string { return fmt.Sprintf("%s|%d|%s|%s", c.Genre, c.Band, c.Style, c.Gen) }

// Source is a human training sample that generations are made from.
type Source struct {
	Genre  Genre
	Sample corpus.Sample
	Words  int
	Band   int
}

// Sources returns a genre's train-split human samples, by length band.
// Test-split samples are frozen and never get new mirrors.
func (d Data) Sources(g Genre) (map[int][]Source, error) {
	srcs, err := d.Store(g).Sources(g.Kind)
	if err != nil {
		return nil, err
	}
	out := map[int][]Source{}
	for _, sm := range srcs {
		if !Trainable(sm.ID) {
			continue
		}
		w := len(strings.Fields(sm.Text))
		if b := BandOf(g, w); b >= 0 {
			out[b] = append(out[b], Source{Genre: g, Sample: sm, Words: w, Band: b})
		}
	}
	return out, nil
}

// SampleName names a generated file: the source id for plain mirrors (the
// existing convention), "<id>~<style>" otherwise. split.Base maps variants
// back to their source, so they share its train/test split.
func SampleName(id string, s Style) string {
	if s == StylePlain {
		return id
	}
	return id + "~" + string(s)
}

// OutputPath is where a generation for (source, generator, style) lives. It
// doubles as the cache: an existing file is never regenerated.
// Edits go to ai_edited/, outside ai/, so fit and eval never see them.
func (d Data) OutputPath(g Genre, gen, id string, s Style) string {
	return filepath.Join(d.Dir(g), aiDir(s), gen, SampleName(id, s)+".txt")
}

func aiDir(s Style) string {
	if s == StyleEdit {
		return "ai_edited"
	}
	return "ai"
}

// Briefer writes and caches content briefs for prose sources, so every
// generator writes from the same brief. Briefs live outside the corpus so
// frozen files are never modified.
type Briefer struct {
	Gen corpus.Generator
	Dir string
	mu  sync.Mutex
}

func needsBrief(g Genre, s Style) bool {
	return g.Name != "pr" && (s == StylePlain || s == StylePersona)
}

func briefKind(g Genre) string {
	if g.Name == "techblog" {
		return "blog"
	}
	return g.Name
}

func (b *Briefer) Brief(ctx context.Context, g Genre, sm corpus.Sample) (string, error) {
	path := filepath.Join(b.Dir, g.Name, sm.ID+".txt")
	if data, err := os.ReadFile(path); err == nil {
		return string(data), nil
	}
	tmpl := corpus.BriefPrompt(briefKind(g))
	if tmpl == "" {
		return "", fmt.Errorf("no brief prompt for %s", g.Name)
	}
	out, err := b.Gen.Generate(ctx, corpus.Fill(tmpl, map[string]string{"text": sm.Text}))
	if err != nil {
		return "", err
	}
	out = strings.TrimSpace(out)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	return out, os.WriteFile(path, []byte(out+"\n"), 0o644)
}

var nouns = map[string]string{"blog": "blog post", "techblog": "technical blog post", "memo": "work email", "essay": "essay"}

var (
	personas = map[string][]string{
		"blog":     {"a small business owner", "a parent who writes on weekends", "a career coach", "a nurse"},
		"techblog": {"a staff engineer", "a developer advocate", "an SRE lead", "a solo founder"},
		"memo":     {"an operations manager", "a finance director", "a project lead", "an office administrator"},
		"essay":    {"a high school junior", "a first-year college student", "an adult returning to school"},
		"pr":       {"a senior engineer", "a new team member", "a maintainer of the project"},
	}
	audiences = map[string][]string{
		"blog":     {"first-time readers", "people who are skeptical of the idea", "busy professionals"},
		"techblog": {"engineers new to the tool", "engineering managers", "people migrating from another platform"},
		"memo":     {"the whole department", "two senior executives", "an outside vendor"},
		"essay":    {"a teacher grading for argument", "a scholarship committee"},
		"pr":       {"reviewers on a busy team", "someone reading the history a year from now"},
	}
)

// opening returns the first paragraphs of a human text, up to about a
// quarter of it (at most ~150 words), for continuation prompts.
func opening(text string) (string, int) {
	paras := strings.Split(strings.TrimSpace(text), "\n\n")
	total := len(strings.Fields(text))
	var out []string
	n := 0
	for _, p := range paras {
		out = append(out, p)
		n += len(strings.Fields(p))
		if n*4 >= total || n >= 150 {
			break
		}
	}
	return strings.Join(out, "\n\n"), n
}

// BuildPrompt builds a generation prompt. Every prompt must pass
// corpus.CheckPrompt (no instructions to sound human or avoid AI tells).
func BuildPrompt(g Genre, s Style, sm corpus.Sample, brief string, rng *rand.Rand) (prompt string, words int, err error) {
	words = len(strings.Fields(sm.Text))
	pick := func(xs []string) string { return xs[rng.IntN(len(xs))] }
	v := map[string]string{"noun": nouns[g.Name], "brief": brief, "text": sm.Text, "words": fmt.Sprint(words),
		"title": sm.Context["title"], "diff": sm.Context["diff"], "repo": sm.Context["repo"]}
	if g.Name != "pr" {
		v["persona"], v["audience"] = pick(personas[g.Name]), pick(audiences[g.Name])
	} else {
		v["persona"], v["audience"] = pick(personas["pr"]), pick(audiences["pr"])
	}
	var tmpl string
	switch {
	case g.Name == "pr" && s == StylePlain:
		tmpl = "Write a pull request description for this change.\n\nTitle: {title}\n\nDiff:\n{diff}\n\nOutput only the PR description in Markdown (no title)."
	case g.Name == "pr" && s == StylePersona:
		tmpl = "Write the pull request description {persona} would post for this change, for {audience}.\n\nTitle: {title}\n\nDiff:\n{diff}\n\nOutput only the description."
	case g.Name == "pr" && s == StyleEdit:
		tmpl = "Here's my draft pull request description. Clean it up and make it clearer. Return only the revised description.\n\n{text}"
	case g.Name == "pr" && s == StyleContinue:
		op, n := opening(sm.Text)
		v["opening"], words = op, max(words-n, 20)
		tmpl = "Here's the start of a pull request description and the diff. Finish the description. Return only what comes after my opening.\n\nOpening:\n{opening}\n\nDiff:\n{diff}"
	case s == StylePlain:
		tmpl = "Write a {noun} of about {words} words.\n\nBrief:\n{brief}\n\nOutput only the {noun}."
	case s == StylePersona:
		tmpl = "Write a {noun} of about {words} words for {audience}, from the perspective of {persona}.\n\nBrief:\n{brief}\n\nOutput only the {noun}."
	case s == StyleEdit:
		tmpl = "Here's a draft {noun}. Edit it to improve clarity, flow, and wording. Keep the meaning and roughly the same length. Return only the revised text.\n\n{text}"
	case s == StyleContinue:
		op, n := opening(sm.Text)
		v["opening"], words = op, max(words-n, 50)
		v["words"] = fmt.Sprint(words)
		tmpl = "Here's the beginning of a {noun}. Continue it for about {words} more words on the same topic. Return only the continuation, not the part I gave you.\n\n{opening}"
	default:
		return "", 0, fmt.Errorf("unknown style %q", s)
	}
	prompt = corpus.Fill(tmpl, v)
	// Check our instructions, not the human content being mirrored (a post
	// about the Humanizer library is fine; asking for human-sounding text isn't).
	instr := map[string]string{}
	for k, val := range v {
		instr[k] = val
	}
	for _, k := range []string{"text", "brief", "diff", "opening", "title", "repo"} {
		instr[k] = "<content>"
	}
	return prompt, words, corpus.CheckPrompt(corpus.Fill(tmpl, instr))
}

// ReqMeta identifies a generation request; it's the batch custom ID.
type ReqMeta struct {
	Genre string `json:"genre"`
	ID    string `json:"id"`
	Style Style  `json:"style"`
	Band  int    `json:"band"`
	Track string `json:"track"` // "train", or "frozen" for frozen generator sets
}

func (m ReqMeta) Encode() string {
	return strings.Join([]string{m.Track, m.Genre, string(m.Style), fmt.Sprint(m.Band), m.ID}, "|")
}

func DecodeReq(s string) (ReqMeta, error) {
	p := strings.SplitN(s, "|", 5)
	if len(p) != 5 {
		return ReqMeta{}, fmt.Errorf("bad request id %q", s)
	}
	var band int
	fmt.Sscan(p[3], &band)
	return ReqMeta{Track: p[0], Genre: p[1], Style: Style(p[2]), Band: band, ID: p[4]}, nil
}

// cleanKind is the corpus.Clean kind for a genre.
func cleanKind(g Genre) string {
	if g.Name == "techblog" {
		return "blog"
	}
	return g.Name
}

// GenRecord logs one generated file, for bandit feedback and reports.
type GenRecord struct {
	Path  string `json:"path"`
	Cell  Cell   `json:"cell"`
	Track string `json:"track"`
	Night string `json:"night"`
	Mode  string `json:"mode"`
}

// WriteOutput cleans and saves a generated text. It returns false if the
// text was too short to keep.
func (d Data) WriteOutput(g Genre, gen string, m ReqMeta, text, title string, outRoot string) (string, bool, error) {
	text = corpus.Clean(cleanKind(g), text, title)
	minWords := 30
	if g.Name == "pr" {
		minWords = 10
	}
	if len(strings.Fields(text)) < minWords {
		return "", false, nil
	}
	path := d.OutputPath(g, gen, m.ID, m.Style)
	if outRoot != "" { // frozen sets live under corpus_frozen/<genre>/ai/<gen>/
		path = filepath.Join(outRoot, g.Name, aiDir(m.Style), gen, SampleName(m.ID, m.Style)+".txt")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", false, err
	}
	return path, true, os.WriteFile(path, []byte(strings.TrimSpace(text)+"\n"), 0o644)
}

// Bandit is Thompson sampling over cells on the miss rate (AI scored below
// 60), with a floor of uniform sampling so no cell is starved.
type Bandit struct {
	Cells map[string]*Stat `json:"cells"`
	Gens  map[string]*Stat `json:"gens"`
	rng   *rand.Rand
	mu    sync.Mutex // planners for different generators share one bandit
}

type Stat struct {
	Miss int `json:"miss"`
	N    int `json:"n"`
}

func (s *Stat) Rate() float64 {
	if s == nil || s.N == 0 {
		return math.NaN()
	}
	return float64(s.Miss) / float64(s.N)
}

func LoadBandit(path string, seed uint64) *Bandit {
	b := &Bandit{Cells: map[string]*Stat{}, Gens: map[string]*Stat{}}
	if data, err := os.ReadFile(path); err == nil {
		json.Unmarshal(data, b)
	}
	if b.Cells == nil {
		b.Cells = map[string]*Stat{}
	}
	if b.Gens == nil {
		b.Gens = map[string]*Stat{}
	}
	b.rng = rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	return b
}

func (b *Bandit) seed() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rng.Uint64()
}

func (b *Bandit) Save(path string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	data, _ := json.MarshalIndent(b, "", " ")
	return os.WriteFile(path, data, 0o644)
}

func (b *Bandit) Observe(c Cell, miss bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, m := range []struct {
		tbl map[string]*Stat
		key string
	}{{b.Cells, c.Key()}, {b.Gens, c.Gen}} {
		s := m.tbl[m.key]
		if s == nil {
			s = &Stat{}
			m.tbl[m.key] = s
		}
		s.N++
		if miss {
			s.Miss++
		}
	}
}

// sampleBeta draws from Beta(a, b) via two gamma draws.
func (b *Bandit) sampleBeta(a, c float64) float64 {
	x, y := b.gamma(a), b.gamma(c)
	return x / (x + y)
}

// gamma draws from Gamma(k, 1) (Marsaglia and Tsang).
func (b *Bandit) gamma(k float64) float64 {
	if k < 1 {
		return b.gamma(k+1) * math.Pow(b.rng.Float64(), 1/k)
	}
	d := k - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := b.rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		if u := b.rng.Float64(); math.Log(u) < 0.5*x*x+d-d*v+d*math.Log(v) {
			return d * v
		}
	}
}

// Pick chooses a cell: uniformly with probability floor, otherwise the
// cell with the highest sampled miss rate.
func (b *Bandit) Pick(cells []Cell, floor float64) Cell {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.rng.Float64() < floor {
		return cells[b.rng.IntN(len(cells))]
	}
	best, bestV := cells[0], -1.0
	for _, c := range cells {
		s := b.Cells[c.Key()]
		miss, n := 0, 0
		if s != nil {
			miss, n = s.Miss, s.N
		}
		if v := b.sampleBeta(float64(1+miss), float64(1+n-miss)); v > bestV {
			best, bestV = c, v
		}
	}
	return best
}

// GenShares moves budget between paid generators by sampled miss rate,
// keeping each between its floor and cap.
func (b *Bandit) GenShares(base map[string]float64, floor, capMult float64) map[string]float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	w := map[string]float64{}
	total := 0.0
	for g, share := range base {
		s := b.Gens[g]
		miss, n := 0, 0
		if s != nil {
			miss, n = s.Miss, s.N
		}
		w[g] = share * (0.5 + b.sampleBeta(float64(1+miss), float64(1+n-miss)))
		total += w[g]
	}
	out := map[string]float64{}
	for g := range w {
		out[g] = min(max(w[g]/total, floor), base[g]*capMult)
	}
	// Renormalize after clamping.
	sum := 0.0
	for _, v := range out {
		sum += v
	}
	for g := range out {
		out[g] /= sum
	}
	return out
}

// Planner picks (cell, source) pairs for a generator, skipping anything
// already generated (the cache) and any source already planned tonight.
type Planner struct {
	Data    Data
	Sources map[string]map[int][]Source // genre -> band -> sources
	Bandit  *Bandit
	Floor   float64
	planned map[string]bool
	rng     *rand.Rand
	mu      sync.Mutex
}

func NewPlanner(d Data, b *Bandit, genres []Genre) (*Planner, error) {
	p := &Planner{Data: d, Sources: map[string]map[int][]Source{}, Bandit: b, Floor: 0.2, planned: map[string]bool{},
		rng: rand.New(rand.NewPCG(b.seed(), 3))}
	for _, g := range genres {
		s, err := d.Sources(g)
		if err != nil {
			return nil, err
		}
		p.Sources[g.Name] = s
	}
	return p, nil
}

func (p *Planner) cellsFor(gen string) []Cell {
	var out []Cell
	names := make([]string, 0, len(p.Sources))
	for n := range p.Sources {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		for band, srcs := range p.Sources[n] {
			if len(srcs) == 0 {
				continue
			}
			for _, s := range TrainStyles {
				out = append(out, Cell{Genre: n, Band: band, Style: s, Gen: gen})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key() < out[j].Key() })
	return out
}

// Next returns the next (cell, source) to generate for gen, or false when
// nothing uncached is left.
func (p *Planner) Next(gen string) (Cell, Source, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	cells := p.cellsFor(gen)
	if len(cells) == 0 {
		return Cell{}, Source{}, false
	}
	for range 50 {
		c := p.Bandit.Pick(cells, p.Floor)
		srcs := p.Sources[c.Genre][c.Band]
		for range 20 {
			s := srcs[p.rng.IntN(len(srcs))]
			key := c.Gen + "|" + string(c.Style) + "|" + s.Sample.ID
			if p.planned[key] {
				continue
			}
			if _, err := os.Stat(p.Data.OutputPath(s.Genre, gen, s.Sample.ID, c.Style)); err == nil {
				p.planned[key] = true // cached: never regenerate
				continue
			}
			p.planned[key] = true
			return c, s, true
		}
	}
	return Cell{}, Source{}, false
}

// thinkingAllowance estimates hidden reasoning tokens per call, from the
// effort experiment (2026-10-28, 50 samples each). Claude batch output
// includes its thinking, so Claude's allowance is the extra output over
// low effort.
func thinkingAllowance(s corpus.Spec) int {
	low := s.Effort == "low"
	switch m := s.Model; {
	case strings.Contains(m, "flash"):
		if low {
			return 150
		}
		return 4200
	case strings.Contains(m, "argon"):
		if low {
			return 300
		}
		return 3000 // not measured yet; assume Flash-like
	case strings.Contains(m, "haiku"):
		if low {
			return 300
		}
		return 1000
	case strings.Contains(m, "sonnet"), strings.Contains(m, "opus"):
		if low {
			return 200
		}
		return 800
	}
	return 600
}

// EstimateUsage guesses a request's tokens before it runs.
func EstimateUsage(s corpus.Spec, prompt string, words int) corpus.Usage {
	return corpus.Usage{Input: len(prompt)/4 + 50, Output: int(float64(words)*1.4) + 50, Thinking: thinkingAllowance(s)}
}

// stableSeed derives a per-night RNG seed.
func stableSeed(night string) uint64 { return uint64(split.Hash("night:" + night)) }

// deadlineLeft is the time remaining before a deadline.
func deadlineLeft(deadline time.Time) time.Duration { return time.Until(deadline) }
