package overnight

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/split"
)

// FrozenGenres are the genres with frozen generator sets, and how many human
// sources each gets. Every frozen generator (Opus, Sonnet, Argon, Gemma)
// mirrors the same sources from the same cached briefs.
var FrozenGenres = map[string]int{"blog": 100, "memo": 100, "essay": 100, "pr": 150}

// FrozenGenerators are the generators with frozen, never-trained-on sets.
var FrozenGenerators = []string{
	"claude:claude-opus-5-5", "claude:claude-sonnet-5-5", "gemini:gemini-4-argon",
}

// Manifest records a checksum for every frozen file, by set. Sealed sets
// are complete and verified nightly; the final holdout is recorded but only
// verified by a manual run, because the loop never reads it.
type Manifest struct {
	Created string                `json:"created"`
	Sets    map[string]*FrozenSet `json:"sets"`
}

type FrozenSet struct {
	Sealed bool              `json:"sealed"`
	Manual bool              `json:"manual,omitempty"` // verified only by a manual run
	Files  map[string]string `json:"files"`            // path (relative to data root) -> sha256
	Note   string            `json:"note,omitempty"`
}

func (d Data) ManifestPath() string { return d.Path("corpus_frozen", "manifest.json") }

func (d Data) LoadManifest() (*Manifest, error) {
	b, err := os.ReadFile(d.ManifestPath())
	if err != nil {
		return nil, err
	}
	var m Manifest
	return &m, json.Unmarshal(b, &m)
}

func (d Data) SaveManifest(m *Manifest) error {
	b, _ := json.MarshalIndent(m, "", " ")
	return os.WriteFile(d.ManifestPath(), append(b, '\n'), 0o644)
}

func sha(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// collect checksums every .txt/.json under the given roots that match keep.
func (d Data) collect(keep func(rel, id string) bool, roots ...string) (map[string]string, error) {
	out := map[string]string{}
	for _, root := range roots {
		root, err := filepath.Abs(root)
		if err != nil {
			return nil, err
		}
		err = filepath.WalkDir(root, func(p string, de fs.DirEntry, err error) error {
			if err != nil {
				if os.IsNotExist(err) {
					return nil
				}
				return err
			}
			if de.IsDir() || !(strings.HasSuffix(p, ".txt") || strings.HasSuffix(p, ".md") || strings.HasSuffix(p, ".json")) {
				return nil
			}
			rel, _ := filepath.Rel(d.Root, p)
			id := strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))
			if keep != nil && !keep(rel, id) {
				return nil
			}
			s, err := sha(p)
			if err != nil {
				return err
			}
			out[rel] = s
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// testSplitFiles are the frozen test-split files of every training genre.
func (d Data) testSplitFiles() (map[string]string, error) {
	var roots []string
	for _, g := range Genres {
		roots = append(roots, filepath.Join(d.Dir(g), "human"), filepath.Join(d.Dir(g), "ai"))
	}
	return d.collect(func(rel, id string) bool { return split.Of(id) == "test" }, roots...)
}

// InitFrozen sets up the frozen sets: it moves fresh human sources for the
// frozen generator sets out of the main pools into corpus_frozen/, and
// records checksums for the test split, fresh pools, final holdout, and
// fixtures. It refuses to run twice.
func (d Data) InitFrozen(ctx context.Context, fixturesDir string, log io.Writer) (*Manifest, error) {
	if _, err := os.Stat(d.ManifestPath()); err == nil {
		return nil, fmt.Errorf("%s already exists; frozen sets are created once", d.ManifestPath())
	}
	frozen := &corpus.Store{Root: d.Path("corpus_frozen")}
	main := &corpus.Store{Root: d.Path("corpus")}
	for genre, n := range FrozenGenres {
		pool, err := main.Pool(genre)
		if err != nil {
			return nil, err
		}
		sort.Slice(pool, func(i, j int) bool { return split.Hash(pool[i].ID) < split.Hash(pool[j].ID) })
		moved := 0
		for _, sm := range pool {
			if moved >= n || ctx.Err() != nil {
				break
			}
			if words := len(strings.Fields(sm.Text)); genre != "pr" && (words < 150 || words > 1200) {
				continue
			}
			if err := corpus.FetchDiff(ctx, &sm); err != nil {
				continue // PR diffs that can't be fetched are skipped
			}
			if err := frozen.SaveHuman(sm); err != nil {
				return nil, err
			}
			os.Remove(d.Path("corpus", genre, "pool", sm.ID+".txt"))
			os.Remove(d.Path("corpus", genre, "poolsrc", sm.ID+".json"))
			moved++
		}
		fmt.Fprintf(log, "frozen %s: %d human sources moved out of corpus/%s/pool\n", genre, moved, genre)
	}

	m := &Manifest{Created: time.Now().UTC().Format(time.RFC3339), Sets: map[string]*FrozenSet{}}
	var err error
	add := func(name string, sealed, manual bool, note string, files map[string]string, e error) {
		if e != nil && err == nil {
			err = e
		}
		m.Sets[name] = &FrozenSet{Sealed: sealed, Manual: manual, Files: files, Note: note}
		fmt.Fprintf(log, "frozen set %-16s %5d files\n", name, len(files))
	}
	ts, e := d.testSplitFiles()
	add("test-split", true, false, "20% test split of every training genre, by id hash", ts, e)
	var pools []string
	for _, p := range d.FreshPools() {
		pools = append(pools, p)
	}
	fp, e := d.collect(func(rel, id string) bool { return !FinalHoldout(id) }, pools...)
	add("fresh-pools", true, false, "human-only false-positive pools, minus the final holdout", fp, e)
	fh, e := d.collect(func(rel, id string) bool { return FinalHoldout(id) }, pools...)
	add("final-holdout", true, true, "10% of each fresh pool; the loop never reads it", fh, e)
	fx, e := d.collect(nil, fixturesDir)
	add("fixtures", true, false, "regression fixtures and monitoring set", fx, e)
	hs, e := d.collect(nil, d.Path("corpus_frozen"))
	delete(hs, "corpus_frozen/manifest.json")
	add("frozen-sources", true, false, "human sources for the frozen generator sets", hs, e)
	if err != nil {
		return nil, err
	}
	return m, d.SaveManifest(m)
}

// SealGenerated records a frozen generator set's files (after its
// generation completes) and seals it.
func (d Data) SealGenerated(m *Manifest, gen string) (int, error) {
	files, err := d.collect(func(rel, id string) bool {
		return strings.Contains(rel, string(filepath.Separator)+"ai"+string(filepath.Separator)+gen+string(filepath.Separator))
	}, d.Path("corpus_frozen"))
	if err != nil {
		return 0, err
	}
	m.Sets["gen-"+gen] = &FrozenSet{Sealed: true, Files: files, Note: "frozen generator set; never trained on"}
	return len(files), d.SaveManifest(m)
}

// Verification is the result of checking sealed sets against the manifest.
type Verification struct {
	Set                     string
	Files                   int
	Changed, Missing, Added int
}

func (v Verification) OK() bool { return v.Changed == 0 && v.Missing == 0 && v.Added == 0 }

// Verify recomputes checksums for every sealed set (skipping manual-only
// sets unless includeManual) and reports differences.
func (d Data) Verify(m *Manifest, fixturesDir string, includeManual bool) ([]Verification, error) {
	var out []Verification
	names := make([]string, 0, len(m.Sets))
	for n := range m.Sets {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, name := range names {
		set := m.Sets[name]
		if !set.Sealed || (set.Manual && !includeManual) {
			continue
		}
		v := Verification{Set: name, Files: len(set.Files)}
		for rel, want := range set.Files {
			got, err := sha(filepath.Join(d.Root, rel))
			switch {
			case err != nil:
				v.Missing++
			case got != want:
				v.Changed++
			}
		}
		// New files appearing in a frozen location count too.
		var now map[string]string
		var err error
		var pools []string
		for _, p := range d.FreshPools() {
			pools = append(pools, p)
		}
		switch {
		case name == "test-split":
			now, err = d.testSplitFiles()
		case name == "fresh-pools":
			now, err = d.collect(func(rel, id string) bool { return !FinalHoldout(id) }, pools...)
		case name == "final-holdout":
			now, err = d.collect(func(rel, id string) bool { return FinalHoldout(id) }, pools...)
		case name == "fixtures" && fixturesDir != "":
			now, err = d.collect(nil, fixturesDir)
		case strings.HasPrefix(name, "gen-"):
			gen := strings.TrimPrefix(name, "gen-")
			now, err = d.collect(func(rel, id string) bool {
				return strings.Contains(rel, string(filepath.Separator)+"ai"+string(filepath.Separator)+gen+string(filepath.Separator))
			}, d.Path("corpus_frozen"))
		case name == "frozen-sources":
			now, err = d.collect(func(rel, id string) bool {
				return !strings.Contains(rel, string(filepath.Separator)+"ai"+string(filepath.Separator))
			}, d.Path("corpus_frozen"))
			delete(now, "corpus_frozen/manifest.json")
		}
		if err != nil {
			return nil, err
		}
		for rel := range now {
			if _, ok := set.Files[rel]; !ok {
				v.Added++
			}
		}
		out = append(out, v)
	}
	return out, nil
}

// GPTSplit reports the frozen GPT test-split mirrors and checks that no
// source text or PR diff behind them also appears in the train split
// (under another id). Briefs were written per source, so a shared brief
// implies a shared source.
type GPTSplit struct {
	TestMirrors map[string]int // genre -> test-split GPT mirrors
	Leaks       []string
}

func (d Data) CheckGPTSplit() (*GPTSplit, error) {
	out := &GPTSplit{TestMirrors: map[string]int{}}
	norm := func(s string) string {
		w := strings.Fields(strings.ToLower(s))
		h := sha256.Sum256([]byte(strings.Join(w[:min(200, len(w))], " ")))
		return hex.EncodeToString(h[:8])
	}
	for _, g := range Genres {
		srcs, err := d.Store(g).Sources(g.Kind)
		if err != nil {
			return nil, err
		}
		gptIDs := map[string]bool{}
		dirs, _ := filepath.Glob(filepath.Join(d.Dir(g), "ai", "gpt-*"))
		for _, dir := range dirs {
			files, _ := filepath.Glob(filepath.Join(dir, "*.txt"))
			for _, f := range files {
				id := strings.TrimSuffix(filepath.Base(f), ".txt")
				gptIDs[split.Base(id)] = true
				if split.Of(id) == "test" {
					out.TestMirrors[g.Name]++
				}
			}
		}
		seen := map[string]map[string][]string{} // hash -> split -> ids
		addHash := func(h, id string) {
			if seen[h] == nil {
				seen[h] = map[string][]string{}
			}
			seen[h][split.Of(id)] = append(seen[h][split.Of(id)], id)
		}
		for _, sm := range srcs {
			addHash("text:"+norm(sm.Text), sm.ID)
			if diff := sm.Context["diff"]; diff != "" {
				addHash("diff:"+norm(diff), sm.ID)
			}
		}
		for h, bySplit := range seen {
			if len(bySplit["train"]) == 0 || len(bySplit["test"]) == 0 {
				continue
			}
			involved := false
			for _, id := range append(bySplit["train"], bySplit["test"]...) {
				involved = involved || gptIDs[id]
			}
			if involved {
				out.Leaks = append(out.Leaks, fmt.Sprintf("%s %s: train %v, test %v", g.Name, strings.SplitN(h, ":", 2)[0], bySplit["train"], bySplit["test"]))
			}
		}
	}
	sort.Strings(out.Leaks)
	return out, nil
}

// RedrawHoldout recomputes the fresh-pools and final-holdout sets after
// the holdout draw changed (FNV to SHA-256: the FNV holdout fell entirely
// inside the calibration half). It runs once.
func (d Data) RedrawHoldout(m *Manifest) (pools, holdout int, err error) {
	if s := m.Sets["final-holdout"]; s != nil && strings.Contains(s.Note, "SHA-256") {
		return 0, 0, fmt.Errorf("the final holdout was already redrawn")
	}
	return d.ResealPools(m)
}

// ResealPools recomputes the fresh-pools and final-holdout sets from the
// pools on disk: after the holdout draw changed, or when a new pool is
// added (existing files must be unchanged; Verify first).
func (d Data) ResealPools(m *Manifest) (pools, holdout int, err error) {
	var roots []string
	for _, p := range d.FreshPools() {
		roots = append(roots, p)
	}
	fp, err := d.collect(func(rel, id string) bool { return !FinalHoldout(id) }, roots...)
	if err != nil {
		return 0, 0, err
	}
	fh, err := d.collect(func(rel, id string) bool { return FinalHoldout(id) }, roots...)
	if err != nil {
		return 0, 0, err
	}
	m.Sets["fresh-pools"] = &FrozenSet{Sealed: true, Files: fp, Note: "human-only false-positive pools, minus the final holdout"}
	m.Sets["final-holdout"] = &FrozenSet{Sealed: true, Manual: true, Files: fh,
		Note: "10% of each fresh pool by SHA-256 (redrawn " + time.Now().Format("2006-01-02") + "); the loop never reads it. " +
			"Production's shipped cutoffs were fit on all pool text, so this holdout is untouched only for candidates."}
	return len(fp), len(fh), d.SaveManifest(m)
}
