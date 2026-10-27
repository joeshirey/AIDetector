package corpus

import (
	"encoding/json"
	"errors"
	"hash/fnv"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/joeshirey/AIDetector/internal/text"
)

// Store lays out a corpus on disk:
//
//	<root>/<kind>/human/<id>.txt          human text
//	<root>/<kind>/src/<id>.json           sample + generation context
//	<root>/<kind>/ai/<generator>/<id>.txt AI mirror of the same source
//	<root>/<kind>/manifest.jsonl          generation log
//	<root>/<kind>/pool/<id>.txt           human-only reserve, never trained on
//	<root>/<kind>/poolsrc/<id>.json       sample + context for pool texts
type Store struct {
	Root string
	mu   sync.Mutex
}

func (s *Store) dir(kind string, parts ...string) string {
	return filepath.Join(append([]string{s.Root, kind}, parts...)...)
}

// Has reports whether id is already in the corpus, as a training sample or
// in the reserve pool.
func (s *Store) Has(kind, id string) bool {
	for _, p := range []string{s.dir(kind, "src", id+".json"), s.dir(kind, "poolsrc", id+".json")} {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

func (s *Store) SavePool(sm Sample) error {
	if err := writeFile(s.dir(sm.Kind, "pool", sm.ID+".txt"), sm.Text); err != nil {
		return err
	}
	b, err := json.MarshalIndent(sm, "", "  ")
	if err != nil {
		return err
	}
	return writeFile(s.dir(sm.Kind, "poolsrc", sm.ID+".json"), string(b))
}

func (s *Store) Pool(kind string) ([]Sample, error) { return s.readSamples(s.dir(kind, "poolsrc")) }

// Promote moves a pool sample into the training corpus so it gets mirrored.
func (s *Store) Promote(sm Sample) error {
	if err := s.SaveHuman(sm); err != nil {
		return err
	}
	os.Remove(s.dir(sm.Kind, "pool", sm.ID+".txt"))
	return os.Remove(s.dir(sm.Kind, "poolsrc", sm.ID+".json"))
}

func (s *Store) SaveHuman(sm Sample) error {
	if err := writeFile(s.dir(sm.Kind, "human", sm.ID+".txt"), sm.Text); err != nil {
		return err
	}
	b, err := json.MarshalIndent(sm, "", "  ")
	if err != nil {
		return err
	}
	return writeFile(s.dir(sm.Kind, "src", sm.ID+".json"), string(b))
}

func (s *Store) Sources(kind string) ([]Sample, error) { return s.readSamples(s.dir(kind, "src")) }

func (s *Store) readSamples(dir string) ([]Sample, error) {
	var out []Sample
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".json") {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		var sm Sample
		if err := json.Unmarshal(b, &sm); err != nil {
			return err
		}
		out = append(out, sm)
		return nil
	})
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	return out, err
}

func (s *Store) AIPath(kind, gen, id string) string { return s.dir(kind, "ai", gen, id+".txt") }

func (s *Store) HasAI(kind, id string) bool {
	m, _ := filepath.Glob(s.dir(kind, "ai", "*", id+".txt"))
	return len(m) > 0
}

type ManifestEntry struct {
	ID        string    `json:"id"`
	Generator string    `json:"generator"`
	Variant   int       `json:"variant"`
	Split     string    `json:"split"`
	Error     string    `json:"error,omitempty"`
	At        time.Time `json:"at"`
}

func (s *Store) Log(kind string, e ManifestEntry) error { return s.logTo(kind, "manifest.jsonl", e) }

func (s *Store) logTo(kind, name string, e any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.dir(kind, name), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(e)
}

// Renormalize re-applies the current normalization to every human and AI
// text of a kind, so corpus files match what new harvests would produce.
func (s *Store) Renormalize(kind string) (changed int, err error) {
	paths, _ := filepath.Glob(s.dir(kind, "human", "*.txt"))
	ai, _ := filepath.Glob(s.dir(kind, "ai", "*", "*.txt"))
	for _, p := range append(paths, ai...) {
		b, err := os.ReadFile(p)
		if err != nil {
			return changed, err
		}
		orig := strings.TrimSpace(string(b))
		norm := text.Normalize(kind, orig)
		if norm == orig {
			continue
		}
		if err := writeFile(p, norm); err != nil {
			return changed, err
		}
		changed++
	}
	// Keep src/*.json in step with human/*.txt.
	srcs, err := s.Sources(kind)
	if err != nil {
		return changed, err
	}
	for _, sm := range srcs {
		if norm := text.Normalize(kind, sm.Text); norm != sm.Text {
			sm.Text = norm
			if err := s.SaveHuman(sm); err != nil {
				return changed, err
			}
		}
	}
	return changed, nil
}

func hashN(id string, n int) int {
	h := fnv.New32a()
	h.Write([]byte("v:" + id))
	return int(h.Sum32() % uint32(n))
}

func writeFile(path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content+"\n"), 0o644)
}
