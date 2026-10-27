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
)

// Store lays out a corpus on disk:
//
//	<root>/<kind>/human/<id>.txt          human text
//	<root>/<kind>/src/<id>.json           sample + generation context
//	<root>/<kind>/ai/<generator>/<id>.txt AI mirror of the same source
//	<root>/<kind>/manifest.jsonl          generation log
type Store struct {
	Root string
	mu   sync.Mutex
}

func (s *Store) dir(kind string, parts ...string) string {
	return filepath.Join(append([]string{s.Root, kind}, parts...)...)
}

func (s *Store) Has(kind, id string) bool {
	_, err := os.Stat(s.dir(kind, "src", id+".json"))
	return err == nil
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

func (s *Store) Sources(kind string) ([]Sample, error) {
	var out []Sample
	err := filepath.WalkDir(s.dir(kind, "src"), func(p string, d fs.DirEntry, err error) error {
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

func (s *Store) Log(kind string, e ManifestEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.dir(kind, "manifest.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(e)
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
