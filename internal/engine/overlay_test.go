package engine

import (
	"io"
	"testing"
	"testing/fstest"
)

// A candidate directory with its own <kind>.lm.json but no
// <kind>.lm.ngram.json must pair its LM model with its own <kind>.ngram.json,
// never with an embedded <kind>.lm.ngram.json fitted for a different model.
func TestOverlayLegacyLMDirUsesOwnNgram(t *testing.T) {
	base := fstest.MapFS{
		"prose.lm.ngram.json": {Data: []byte("embedded-lm-ngram")},
		"prose.ngram.json":    {Data: []byte("embedded-ngram")},
		"pr.lm.ngram.json":    {Data: []byte("embedded-pr-lm-ngram")},
	}
	read := func(o overlayFS, name string) string {
		t.Helper()
		f, err := o.Open(name)
		if err != nil {
			t.Fatalf("open %s: %v", name, err)
		}
		defer f.Close()
		b, _ := io.ReadAll(f)
		return string(b)
	}

	legacy := overlayFS{fstest.MapFS{
		"prose.lm.json":    {Data: []byte("{}")},
		"prose.ngram.json": {Data: []byte("dir-ngram")},
	}, base}
	if got := read(legacy, "prose.lm.ngram.json"); got != "dir-ngram" {
		t.Errorf("legacy dir: prose.lm.ngram.json = %q, want the dir's own prose.ngram.json", got)
	}
	// Another kind without its own LM model still falls through to embedded.
	if got := read(legacy, "pr.lm.ngram.json"); got != "embedded-pr-lm-ngram" {
		t.Errorf("legacy dir: pr.lm.ngram.json = %q, want embedded", got)
	}

	current := overlayFS{fstest.MapFS{
		"prose.lm.json":       {Data: []byte("{}")},
		"prose.lm.ngram.json": {Data: []byte("dir-lm-ngram")},
		"prose.ngram.json":    {Data: []byte("dir-ngram")},
	}, base}
	if got := read(current, "prose.lm.ngram.json"); got != "dir-lm-ngram" {
		t.Errorf("current dir: prose.lm.ngram.json = %q, want the dir's own", got)
	}

	noLM := overlayFS{fstest.MapFS{"prose.ngram.json": {Data: []byte("dir-ngram")}}, base}
	if got := read(noLM, "prose.lm.ngram.json"); got != "embedded-lm-ngram" {
		t.Errorf("dir without an LM model: prose.lm.ngram.json = %q, want embedded", got)
	}
}
