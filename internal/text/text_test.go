package text

import (
	"slices"
	"testing"
)

func TestSentences(t *testing.T) {
	got := Sentences("Hello there. This is e.g. a test! Is it?\n\n# Heading\n\n- item one\n- item two")
	want := []string{"Hello there.", "This is e.g. a test!", "Is it?", "item one", "item two"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestWindowsFoldsShortTail(t *testing.T) {
	long := "one two three four five six seven eight nine ten"
	got := Windows(long+"\n\n"+long+"\n\nshort", 10)
	if len(got) != 2 {
		t.Errorf("got %d windows, want 2: %q", len(got), got)
	}
}
