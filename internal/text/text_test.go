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

func TestStripLinks(t *testing.T) {
	in := "Fix crash on empty config (#4521)\n\nFixes #4520. See https://github.com/x/y/issues/1 and thanks @alice for the report.\nCloses golang/go#123"
	want := "Fix crash on empty config\n\nSee and thanks for the report."
	if got := StripLinks(in); got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

func TestStripEnvelope(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Hi Phillip,\n\nThe draw request is attached. Please wire the funds.\n\nBest,\nGreg", "The draw request is attached. Please wire the funds."},
		{"Phillip & Keith\n\nAttached is the first draw request. Again, thanks for your help. Sincerely,", "Attached is the first draw request. Again, thanks for your help."},
		{"Thank you for meeting with us today. We will send the list by Friday.", "Thank you for meeting with us today. We will send the list by Friday."},
	}
	for _, c := range cases {
		if got := StripEnvelope(c.in); got != c.want {
			t.Errorf("got  %q\nwant %q", got, c.want)
		}
	}
}

func TestSanitize(t *testing.T) {
	in := "The re\u200bsult was fіne (Cyrillic і), but Москва stays. ｆｕｌｌ width"
	got, tm := Sanitize(in)
	want := "The result was fine (Cyrillic і), but Москва stays. full width"
	if got != want || tm.ZeroWidth != 1 || tm.Homoglyphs != 1 {
		t.Errorf("got %q %+v", got, tm)
	}
}
