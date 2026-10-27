package corpus

import "testing"

func TestClean(t *testing.T) {
	cases := []struct{ kind, in, want string }{
		{"commit", "Here's a commit message:\n```\nFix thing (#12)\n\nSigned-off-by: A <a@b.c>\n```", "Fix thing"},
		{"blog", "# My Title\n\nIntro with **bold** and *italic*.\n\n## Section\n\n- **Point:** detail\n- other", "Intro with bold and italic.\n\nSection\n\nPoint: detail\nother"},
		{"memo", "Subject: Q3 plan\n\nHi team,\n\n**Action:** reply by Friday.", "Action: reply by Friday."},
	}
	for _, c := range cases {
		if got := clean(c.kind, c.in, ""); got != c.want {
			t.Errorf("%s:\n got %q\nwant %q", c.kind, got, c.want)
		}
	}
}

func TestNormalizeCommit(t *testing.T) {
	in := "Fix the parser (#4521)\n\nLonger explanation.\n\nChange-Id: I123\nReviewed-on: https://go-review\nRun-TryBot: true"
	if got := NormalizeCommit(in); got != "Fix the parser\n\nLonger explanation." {
		t.Errorf("got %q", got)
	}
}

func TestPromptsDoNotHumanize(t *testing.T) {
	if err := checkPrompts(); err != nil {
		t.Fatal(err)
	}
	writePrompts["blog"] = append(writePrompts["blog"], "Write it so it sounds human.")
	defer func() { writePrompts["blog"] = writePrompts["blog"][:len(writePrompts["blog"])-1] }()
	if checkPrompts() == nil {
		t.Fatal("evasion wording was not caught")
	}
}
