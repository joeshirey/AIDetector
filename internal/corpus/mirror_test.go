package corpus

import "testing"

func TestClean(t *testing.T) {
	cases := []struct{ kind, in, want string }{
		{"pr", "Here's the PR description:\n```\nFixes the parser when the config file is empty.\n\nhttps://example.com/issue\n```", "Fixes the parser when the config file is empty."},
		{"blog", "# My Title\n\nIntro with **bold** and *italic*.\n\n## Section\n\n- **Point:** detail\n- other", "Intro with bold and italic.\n\nSection\n\nPoint: detail\nother"},
		{"memo", "Subject: Q3 plan\n\nHi team,\n\n**Action:** reply by Friday.", "Action: reply by Friday."},
	}
	for _, c := range cases {
		if got := clean(c.kind, c.in, ""); got != c.want {
			t.Errorf("%s:\n got %q\nwant %q", c.kind, got, c.want)
		}
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
