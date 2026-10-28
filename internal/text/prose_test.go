package text

import (
	"strings"
	"testing"
)

func TestRepairGlued(t *testing.T) {
	cases := []struct{ in, want string }{
		// Repaired: a lost space or line break between sentences.
		{"Nobody knew why.In 1938 a painter found it.", "Nobody knew why. In 1938 a painter found it."},
		{"It worked!Then it broke.", "It worked! Then it broke."},
		{"Really?Yes.", "Really? Yes."},
		{"Why Does It ExistA clay pot was found.", "Why Does It Exist A clay pot was found."},
		// Guarded: not sentence boundaries.
		{"Edit cloudbuild.yaml and rerun.", "Edit cloudbuild.yaml and rerun."},
		{"Upgrade to v1.2.3 first.", "Upgrade to v1.2.3 first."},
		{"Use a helper, e.g. a script.", "Use a helper, e.g. a script."},
		{"The U.S.Army bought it.", "The U.S.Army bought it."},
		{"Call Microsoft.BizTalk.Pipeline from the host.", "Call Microsoft.BizTalk.Pipeline from the host."},
		{"See https://example.com/a.Thing for details.", "See https://example.com/a.Thing for details."},
		{"Mail joe@example.com.Today works.", "Mail joe@example.com.Today works."},
		{"OpenAI is a company.", "OpenAI is a company."},
		{"Run it with Node.js today.", "Run it with Node.js today."},
	}
	for _, c := range cases {
		if got := repairGlued(c.in); got != c.want {
			t.Errorf("repairGlued(%q)\n got  %q\n want %q", c.in, got, c.want)
		}
	}
}

func TestPrepareProseExclusions(t *testing.T) {
	prose := "I spent the weekend moving our build to a new runner, and it went better than expected."
	cases := []struct {
		name  string
		extra string
		check func(Exclusions) bool
	}{
		{"fenced code", "```yaml\nsteps:\n  - name: build\n```", func(x Exclusions) bool { return x.Code > 0 }},
		{"indented code", "    go build ./...\n    go test ./...", func(x Exclusions) bool { return x.Code > 0 }},
		{"pasted code", "docker build -t app .\ndocker run -p 8080:8080 app", func(x Exclusions) bool { return x.Code > 0 }},
		{"markdown table", "| Name | Score |\n|---|---|\n| a | 1 |", func(x Exclusions) bool { return x.Tables > 0 }},
		{"html table", "<table><tr><td>Name</td><td>Score</td></tr></table>", func(x Exclusions) bool { return x.Tables > 0 }},
		{"flattened table", "Feature Free Plan Pro Plan Enterprise Storage 5 GB 100 GB Unlimited Support Email Priority Dedicated", func(x Exclusions) bool { return x.Tables > 0 }},
		{"front matter", "---\nname: release-notes\ndescription: Draft release notes\n---", func(x Exclusions) bool { return x.FrontMatter > 0 }},
		{"label", "Figure 2", func(x Exclusions) bool { return x.Labels > 0 }},
		{"heading", "## Getting started", func(x Exclusions) bool { return x.Labels > 0 }},
	}
	for _, c := range cases {
		got, x := PrepareProse(prose + "\n\n" + c.extra + "\n\n" + prose)
		if !c.check(x) {
			t.Errorf("%s: not excluded (%+v)\n%s", c.name, x, got)
		}
		if strings.Count(got, prose) != 2 {
			t.Errorf("%s: prose was damaged:\n%s", c.name, got)
		}
	}
}

func TestPrepareProseKeepsProse(t *testing.T) {
	in := "Here's the thing about migrations: they always take longer than planned.\n\nWe moved 40 services in March. Two of them broke, and one (the billing worker) took a week to fix.\n\n- Plan for the slow ones first.\n- Keep the old path running until traffic drains."
	got, x := PrepareProse(in)
	if x.Total() != 0 || got != in {
		t.Errorf("prose changed (%+v):\n%s", x, got)
	}
}

func TestPrepareProseIdempotent(t *testing.T) {
	in := "Intro sentence goes here, explaining the setup.\n\n```\ncode\n```\n\n| a | b |\n|---|---|\n\nCaption\n\nIt failed.Then it worked again after a restart."
	once, _ := PrepareProse(in)
	twice, x := PrepareProse(once)
	if once != twice || x.Total() != 0 {
		t.Errorf("not idempotent:\n%q\n%q", once, twice)
	}
}

func TestWindowsSplitsLongParagraphs(t *testing.T) {
	// A pasted article with no paragraph breaks still gets segmented.
	one := strings.Repeat("This sentence is ten words long and ends with a period. ", 60)
	if got := Windows(one, 200); len(got) < 2 {
		t.Errorf("got %d windows for a 600-word paragraph, want several", len(got))
	}
}

func TestEmbeddedBlocks(t *testing.T) {
	prose := "I built this over a weekend and wanted to share how it went, including the parts that didn't work."
	md := "## Workflow\n\n1. **Assess**: Evaluate the workload against the key dimensions before picking anything.\n2. **Select**: Map the workload to the right database and check the reference matrix.\n\n> 📖 Consult the [decision matrix](./matrix.md) for complete comparison tables and guidance.\n\n### Step 2\n\n- **Install**: Download the skill into the workspace and verify it loads correctly.\n- **Run**: Execute the guided solution and report the results back to the user."
	in := prose + "\n\n" + prose + "\n\n" + md + "\n\n" + prose + "\n\n" + prose
	blocks := EmbeddedBlocks(in)
	if len(blocks) != 1 || blocks[0].What != "embedded Markdown block" || strings.Contains(blocks[0].Without, "Assess") {
		t.Fatalf("got %+v", blocks)
	}
	// Ordinary prose with one short list isn't an embedded block.
	plain := prose + "\n\n- first point here\n- second point here\n\n" + prose
	if b := EmbeddedBlocks(plain); len(b) != 0 {
		t.Errorf("false positive: %+v", b)
	}
}
