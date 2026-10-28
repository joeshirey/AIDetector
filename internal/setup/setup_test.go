package setup

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Fake client CLIs record their arguments, so Install runs for real against
// a temporary home without touching any actual client configuration.
func TestInstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake CLIs are shell scripts")
	}
	home, bin := t.TempDir(), t.TempDir()
	log := filepath.Join(bin, "calls.log")
	for _, cli := range []string{"claude", "codex", "agy"} {
		script := "#!/bin/sh\necho \"$(basename $0) $*\" >> " + log + "\n"
		if err := os.WriteFile(filepath.Join(bin, cli), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("HOME", home)
	t.Setenv("CODEX_HOME", "")
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))

	var out bytes.Buffer
	o := Options{MCPPath: "/opt/aidetector-mcp", Env: []string{"AIDETECTOR_SCORER_URL=http://x"}, Skill: true, Server: true, Out: &out}
	for _, c := range Clients {
		if err := Install(c, o); err != nil {
			t.Fatal(err)
		}
	}

	calls, _ := os.ReadFile(log)
	for _, want := range []string{
		"claude mcp add --scope user -e AIDETECTOR_SCORER_URL=http://x aidetector -- /opt/aidetector-mcp",
		"codex mcp add --env AIDETECTOR_SCORER_URL=http://x aidetector -- /opt/aidetector-mcp",
		"agy mcp add --env AIDETECTOR_SCORER_URL=http://x aidetector /opt/aidetector-mcp",
	} {
		if !strings.Contains(string(calls), want) {
			t.Errorf("missing call %q in:\n%s", want, calls)
		}
	}
	for _, dir := range []string{".claude/skills", ".codex/skills", ".gemini/config/skills"} {
		got, err := os.ReadFile(filepath.Join(home, dir, "ai-text-detection", "SKILL.md"))
		if err != nil || !bytes.Equal(got, Skill) {
			t.Errorf("skill not installed under %s: %v", dir, err)
		}
	}
}

func TestSkillFrontmatter(t *testing.T) {
	s := string(Skill)
	if !strings.HasPrefix(s, "---\nname: ai-text-detection\ndescription: ") || !strings.Contains(s[4:], "\n---\n") {
		t.Fatal("SKILL.md must start with name and description frontmatter")
	}
}
