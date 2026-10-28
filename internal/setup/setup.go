// Package setup registers the MCP server with agent clients (Claude Code,
// Codex, Antigravity) and installs the companion skill into each one.
package setup

import (
	_ "embed"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Skill is the ai-text-detection SKILL.md, in the shared Agent Skills format
// all three clients read.
//
//go:embed skill/SKILL.md
var Skill []byte

const (
	serverName = "aidetector"
	skillName  = "ai-text-detection"
)

// Client describes how one agent client registers MCP servers and where it
// looks for user-level skills.
type Client struct {
	Name     string
	CLI      string
	skillDir func(home string) string
	commands func(mcpPath string, env []string) [][]string // run in order; all but the last may fail
}

var Clients = []Client{
	{
		Name: "claude", CLI: "claude",
		skillDir: func(home string) string { return filepath.Join(home, ".claude", "skills") },
		commands: func(mcp string, env []string) [][]string {
			add := []string{"claude", "mcp", "add", "--scope", "user"}
			for _, e := range env {
				add = append(add, "-e", e)
			}
			return [][]string{
				{"claude", "mcp", "remove", "--scope", "user", serverName}, // re-runs replace the old entry
				append(add, serverName, "--", mcp),
			}
		},
	},
	{
		Name: "codex", CLI: "codex",
		skillDir: func(home string) string {
			if h := os.Getenv("CODEX_HOME"); h != "" {
				return filepath.Join(h, "skills")
			}
			return filepath.Join(home, ".codex", "skills")
		},
		commands: func(mcp string, env []string) [][]string {
			add := []string{"codex", "mcp", "add"}
			for _, e := range env {
				add = append(add, "--env", e)
			}
			return [][]string{
				{"codex", "mcp", "remove", serverName},
				append(add, serverName, "--", mcp),
			}
		},
	},
	{
		Name: "antigravity", CLI: "agy",
		skillDir: func(home string) string { return filepath.Join(home, ".gemini", "config", "skills") },
		commands: func(mcp string, env []string) [][]string {
			add := []string{"agy", "mcp", "add"}
			for _, e := range env {
				add = append(add, "--env", e)
			}
			return [][]string{append(add, serverName, mcp)} // add also updates
		},
	},
}

// Find returns the named client.
func Find(name string) (Client, bool) {
	for _, c := range Clients {
		if c.Name == name {
			return c, true
		}
	}
	return Client{}, false
}

type Options struct {
	MCPPath string   // absolute path to aidetector-mcp
	Env     []string // KEY=VALUE pairs for the server, e.g. the scorer URL
	Skill   bool     // install the skill
	Server  bool     // register the MCP server
	DryRun  bool
	Out     io.Writer
}

// Install registers the server and installs the skill for one client. A
// client whose CLI isn't installed is reported, with the manual command,
// rather than treated as an error.
func Install(c Client, o Options) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	if o.Server {
		cmds := c.commands(o.MCPPath, o.Env)
		if _, err := exec.LookPath(c.CLI); err != nil {
			fmt.Fprintf(o.Out, "%s: %s not found on PATH; register manually with:\n  %s\n", c.Name, c.CLI, shellJoin(cmds[len(cmds)-1]))
		} else {
			for i, args := range cmds {
				last := i == len(cmds)-1
				if o.DryRun {
					fmt.Fprintf(o.Out, "%s: would run %s\n", c.Name, shellJoin(args))
					continue
				}
				out, err := exec.Command(args[0], args[1:]...).CombinedOutput()
				if err != nil && last {
					return fmt.Errorf("%s: %s: %v\n%s", c.Name, shellJoin(args), err, strings.TrimSpace(string(out)))
				}
			}
			if !o.DryRun {
				fmt.Fprintf(o.Out, "%s: registered MCP server %q -> %s\n", c.Name, serverName, o.MCPPath)
			}
		}
	}
	if o.Skill {
		dir := filepath.Join(c.skillDir(home), skillName)
		path := filepath.Join(dir, "SKILL.md")
		if o.DryRun {
			fmt.Fprintf(o.Out, "%s: would write %s\n", c.Name, path)
			return nil
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(path, Skill, 0o644); err != nil {
			return err
		}
		fmt.Fprintf(o.Out, "%s: installed skill %s\n", c.Name, path)
	}
	return nil
}

// DefaultMCPPath finds aidetector-mcp next to the running binary, where
// `go install` puts both.
func DefaultMCPPath() (string, error) {
	self, err := os.Executable()
	if err != nil {
		return "", err
	}
	self, _ = filepath.EvalSymlinks(self)
	p := filepath.Join(filepath.Dir(self), "aidetector-mcp")
	if _, err := os.Stat(p); err != nil {
		return "", errors.New("aidetector-mcp not found next to this binary; install it or pass -mcp /path/to/aidetector-mcp")
	}
	return p, nil
}

func shellJoin(args []string) string {
	q := make([]string, len(args))
	for i, a := range args {
		if a == "" || strings.ContainsAny(a, " \t'\"$\\") {
			a = "'" + strings.ReplaceAll(a, "'", `'\''`) + "'"
		}
		q[i] = a
	}
	return strings.Join(q, " ")
}
