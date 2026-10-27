// Package mcpserver exposes the detection engine as MCP tools.
package mcpserver

import (
	"context"
	"fmt"
	"os"
	"slices"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/gitlog"
)

const maxFileBytes = 2 << 20

type textArgs struct {
	Text string `json:"text" jsonschema:"the text to analyze"`
	Kind string `json:"kind,omitempty" jsonschema:"auto (default), prose (blog/memo/report), commit, or pr"`
}

type fileArgs struct {
	Path string `json:"path" jsonschema:"absolute path to a .txt or .md file"`
	Kind string `json:"kind,omitempty" jsonschema:"auto (default), prose, commit, or pr"`
}

type commitArgs struct {
	Repo  string `json:"repo,omitempty" jsonschema:"path to a git repository (default: server working directory)"`
	Range string `json:"range,omitempty" jsonschema:"revision range such as main..HEAD or HEAD~20..HEAD (default: current branch)"`
	Max   int    `json:"max,omitempty" jsonschema:"maximum commits to score (default 20, max 200)"`
}

type commitScore struct {
	SHA        string `json:"sha"`
	Subject    string `json:"subject"`
	Score      int    `json:"score"`
	Label      string `json:"label"`
	Confidence string `json:"confidence"`
	TopSignal  string `json:"top_signal,omitempty"`
}

type commitsResult struct {
	Commits   []commitScore `json:"commits"`
	MeanScore float64       `json:"mean_score"`
	Flagged   int           `json:"flagged" jsonschema:"commits scoring 60 or higher"`
}

// New builds the MCP server with all detection tools registered.
func New(e *engine.Engine, version string) *mcp.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "aidetector", Version: version}, nil)

	mcp.AddTool(s, &mcp.Tool{
		Name: "detect_ai_text",
		Description: "Score how likely text is to be AI-written, 0 (human) to 100 (AI). " +
			"Scores share one scale across text kinds: 40+ is mixed (possibly AI-assisted), 60+ likely AI, 80+ AI, " +
			"set so that roughly 10%, 2%, and 0.5% of held-out human writing of the same kind reaches each. " +
			"Returns the score, a label, a confidence level, ranked reasons, and per-segment scores for long documents. " +
			"Works best on 150+ words of English prose; short texts get low confidence.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, a textArgs) (*mcp.CallToolResult, engine.Result, error) {
		res, err := e.Analyze(ctx, detect.Input{Text: a.Text, Kind: detect.ParseKind(a.Kind)})
		return nil, res, err
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "detect_ai_file",
		Description: "Like detect_ai_text, but reads the text from a local .txt or .md file.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, a fileArgs) (*mcp.CallToolResult, engine.Result, error) {
		b, err := readFile(a.Path)
		if err != nil {
			return nil, engine.Result{}, err
		}
		res, err := e.Analyze(ctx, detect.Input{Text: string(b), Kind: detect.ParseKind(a.Kind)})
		return nil, res, err
	})

	mcp.AddTool(s, &mcp.Tool{
		Name: "detect_ai_commits",
		Description: "Score the commit messages in a local git repository for AI authorship. " +
			"Returns a score per commit plus the mean. Commit messages are short, so treat individual scores as weak evidence.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, a commitArgs) (*mcp.CallToolResult, commitsResult, error) {
		res, err := ScoreCommits(ctx, e, a.Repo, a.Range, a.Max)
		return nil, res, err
	})

	return s
}

// ScoreCommits scores each commit message in a range. Shared with the CLI.
func ScoreCommits(ctx context.Context, e *engine.Engine, repo, revRange string, max int) (commitsResult, error) {
	if repo == "" {
		repo = "."
	}
	if max <= 0 {
		max = 20
	}
	max = min(max, 200)
	commits, err := gitlog.Log(ctx, repo, revRange, max)
	if err != nil {
		return commitsResult{}, err
	}
	out := commitsResult{Commits: []commitScore{}}
	total := 0
	for _, c := range commits {
		res, err := e.Analyze(ctx, detect.Input{Text: c.Message, Kind: detect.KindCommit})
		if err != nil {
			return out, err
		}
		cs := commitScore{SHA: c.SHA[:min(12, len(c.SHA))], Subject: c.Subject(), Score: res.Score,
			Label: res.Label, Confidence: res.Confidence, TopSignal: topSignal(res.Signals)}
		out.Commits = append(out.Commits, cs)
		total += res.Score
		if res.Score >= 60 {
			out.Flagged++
		}
	}
	if len(commits) > 0 {
		out.MeanScore = float64(total) / float64(len(commits))
	}
	return out, nil
}

// topSignal names the signal that moved the score furthest from neutral.
func topSignal(signals []detect.Signal) string {
	if len(signals) == 0 {
		return ""
	}
	best := slices.MaxFunc(signals, func(a, b detect.Signal) int {
		da, db := a.Weight*abs(a.AIProb-0.5), b.Weight*abs(b.AIProb-0.5)
		switch {
		case da < db:
			return -1
		case da > db:
			return 1
		}
		return 0
	})
	return best.Detail
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func readFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return nil, fmt.Errorf("%s is a directory", path)
	}
	if info.Size() > maxFileBytes {
		return nil, fmt.Errorf("%s is %d bytes; limit is %d", path, info.Size(), maxFileBytes)
	}
	return os.ReadFile(path)
}
