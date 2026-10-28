// Package mcpserver exposes the detection engine as MCP tools.
package mcpserver

import (
	"context"
	"fmt"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/engine"
)

const maxFileBytes = 2 << 20

type textArgs struct {
	Text string `json:"text" jsonschema:"the text to analyze"`
	Kind string `json:"kind,omitempty" jsonschema:"auto (default), prose (blog/memo/report/email/essay), or pr"`
}

type fileArgs struct {
	Path string `json:"path" jsonschema:"absolute path to a .txt or .md file"`
	Kind string `json:"kind,omitempty" jsonschema:"auto (default), prose, or pr"`
}

// instructions reach every client when it connects, so agents interpret
// results responsibly even without the companion skill installed.
const instructions = `aidetector scores English text from 0 (human) to 100 (AI-written) and explains why.

- Pass kind when you know it: prose (blogs, memos, reports, emails, essays) or pr.
- Labels: 80+ AI, 60-79 likely AI, 40-59 inconclusive, below 40 likely human. At 60, about 2% of human writing of the same kind scores as high, so a result is evidence, not proof.
- "too short to judge" means the text is under the minimum length (50 words for prose, 25 for PRs); say so rather than reporting the score.
- Explain results with the top two or three reasons. Always mention a "warning" reason (hidden or look-alike characters removed, or most of the AI evidence coming from one embedded or quoted block). "info" reasons say how much code, tables, or headings were excluded from scoring.
- Describe the text, not the person ("reads as likely AI-written", not "they used AI"), and don't present a score as the sole basis for consequential decisions.`

// New builds the MCP server with all detection tools registered.
func New(e *engine.Engine, version string) *mcp.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "aidetector", Version: version}, &mcp.ServerOptions{Instructions: instructions})

	mcp.AddTool(s, &mcp.Tool{
		Name: "detect_ai_text",
		Description: "Score how likely text is to be AI-written, 0 (human) to 100 (AI). " +
			"Scores share one scale across text kinds: 40+ is inconclusive, 60+ likely AI, 80+ AI, " +
			"set so that roughly 5%, 2%, and 0.5% of held-out human writing of the same kind reaches each. " +
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

	return s
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
