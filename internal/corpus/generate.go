package corpus

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/anthropics/anthropic-sdk-go/v2/option"
	"github.com/anthropics/anthropic-sdk-go/v2/vertex"
	"google.golang.org/genai"
)

// Generator produces AI text for a prompt.
type Generator interface {
	Name() string // stable id used in paths, e.g. "claude-opus-5-5"
	Generate(ctx context.Context, prompt string) (string, error)
}

// ErrRefused means the model declined; the sample is skipped rather than
// filled in by another model, so labels stay accurate.
var ErrRefused = errors.New("model refused")

// NewGenerators parses specs like "claude:claude-opus-5-5,gemini:gemini-3.8-flash".
func NewGenerators(ctx context.Context, specs string) ([]Generator, error) {
	var out []Generator
	var claude *anthropic.Client
	var gem *genai.Client
	for _, spec := range strings.Split(specs, ",") {
		provider, model, ok := strings.Cut(strings.TrimSpace(spec), ":")
		if !ok {
			return nil, fmt.Errorf("generator %q: want provider:model", spec)
		}
		switch provider {
		case "claude":
			if claude == nil {
				c, err := newClaudeClient(ctx)
				if err != nil {
					return nil, err
				}
				claude = c
			}
			out = append(out, &claudeGen{client: claude, model: model})
		case "gemini":
			if gem == nil {
				key := os.Getenv("GEMINI_API_KEY")
				if key == "" {
					return nil, errors.New("GEMINI_API_KEY is not set")
				}
				c, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey: key, Backend: genai.BackendGeminiAPI})
				if err != nil {
					return nil, err
				}
				gem = c
			}
			out = append(out, &geminiGen{client: gem, model: model})
		default:
			return nil, fmt.Errorf("generator %q: unknown provider %q", spec, provider)
		}
	}
	return out, nil
}

// newClaudeClient uses Vertex AI when ANTHROPIC_VERTEX_PROJECT_ID is set
// (auth via gcloud application-default credentials), else the Claude API.
func newClaudeClient(ctx context.Context) (*anthropic.Client, error) {
	opts := []option.RequestOption{option.WithMaxRetries(5)}
	if project := os.Getenv("ANTHROPIC_VERTEX_PROJECT_ID"); project != "" {
		region := os.Getenv("CLOUD_ML_REGION")
		if region == "" {
			region = "global"
		}
		opts = append(opts, vertex.WithGoogleAuth(ctx, region, project))
	}
	c := anthropic.NewClient(opts...)
	return &c, nil
}

type claudeGen struct {
	client *anthropic.Client
	model  string
}

func (g *claudeGen) Name() string { return g.model }

func (g *claudeGen) Generate(ctx context.Context, prompt string) (string, error) {
	resp, err := g.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(g.model),
		MaxTokens: 16000,
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(prompt))},
	})
	if err != nil {
		return "", err
	}
	if resp.StopReason == anthropic.StopReasonRefusal {
		return "", ErrRefused
	}
	var sb strings.Builder
	for _, block := range resp.Content {
		if t, ok := block.AsAny().(anthropic.TextBlock); ok {
			sb.WriteString(t.Text)
		}
	}
	if resp.StopReason == anthropic.StopReasonMaxTokens {
		return "", fmt.Errorf("%s: output truncated at max_tokens", g.model)
	}
	return sb.String(), nil
}

type geminiGen struct {
	client *genai.Client
	model  string
}

func (g *geminiGen) Name() string { return g.model }

func (g *geminiGen) Generate(ctx context.Context, prompt string) (string, error) {
	resp, err := g.client.Models.GenerateContent(ctx, g.model, genai.Text(prompt), nil)
	if err != nil {
		return "", err
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].FinishReason == genai.FinishReasonSafety {
		return "", ErrRefused
	}
	return resp.Text(), nil
}
