package corpus

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/anthropics/anthropic-sdk-go/v2/option"
	"github.com/anthropics/anthropic-sdk-go/v2/vertex"
	"google.golang.org/genai"
)

// Generator produces AI text for a prompt.
type Generator interface {
	Name() string // stable id used in paths, e.g. "claude-opus-5-5" or "claude-haiku-5-5@low"
	Spec() Spec
	Generate(ctx context.Context, prompt string) (string, error)
}

// Spec identifies how a generator is called: provider, model, and effort
// ("" for the model's default, or "low").
type Spec struct {
	Provider, Model, Effort string
}

// Name is the generator's path name: the model, plus "@effort" when it isn't
// the default.
func (s Spec) Name() string {
	if s.Effort != "" {
		return s.Model + "@" + s.Effort
	}
	return s.Model
}

// ErrRefused means the model declined; the sample is skipped rather than
// filled in by another model, so labels stay accurate.
var ErrRefused = errors.New("model refused")

// ParseSpec parses "provider:model" or "provider:model@low".
func ParseSpec(spec string) (Spec, error) {
	provider, model, ok := strings.Cut(strings.TrimSpace(spec), ":")
	if !ok {
		return Spec{}, fmt.Errorf("generator %q: want provider:model", spec)
	}
	model, effort, _ := strings.Cut(model, "@")
	if effort != "" && effort != "low" {
		return Spec{}, fmt.Errorf("generator %q: effort must be omitted (default) or @low", spec)
	}
	return Spec{Provider: provider, Model: model, Effort: effort}, nil
}

// NewGenerators parses specs like "claude:claude-opus-5-5,gemini:gemini-3.8-flash@low".
// Providers:
//   - claude: Vertex AI when ANTHROPIC_VERTEX_PROJECT_ID is set, else the Claude API
//   - gemini: Vertex AI (GOOGLE_CLOUD_PROJECT or ANTHROPIC_VERTEX_PROJECT_ID;
//     GOOGLE_CLOUD_LOCATION, default global), with application-default credentials
//   - gemini-api: the Gemini API with GEMINI_API_KEY (explicit fallback only)
//   - openai: OPENAI_API_KEY
//   - lmstudio: a local LM Studio server at LMSTUDIO_URL (default http://127.0.0.1:1234/v1)
//   - mlx: a local mlx_lm.server at MLX_URL (default http://127.0.0.1:8081/v1); the model
//     is whatever the server loaded, named by the spec for the corpus directory
//
// Paid providers check Spend before each call and record usage to its ledger.
func NewGenerators(ctx context.Context, specs string) ([]Generator, error) {
	var out []Generator
	var claude *anthropic.Client
	var gemVertex, gemAPI *genai.Client
	for _, raw := range strings.Split(specs, ",") {
		s, err := ParseSpec(raw)
		if err != nil {
			return nil, err
		}
		switch s.Provider {
		case "claude":
			if claude == nil {
				if claude, err = newClaudeClient(ctx); err != nil {
					return nil, err
				}
			}
			out = append(out, &claudeGen{client: claude, spec: s})
		case "gemini":
			if gemVertex == nil {
				if gemVertex, err = newGeminiVertexClient(ctx); err != nil {
					return nil, err
				}
			}
			out = append(out, &geminiGen{client: gemVertex, spec: s, provider: "vertex"})
		case "gemini-api":
			if gemAPI == nil {
				key := os.Getenv("GEMINI_API_KEY")
				if key == "" {
					return nil, errors.New("GEMINI_API_KEY is not set")
				}
				if gemAPI, err = genai.NewClient(ctx, &genai.ClientConfig{APIKey: key, Backend: genai.BackendGeminiAPI}); err != nil {
					return nil, err
				}
			}
			out = append(out, &geminiGen{client: gemAPI, spec: s, provider: "gemini-api"})
		case "openai":
			key := os.Getenv("OPENAI_API_KEY")
			if key == "" {
				return nil, errors.New("OPENAI_API_KEY is not set")
			}
			out = append(out, &chatGen{baseURL: "https://api.openai.com/v1", key: key, model: s.Model, budget: Spend})
		case "lmstudio":
			base := os.Getenv("LMSTUDIO_URL")
			if base == "" {
				base = "http://127.0.0.1:1234/v1"
			}
			out = append(out, &chatGen{baseURL: strings.TrimRight(base, "/"), model: s.Model})
		case "mlx":
			base := os.Getenv("MLX_URL")
			if base == "" {
				base = "http://127.0.0.1:8081/v1"
			}
			out = append(out, &chatGen{baseURL: strings.TrimRight(base, "/"), model: s.Model, prov: "mlx", maxTokens: 16000})
		default:
			return nil, fmt.Errorf("generator %q: unknown provider %q", raw, s.Provider)
		}
	}
	return out, nil
}

// VertexProject is the Google Cloud project for Vertex AI calls.
func VertexProject() string {
	if p := os.Getenv("GOOGLE_CLOUD_PROJECT"); p != "" {
		return p
	}
	return os.Getenv("ANTHROPIC_VERTEX_PROJECT_ID")
}

func newGeminiVertexClient(ctx context.Context) (*genai.Client, error) {
	project := VertexProject()
	if project == "" {
		return nil, errors.New("set GOOGLE_CLOUD_PROJECT (or ANTHROPIC_VERTEX_PROJECT_ID) for Gemini on Vertex AI; gemini-api:<model> uses GEMINI_API_KEY instead")
	}
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "global" // regional endpoints cost about 10% more
	}
	return genai.NewClient(ctx, &genai.ClientConfig{Backend: genai.BackendVertexAI, Project: project, Location: location})
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
	spec   Spec
}

func (g *claudeGen) Name() string { return g.spec.Name() }
func (g *claudeGen) Spec() Spec   { return g.spec }

func (g *claudeGen) Generate(ctx context.Context, prompt string) (string, error) {
	if err := Spend.Check(g.spec.Model); err != nil {
		return "", err
	}
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(g.spec.Model),
		MaxTokens: 16000,
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(prompt))},
	}
	if g.spec.Effort == "low" {
		params.OutputConfig = anthropic.OutputConfigParam{Effort: anthropic.OutputConfigEffortLow}
	}
	start := time.Now()
	resp, err := g.client.Messages.New(ctx, params)
	if err != nil {
		return "", err
	}
	Spend.Record(Entry{Provider: "vertex-claude", Model: g.spec.Model, Mode: ModeOnline, Usage: Usage{
		Input:  int(resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.CacheCreationInputTokens),
		Output: int(resp.Usage.OutputTokens), CacheRead: int(resp.Usage.CacheReadInputTokens), Seconds: time.Since(start).Seconds()}})
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
		return "", fmt.Errorf("%s: output truncated at max_tokens", g.spec.Model)
	}
	return sb.String(), nil
}

type geminiGen struct {
	client   *genai.Client
	spec     Spec
	provider string
}

func (g *geminiGen) Name() string { return g.spec.Name() }
func (g *geminiGen) Spec() Spec   { return g.spec }

// GeminiLowThinking is the thinking level used for "@low".
var GeminiLowThinking = genai.ThinkingLevelLow // MINIMAL is rejected on Vertex for 3.8 Flash and Argon

func (g *geminiGen) Generate(ctx context.Context, prompt string) (string, error) {
	if err := Spend.Check(g.spec.Model); err != nil {
		return "", err
	}
	var cfg *genai.GenerateContentConfig
	if g.spec.Effort == "low" {
		cfg = &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: GeminiLowThinking}}
	}
	start := time.Now()
	resp, err := g.client.Models.GenerateContent(ctx, g.spec.Model, genai.Text(prompt), cfg)
	if err != nil {
		return "", err
	}
	if u := resp.UsageMetadata; u != nil {
		Spend.Record(Entry{Provider: g.provider, Model: g.spec.Model, Mode: ModeOnline, Usage: Usage{
			Input: int(u.PromptTokenCount), Output: int(u.CandidatesTokenCount), Thinking: int(u.ThoughtsTokenCount),
			CacheRead: int(u.CachedContentTokenCount), Seconds: time.Since(start).Seconds()}})
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].FinishReason == genai.FinishReasonSafety {
		return "", ErrRefused
	}
	return resp.Text(), nil
}
