package corpus

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// chatGen calls an OpenAI-compatible /chat/completions endpoint: OpenAI
// itself, or a local server such as LM Studio.
type chatGen struct {
	baseURL, key, model string
	budget              *Budget // nil for free local servers
	prov                string  // provider name when not inferred (mlx)
	maxTokens           int     // sent when set (mlx_lm.server defaults to 512)
}

func (g *chatGen) Name() string { return g.model }

func (g *chatGen) Spec() Spec { return Spec{Provider: g.provider(), Model: g.model} }

func (g *chatGen) provider() string {
	if g.prov != "" {
		return g.prov
	}
	if g.budget == nil {
		return "lmstudio"
	}
	return "openai"
}

var thinkRE = regexp.MustCompile(`(?s)<think>.*?</think>`)

var chatClient = &http.Client{Timeout: 20 * time.Minute} // local reasoning models can be slow

func (g *chatGen) Generate(ctx context.Context, prompt string) (string, error) {
	if g.budget != nil {
		if err := g.budget.Check(g.model); err != nil {
			return "", err
		}
	}
	start := time.Now()
	reqBody := map[string]any{
		"model":    g.model,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
	}
	if g.maxTokens > 0 {
		reqBody["max_tokens"] = g.maxTokens
	}
	if g.prov == "mlx" {
		delete(reqBody, "model") // mlx_lm.server then uses the model it was started with
	}
	body, _ := json.Marshal(reqBody)
	var lastErr error
	for attempt := range 4 {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(attempt*attempt) * 5 * time.Second):
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
		req, err := http.NewRequestWithContext(ctx, "POST", g.baseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			return "", err
		}
		req.Header.Set("Content-Type", "application/json")
		if g.key != "" {
			req.Header.Set("Authorization", "Bearer "+g.key)
		}
		resp, err := chatClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		var out struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
					Refusal string `json:"refusal"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Error json.RawMessage `json:"error"` // an object from OpenAI, a plain string from some local servers
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"` // includes reasoning tokens
			} `json:"usage"`
		}
		err = json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
		if err == nil {
			mode, b := ModeOnline, g.budget
			if b == nil { // free local server: log usage for throughput only
				mode, b = ModeLocal, Spend
			}
			b.Record(Entry{Provider: g.provider(), Model: g.model, Mode: mode,
				Usage: Usage{Input: out.Usage.PromptTokens, Output: out.Usage.CompletionTokens, Seconds: time.Since(start).Seconds()}})
		}
		switch {
		case resp.StatusCode == 429 || resp.StatusCode >= 500:
			lastErr = fmt.Errorf("%s: HTTP %d", g.model, resp.StatusCode)
			continue
		case err != nil:
			return "", fmt.Errorf("%s: %w", g.model, err)
		case len(out.Error) > 0 && string(out.Error) != "null":
			return "", fmt.Errorf("%s: %s", g.model, errorText(out.Error))
		case len(out.Choices) == 0:
			return "", fmt.Errorf("%s: no choices", g.model)
		}
		c := out.Choices[0]
		if c.Message.Refusal != "" || c.FinishReason == "content_filter" {
			return "", ErrRefused
		}
		if c.FinishReason == "length" {
			return "", fmt.Errorf("%s: output truncated", g.model)
		}
		// Some servers return reasoning inline; keep only the answer.
		return strings.TrimSpace(thinkRE.ReplaceAllString(c.Message.Content, "")), nil
	}
	return "", lastErr
}

// errorText extracts a message from an error that may be a string or an
// object with a "message" field.
func errorText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var o struct {
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &o) == nil && o.Message != "" {
		return o.Message
	}
	return string(raw)
}
