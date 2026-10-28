// Package perplexity gets zero-shot token-probability statistics (log
// perplexity, Binoculars, Fast-DetectGPT curvature) from the local scorer
// service in scorer/. It is optional: without a reachable scorer it produces
// no signals and the engine uses its text-only models.
package perplexity

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/text"
)

// EnvURL names the environment variable holding the scorer's base URL.
const EnvURL = "AIDETECTOR_SCORER_URL"

type Detector struct {
	url    string
	model  string
	client *http.Client
}

// FromEnv returns a detector for the scorer named in AIDETECTOR_SCORER_URL,
// or nil if it isn't set or doesn't answer a health check.
func FromEnv() *Detector {
	url := strings.TrimRight(os.Getenv(EnvURL), "/")
	if url == "" {
		return nil
	}
	d := &Detector{url: url, client: &http.Client{Timeout: 5 * time.Minute}}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", url+"/health", nil)
	resp, err := d.client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var h struct {
		Model string `json:"model"`
	}
	if resp.StatusCode != 200 || json.NewDecoder(resp.Body).Decode(&h) != nil {
		return nil
	}
	d.model = h.Model
	return d
}

func (*Detector) Name() string { return "perplexity" }

// Optional tells the engine to carry on without these signals if the scorer
// fails mid-run.
func (*Detector) Optional() bool { return true }

// Model is the scorer's model (or observer+performer pair) name.
func (d *Detector) Model() string { return d.model }

type result struct {
	Tokens     int      `json:"tokens"`
	TooShort   bool     `json:"too_short"`
	LogPPL     *float64 `json:"log_ppl"`
	XPPL       *float64 `json:"x_ppl"`
	Binoculars *float64 `json:"binoculars"`
	Curvature  *float64 `json:"curvature"`
}

func (d *Detector) Detect(ctx context.Context, in detect.Input) ([]detect.Signal, error) {
	body, _ := json.Marshal(map[string][]string{"texts": {text.Normalize(string(in.Kind), in.Text)}})
	req, err := http.NewRequestWithContext(ctx, "POST", d.url+"/score", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("scorer: %w", err)
	}
	defer resp.Body.Close()
	var out struct {
		Results []result `json:"results"`
		Error   string   `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("scorer: %w", err)
	}
	if out.Error != "" || len(out.Results) != 1 {
		return nil, fmt.Errorf("scorer: %s", out.Error)
	}
	r := out.Results[0]
	if r.TooShort {
		return nil, nil
	}
	// Weight 0: the hand-set engine ignores these; fitted models learn them.
	var sigs []detect.Signal
	add := func(name string, v *float64, detail string) {
		if v != nil {
			sigs = append(sigs, detect.Signal{Name: name, Detector: "perplexity", Value: *v, AIProb: 0.5,
				Detail: fmt.Sprintf(detail, *v, r.Tokens, d.model)})
		}
	}
	add("lm_log_ppl", r.LogPPL, "log-perplexity %.2f over %d tokens under %s (lower is more predictable)")
	add("lm_binoculars", r.Binoculars, "Binoculars score %.3f over %d tokens (%s)")
	add("lm_curvature", r.Curvature, "Fast-DetectGPT curvature %.2f over %d tokens (%s)")
	return sigs, nil
}
