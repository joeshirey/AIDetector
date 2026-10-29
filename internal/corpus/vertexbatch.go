package corpus

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// Vertex AI batch prediction for Claude and Gemini. Anthropic's Message
// Batches API isn't available on Vertex, so both go through Vertex batch
// prediction jobs: requests are uploaded to Cloud Storage as JSONL, a job is
// created, and results are read back from Cloud Storage and matched by
// request ID. Batch runs in a region (not the global endpoint) at about half
// the online price; see PriceTable.BatchMultiplier.

// BatchRequest is one prompt in a batch.
type BatchRequest struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
}

// BatchJob is a submitted job. It's persisted between nights.
type BatchJob struct {
	Name         string    `json:"name"` // projects/.../batchPredictionJobs/...
	Spec         Spec      `json:"spec"`
	Location     string    `json:"location"`
	InputURI     string    `json:"input_uri"`
	OutputPrefix string    `json:"output_prefix"`
	Night        string    `json:"night"` // the night it was submitted, which its cost counts against
	Submitted    time.Time `json:"submitted"`
	Requests     int       `json:"requests"`
	Estimate     float64   `json:"estimate"` // cost reserved at submission
	Retried      bool      `json:"retried,omitempty"`
	State        string    `json:"state,omitempty"`
}

// BatchResult is one request's outcome.
type BatchResult struct {
	ID      string
	Text    string
	Usage   Usage
	Refused bool
	Err     string
}

// Batcher talks to Vertex AI and Cloud Storage with application-default
// credentials.
type Batcher struct {
	Project string
	Bucket  string
	// Locations by provider. Claude batch runs where Claude is hosted
	// (us-east5); newer Gemini models only batch on the global location.
	// Override with
	// AIDETECTOR_BATCH_LOCATION_CLAUDE / _GEMINI.
	ClaudeLocation, GeminiLocation string
	http                           *http.Client
}

func NewBatcher(ctx context.Context, bucket string) (*Batcher, error) {
	ts, err := google.DefaultTokenSource(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("application-default credentials: %w", err)
	}
	b := &Batcher{Project: VertexProject(), Bucket: bucket, ClaudeLocation: "us-east5", GeminiLocation: "global",
		http: oauth2.NewClient(ctx, ts)}
	if v := os.Getenv("AIDETECTOR_BATCH_LOCATION_CLAUDE"); v != "" {
		b.ClaudeLocation = v
	}
	if v := os.Getenv("AIDETECTOR_BATCH_LOCATION_GEMINI"); v != "" {
		b.GeminiLocation = v
	}
	if b.Project == "" || b.Bucket == "" {
		return nil, fmt.Errorf("batch needs a project (GOOGLE_CLOUD_PROJECT or ANTHROPIC_VERTEX_PROJECT_ID) and a bucket")
	}
	return b, nil
}

func (b *Batcher) do(ctx context.Context, method, u string, body io.Reader, contentType string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-goog-user-project", b.Project)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := b.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%s %s: HTTP %d: %s", method, u, resp.StatusCode, strings.TrimSpace(string(data[:min(len(data), 400)])))
	}
	return data, nil
}

func host(location string) string {
	if location == "global" {
		return "aiplatform.googleapis.com"
	}
	return location + "-aiplatform.googleapis.com"
}

// requestLine builds one JSONL input line for the provider's batch format.
func requestLine(s Spec, r BatchRequest) ([]byte, error) {
	switch s.Provider {
	case "claude":
		req := map[string]any{
			"anthropic_version": "vertex-2023-10-16",
			"max_tokens":        16000,
			"messages":          []map[string]any{{"role": "user", "content": r.Prompt}},
		}
		if s.Effort == "low" {
			req["output_config"] = map[string]any{"effort": "low"}
		}
		return json.Marshal(map[string]any{"custom_id": r.ID, "request": req})
	case "gemini":
		req := map[string]any{
			"contents": []map[string]any{{"role": "user", "parts": []map[string]any{{"text": r.Prompt}}}},
			"labels":   map[string]string{"custom_id": r.ID},
		}
		if s.Effort == "low" {
			req["generationConfig"] = map[string]any{"thinkingConfig": map[string]any{"thinkingLevel": string(GeminiLowThinking)}}
		}
		return json.Marshal(map[string]any{"request": req})
	}
	return nil, fmt.Errorf("batch isn't supported for provider %q", s.Provider)
}

func publisher(s Spec) string {
	if s.Provider == "claude" {
		return "anthropic"
	}
	return "google"
}

// Submit uploads the requests and creates a batch prediction job.
func (b *Batcher) Submit(ctx context.Context, s Spec, night string, reqs []BatchRequest, estimate float64) (*BatchJob, error) {
	var buf bytes.Buffer
	for _, r := range reqs {
		line, err := requestLine(s, r)
		if err != nil {
			return nil, err
		}
		buf.Write(line)
		buf.WriteByte('\n')
	}
	stamp := time.Now().UTC().Format("20060102-150405")
	object := fmt.Sprintf("batch/%s/%s-%s/input.jsonl", night, s.Name(), stamp)
	up := fmt.Sprintf("https://storage.googleapis.com/upload/storage/v1/b/%s/o?uploadType=media&name=%s", b.Bucket, url.QueryEscape(object))
	if _, err := b.do(ctx, "POST", up, &buf, "application/jsonl"); err != nil {
		return nil, fmt.Errorf("upload batch input: %w", err)
	}
	loc := b.GeminiLocation
	if s.Provider == "claude" {
		loc = b.ClaudeLocation
	}
	job := &BatchJob{Spec: s, Location: loc, Night: night, Submitted: time.Now().UTC(), Requests: len(reqs), Estimate: estimate,
		InputURI:     fmt.Sprintf("gs://%s/%s", b.Bucket, object),
		OutputPrefix: fmt.Sprintf("gs://%s/%s", b.Bucket, strings.TrimSuffix(object, "input.jsonl")+"output")}
	body, _ := json.Marshal(map[string]any{
		"displayName":  "aidetector-" + s.Name() + "-" + stamp,
		"model":        fmt.Sprintf("publishers/%s/models/%s", publisher(s), s.Model),
		"inputConfig":  map[string]any{"instancesFormat": "jsonl", "gcsSource": map[string]any{"uris": []string{job.InputURI}}},
		"outputConfig": map[string]any{"predictionsFormat": "jsonl", "gcsDestination": map[string]any{"outputUriPrefix": job.OutputPrefix}},
		"labels":       map[string]string{"purpose": "aidetector", "night": night},
	})
	data, err := b.do(ctx, "POST", fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/batchPredictionJobs", host(loc), b.Project, loc),
		bytes.NewReader(body), "application/json")
	if err != nil {
		return nil, fmt.Errorf("create batch job: %w", err)
	}
	var created struct{ Name, State string }
	if err := json.Unmarshal(data, &created); err != nil {
		return nil, err
	}
	job.Name, job.State = created.Name, created.State
	return job, nil
}

// Resubmit creates a new job from a failed job's input file (no new upload).
func (b *Batcher) Resubmit(ctx context.Context, failed *BatchJob, night string) (*BatchJob, error) {
	stamp := time.Now().UTC().Format("20060102-150405")
	job := *failed
	job.Night, job.Submitted, job.Retried, job.State = night, time.Now().UTC(), true, ""
	job.OutputPrefix = strings.TrimSuffix(failed.OutputPrefix, "/") + "-retry-" + stamp
	body, _ := json.Marshal(map[string]any{
		"displayName":  "aidetector-retry-" + failed.Spec.Name() + "-" + stamp,
		"model":        fmt.Sprintf("publishers/%s/models/%s", publisher(failed.Spec), failed.Spec.Model),
		"inputConfig":  map[string]any{"instancesFormat": "jsonl", "gcsSource": map[string]any{"uris": []string{failed.InputURI}}},
		"outputConfig": map[string]any{"predictionsFormat": "jsonl", "gcsDestination": map[string]any{"outputUriPrefix": job.OutputPrefix}},
		"labels":       map[string]string{"purpose": "aidetector", "night": night},
	})
	data, err := b.do(ctx, "POST", fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/batchPredictionJobs", host(failed.Location), b.Project, failed.Location),
		bytes.NewReader(body), "application/json")
	if err != nil {
		return nil, err
	}
	var created struct{ Name, State string }
	if err := json.Unmarshal(data, &created); err != nil {
		return nil, err
	}
	job.Name, job.State = created.Name, created.State
	return &job, nil
}

// Refresh updates the job's state (JOB_STATE_SUCCEEDED, _FAILED, _RUNNING, ...).
func (b *Batcher) Refresh(ctx context.Context, job *BatchJob) error {
	data, err := b.do(ctx, "GET", fmt.Sprintf("https://%s/v1/%s", host(job.Location), job.Name), nil, "")
	if err != nil {
		return err
	}
	var j struct {
		State string `json:"state"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	job.State = j.State
	if j.Error.Message != "" && j.State == "JOB_STATE_FAILED" {
		job.State += ": " + j.Error.Message
	}
	return nil
}

// Done reports whether the job reached a final state.
func (j *BatchJob) Done() bool {
	return strings.HasPrefix(j.State, "JOB_STATE_SUCCEEDED") || strings.HasPrefix(j.State, "JOB_STATE_FAILED") ||
		strings.HasPrefix(j.State, "JOB_STATE_CANCELLED") || strings.HasPrefix(j.State, "JOB_STATE_EXPIRED") ||
		strings.HasPrefix(j.State, "JOB_STATE_PARTIALLY_SUCCEEDED")
}

// Results reads every output file under the job's prefix.
func (b *Batcher) Results(ctx context.Context, job *BatchJob) ([]BatchResult, error) {
	prefix := strings.TrimPrefix(job.OutputPrefix, "gs://"+b.Bucket+"/")
	data, err := b.do(ctx, "GET", fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=%s", b.Bucket, url.QueryEscape(prefix)), nil, "")
	if err != nil {
		return nil, err
	}
	var list struct {
		Items []struct{ Name string } `json:"items"`
	}
	if err := json.Unmarshal(data, &list); err != nil {
		return nil, err
	}
	var out []BatchResult
	for _, it := range list.Items {
		if !strings.HasSuffix(it.Name, ".jsonl") {
			continue
		}
		raw, err := b.do(ctx, "GET", fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media", b.Bucket, url.PathEscape(it.Name)), nil, "")
		if err != nil {
			return out, err
		}
		sc := bufio.NewScanner(bytes.NewReader(raw))
		sc.Buffer(make([]byte, 16<<20), 16<<20)
		for sc.Scan() {
			if r, ok := parseResultLine(job.Spec, sc.Bytes()); ok {
				out = append(out, r)
			}
		}
	}
	return out, nil
}

// parseResultLine reads one output line in either provider's format.
func parseResultLine(s Spec, line []byte) (BatchResult, bool) {
	var r BatchResult
	switch s.Provider {
	case "claude":
		var l struct {
			CustomID string          `json:"custom_id"`
			Response json.RawMessage `json:"response"`
			Status   string          `json:"status"`
			Error    json.RawMessage `json:"error"`
		}
		if json.Unmarshal(line, &l) != nil || l.CustomID == "" {
			return r, false
		}
		r.ID = l.CustomID
		var msg struct {
			Content []struct {
				Type, Text string
			} `json:"content"`
			StopReason string `json:"stop_reason"`
			Usage      struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
				CacheRead    int `json:"cache_read_input_tokens"`
			} `json:"usage"`
		}
		if len(l.Response) == 0 || json.Unmarshal(l.Response, &msg) != nil || len(msg.Content) == 0 && msg.StopReason == "" {
			r.Err = strings.TrimSpace(string(l.Error) + " " + l.Status)
			if r.Err == "" {
				r.Err = "no response"
			}
			return r, true
		}
		for _, c := range msg.Content {
			if c.Type == "text" {
				r.Text += c.Text
			}
		}
		r.Usage = Usage{Input: msg.Usage.InputTokens + msg.Usage.CacheRead, Output: msg.Usage.OutputTokens, CacheRead: msg.Usage.CacheRead}
		r.Refused = msg.StopReason == "refusal"
		if msg.StopReason == "max_tokens" {
			r.Err = "output truncated at max_tokens"
		}
		return r, true
	case "gemini":
		var l struct {
			Request struct {
				Labels map[string]string `json:"labels"`
			} `json:"request"`
			Response struct {
				Candidates []struct {
					Content struct {
						Parts []struct {
							Text    string `json:"text"`
							Thought bool   `json:"thought"`
						} `json:"parts"`
					} `json:"content"`
					FinishReason string `json:"finishReason"`
				} `json:"candidates"`
				Usage struct {
					Prompt     int `json:"promptTokenCount"`
					Candidates int `json:"candidatesTokenCount"`
					Thoughts   int `json:"thoughtsTokenCount"`
					Cached     int `json:"cachedContentTokenCount"`
				} `json:"usageMetadata"`
			} `json:"response"`
			Status string `json:"status"`
		}
		if json.Unmarshal(line, &l) != nil || l.Request.Labels["custom_id"] == "" {
			return r, false
		}
		r.ID = l.Request.Labels["custom_id"]
		u := l.Response.Usage
		r.Usage = Usage{Input: u.Prompt, Output: u.Candidates, Thinking: u.Thoughts, CacheRead: u.Cached}
		if len(l.Response.Candidates) == 0 {
			r.Err = "no candidates " + l.Status
			return r, true
		}
		c := l.Response.Candidates[0]
		for _, p := range c.Content.Parts {
			if !p.Thought {
				r.Text += p.Text
			}
		}
		r.Refused = c.FinishReason == "SAFETY" || c.FinishReason == "PROHIBITED_CONTENT"
		return r, true
	}
	return r, false
}
