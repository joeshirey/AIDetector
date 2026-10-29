package overnight

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/joeshirey/AIDetector/internal/corpus"
)

// TrackedJob is a batch job plus what's needed to file its results.
type TrackedJob struct {
	Job       *corpus.BatchJob  `json:"job"`
	Titles    map[string]string `json:"titles,omitempty"`  // request id -> PR title (for cleanup)
	Purpose   string            `json:"purpose,omitempty"` // train, frozen, or effort
	Collected bool              `json:"collected"`
	Written   int               `json:"written"`
	Failed    int               `json:"failed"`
	Actual    float64           `json:"actual"`
	Usage     corpus.Usage      `json:"usage"` // summed over results
	Note      string            `json:"note,omitempty"`
}

// Jobs is the persistent list of batch jobs across nights.
type Jobs struct {
	path string
	List []*TrackedJob `json:"jobs"`
}

func LoadJobs(path string) (*Jobs, error) {
	j := &Jobs{path: path}
	if data, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(data, j); err != nil {
			return nil, err
		}
	}
	return j, nil
}

func (j *Jobs) Save() error {
	data, _ := json.MarshalIndent(j, "", " ")
	return os.WriteFile(j.path, data, 0o644)
}

func (j *Jobs) Pending() []*TrackedJob {
	var out []*TrackedJob
	for _, t := range j.List {
		if !t.Collected {
			out = append(out, t)
		}
	}
	return out
}

// Collector files finished batch results into the corpus.
type Collector struct {
	Data    Data
	Batcher *corpus.Batcher
	Jobs    *Jobs
	OnWrite func(GenRecord) // called for every file written
	Log     io.Writer
}

// Collect refreshes pending jobs and files the results of finished ones. A
// failed job is resubmitted once (as a new submission tonight); a second
// failure is dropped and reported. Cost is corrected against the night the
// job was submitted.
func (c *Collector) Collect(ctx context.Context, night string) (collected, carried int, err error) {
	for _, t := range c.Jobs.Pending() {
		if ctx.Err() != nil {
			break
		}
		if err := c.Batcher.Refresh(ctx, t.Job); err != nil {
			fmt.Fprintf(c.Log, "batch %s: refresh: %v\n", t.Job.Spec.Name(), err)
			carried++
			continue
		}
		if !t.Job.Done() {
			carried++
			continue
		}
		if strings.HasPrefix(t.Job.State, "JOB_STATE_SUCCEEDED") || strings.HasPrefix(t.Job.State, "JOB_STATE_PARTIALLY_SUCCEEDED") {
			if err := c.file(ctx, t); err != nil {
				return collected, carried, err
			}
			collected++
		} else {
			t.Collected = true
			t.Note = t.Job.State
			// Undo the reservation for work that didn't happen.
			corpus.Spend.Record(corpus.Entry{Provider: "vertex-batch", Model: t.Job.Spec.Model, Mode: corpus.ModeBatch,
				Night: t.Job.Night, Purpose: "batch correction", JobID: t.Job.Name, Cost: -t.Job.Estimate})
			if !t.Job.Retried && corpus.Spend.Check(t.Job.Spec.Model) == nil {
				fmt.Fprintf(c.Log, "batch %s failed (%s); retrying once\n", t.Job.Spec.Name(), t.Job.State)
				nj, err := c.Batcher.Resubmit(ctx, t.Job, night)
				if err != nil {
					fmt.Fprintf(c.Log, "batch %s: retry failed to submit: %v\n", t.Job.Spec.Name(), err)
				} else {
					corpus.Spend.Record(corpus.Entry{Provider: "vertex-batch", Model: nj.Spec.Model, Mode: corpus.ModeBatch, Night: night,
						Purpose: "batch reservation (retry)", JobID: nj.Name, Estimated: true, Cost: nj.Estimate})
					c.Jobs.List = append(c.Jobs.List, &TrackedJob{Job: nj, Titles: t.Titles, Purpose: t.Purpose})
					t.Note += "; retried once"
				}
			} else {
				fmt.Fprintf(c.Log, "batch %s failed again (%s); dropped\n", t.Job.Spec.Name(), t.Job.State)
			}
		}
		if err := c.Jobs.Save(); err != nil {
			return collected, carried, err
		}
	}
	return collected, carried, nil
}

func (c *Collector) file(ctx context.Context, t *TrackedJob) error {
	res, err := c.Batcher.Results(ctx, t.Job)
	if err != nil {
		return err
	}
	actual := 0.0
	var usage corpus.Usage
	for _, r := range res {
		actual += corpus.Spend.Cost(t.Job.Spec.Model, corpus.ModeBatch, r.Usage)
		usage.Input += r.Usage.Input
		usage.Output += r.Usage.Output
		usage.Thinking += r.Usage.Thinking
		usage.CacheRead += r.Usage.CacheRead
		m, err := DecodeReq(r.ID)
		if err != nil || r.Err != "" || r.Refused {
			t.Failed++
			continue
		}
		g, ok := GenreByName(m.Genre)
		if !ok {
			t.Failed++
			continue
		}
		root := ""
		switch m.Track {
		case "frozen":
			root = c.Data.Path("corpus_frozen")
		case "effort":
			root = c.Data.Path("runs", "effort")
		}
		path, ok, err := c.Data.WriteOutput(g, t.Job.Spec.Name(), m, r.Text, t.Titles[r.ID], root)
		if err != nil {
			return err
		}
		if !ok {
			t.Failed++
			continue
		}
		t.Written++
		if c.OnWrite != nil {
			c.OnWrite(GenRecord{Path: path, Cell: Cell{Genre: m.Genre, Band: m.Band, Style: m.Style, Gen: t.Job.Spec.Name()},
				Track: m.Track, Night: t.Job.Night, Mode: corpus.ModeBatch})
		}
	}
	t.Actual, t.Usage = actual, usage
	t.Collected = true
	corpus.Spend.Record(corpus.Entry{Provider: "vertex-batch", Model: t.Job.Spec.Model, Mode: corpus.ModeBatch,
		Night: t.Job.Night, Purpose: "batch correction", JobID: t.Job.Name, Cost: actual - t.Job.Estimate, Usage: usage, N: len(res)})
	fmt.Fprintf(c.Log, "batch %s: %d written, %d failed; estimated $%.2f, actual $%.2f\n",
		t.Job.Spec.Name(), t.Written, t.Failed, t.Job.Estimate, actual)
	return nil
}

// PlannedRequest is one request ready to send.
type PlannedRequest struct {
	Meta   ReqMeta
	Prompt string
	Title  string
	Cost   float64 // estimate
}

// PlanPaid builds batch requests for one paid generator until its dollar
// allocation (or maxN) is used. Sources are picked first (with a rough
// estimate), then briefs are written in parallel and the estimate redone.
func PlanPaid(ctx context.Context, p *Planner, br *Briefer, gen corpus.Generator, allocation float64, maxN int, rng *rand.Rand) ([]PlannedRequest, float64) {
	type pick struct {
		c   Cell
		src Source
	}
	var picks []pick
	rough := 0.0
	model := gen.Spec().Model
	for len(picks) < maxN && rough < allocation && ctx.Err() == nil {
		c, src, ok := p.Next(gen.Name())
		if !ok {
			break
		}
		picks = append(picks, pick{c, src})
		// A brief is about 250 words; edit and continue prompts carry the text.
		rough += corpus.Spend.Cost(model, corpus.ModeBatch, EstimateUsage(gen.Spec(), strings.Repeat("word ", 300+src.Words), src.Words))
	}
	briefs := make([]string, len(picks))
	ok := make([]bool, len(picks))
	var wg sync.WaitGroup
	sem := make(chan struct{}, 8)
	for i, pk := range picks {
		if !needsBrief(pk.src.Genre, pk.c.Style) {
			ok[i] = true
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			b, err := br.Brief(ctx, pk.src.Genre, pk.src.Sample)
			briefs[i], ok[i] = b, err == nil
		}()
	}
	wg.Wait()
	var out []PlannedRequest
	spent := 0.0
	for i, pk := range picks {
		if !ok[i] || spent >= allocation {
			continue
		}
		prompt, words, err := BuildPrompt(pk.src.Genre, pk.c.Style, pk.src.Sample, briefs[i], rng)
		if err != nil {
			continue
		}
		cost := corpus.Spend.Cost(model, corpus.ModeBatch, EstimateUsage(gen.Spec(), prompt, words))
		m := ReqMeta{Genre: pk.c.Genre, ID: pk.src.Sample.ID, Style: pk.c.Style, Band: pk.c.Band, Track: "train"}
		out = append(out, PlannedRequest{Meta: m, Prompt: prompt, Title: pk.src.Sample.Context["title"], Cost: cost})
		spent += cost
	}
	return out, spent
}

// SubmitPaid sends planned requests as one batch job and reserves its
// estimated cost against tonight.
func SubmitPaid(ctx context.Context, b *corpus.Batcher, jobs *Jobs, spec corpus.Spec, night, purpose string, reqs []PlannedRequest) (*TrackedJob, error) {
	if len(reqs) == 0 {
		return nil, nil
	}
	var br []corpus.BatchRequest
	titles := map[string]string{}
	est := 0.0
	for _, r := range reqs {
		id := r.Meta.Encode()
		br = append(br, corpus.BatchRequest{ID: id, Prompt: r.Prompt})
		if r.Title != "" {
			titles[id] = r.Title
		}
		est += r.Cost
	}
	if err := corpus.Spend.Check(spec.Model); err != nil {
		return nil, err
	}
	job, err := b.Submit(ctx, spec, night, br, est)
	if err != nil {
		return nil, err
	}
	corpus.Spend.Record(corpus.Entry{Provider: "vertex-batch", Model: spec.Model, Mode: corpus.ModeBatch, Night: night,
		Purpose: "batch reservation (" + purpose + ")", JobID: job.Name, Estimated: true, Cost: est})
	t := &TrackedJob{Job: job, Titles: titles, Purpose: purpose}
	jobs.List = append(jobs.List, t)
	return t, jobs.Save()
}

// LocalStats summarizes a local worker's night.
type LocalStats struct {
	Gen             string
	Written, Failed int
	OutputTokens    int
	Seconds         float64
	Skipped         string // why the worker didn't run, if it didn't
}

func (s LocalStats) TokensPerSec() float64 {
	if s.Seconds == 0 {
		return 0
	}
	return float64(s.OutputTokens) / s.Seconds
}

// RunLocal generates with a local model until stopAt or quota, sequentially
// per worker (LM Studio serves a few requests in parallel per model).
func RunLocal(ctx context.Context, p *Planner, br *Briefer, gen corpus.Generator, d Data, stopAt time.Time, quota, workers int,
	night string, onWrite func(GenRecord), rng *rand.Rand) LocalStats {
	st := LocalStats{Gen: gen.Name()}
	var mu sync.Mutex
	var wg sync.WaitGroup
	done := 0
	// Throughput is total output over wall-clock time, across all workers.
	start, before := time.Now(), corpus.Spend.LocalOutput(gen.Spec().Model)
	for w := range workers {
		wg.Add(1)
		go func(seed uint64) {
			defer wg.Done()
			r := rand.New(rand.NewPCG(seed, seed+1))
			for ctx.Err() == nil && time.Now().Before(stopAt) {
				mu.Lock()
				if done >= quota {
					mu.Unlock()
					return
				}
				done++
				mu.Unlock()
				c, src, ok := p.Next(gen.Name())
				if !ok {
					return
				}
				brief := ""
				if needsBrief(src.Genre, c.Style) {
					b, err := br.Brief(ctx, src.Genre, src.Sample)
					if err != nil {
						mu.Lock()
						st.Failed++
						mu.Unlock()
						continue
					}
					brief = b
				}
				prompt, _, err := BuildPrompt(src.Genre, c.Style, src.Sample, brief, r)
				if err != nil {
					continue
				}
				out, err := gen.Generate(ctx, prompt)
				if err != nil {
					mu.Lock()
					st.Failed++
					mu.Unlock()
					continue
				}
				m := ReqMeta{Genre: c.Genre, ID: src.Sample.ID, Style: c.Style, Band: c.Band, Track: "train"}
				path, ok, err := d.WriteOutput(src.Genre, gen.Name(), m, out, src.Sample.Context["title"], "")
				mu.Lock()
				if err != nil || !ok {
					st.Failed++
				} else {
					st.Written++
					if onWrite != nil {
						onWrite(GenRecord{Path: path, Cell: c, Track: "train", Night: night, Mode: corpus.ModeLocal})
					}
				}
				mu.Unlock()
			}
		}(rng.Uint64() + uint64(w))
	}
	wg.Wait()
	st.Seconds = time.Since(start).Seconds()
	st.OutputTokens = corpus.Spend.LocalOutput(gen.Spec().Model) - before
	return st
}
