package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/joeshirey/AIDetector/internal/corpus"
)

// runBatch is a small tool for checking the Vertex batch round trip:
//
//	corpus batch submit -gens claude:claude-haiku-5-5   (two one-line prompts per generator)
//	corpus batch poll                                    (refresh state; print results when done)
func runBatch(ctx context.Context, args []string, gens, bucket, jobsFile string) error {
	b, err := corpus.NewBatcher(ctx, bucket)
	if err != nil {
		return err
	}
	var jobs []*corpus.BatchJob
	if raw, err := os.ReadFile(jobsFile); err == nil {
		json.Unmarshal(raw, &jobs)
	}
	switch {
	case len(args) > 0 && args[0] == "submit":
		for _, g := range strings.Split(gens, ",") {
			s, err := corpus.ParseSpec(g)
			if err != nil {
				return err
			}
			reqs := []corpus.BatchRequest{{ID: "t1", Prompt: "Reply with exactly: ok"}, {ID: "t2", Prompt: "Reply with exactly: yes"}}
			j, err := b.Submit(ctx, s, corpus.Spend.Night.Format("2006-01-02"), reqs, 0.001)
			if err != nil {
				fmt.Printf("%s: %v\n", s.Name(), err)
				continue
			}
			fmt.Printf("%s: submitted %s (%s)\n", s.Name(), j.Name, j.State)
			jobs = append(jobs, j)
		}
	default:
		for _, j := range jobs {
			if err := b.Refresh(ctx, j); err != nil {
				return err
			}
			fmt.Printf("%s: %s\n", j.Spec.Name(), j.State)
			if strings.HasPrefix(j.State, "JOB_STATE_SUCCEEDED") {
				res, err := b.Results(ctx, j)
				if err != nil {
					return err
				}
				for _, r := range res {
					fmt.Printf("   %s %q usage=%+v err=%q\n", r.ID, strings.TrimSpace(r.Text), r.Usage, r.Err)
				}
			}
		}
	}
	data, _ := json.MarshalIndent(jobs, "", " ")
	return os.WriteFile(jobsFile, data, 0o644)
}
