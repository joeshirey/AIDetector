package overnight

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/joeshirey/AIDetector/internal/corpus"
	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/eval"
)

// Report collects everything the morning report shows.
type Report struct {
	Night             string
	Started, Finished time.Time
	Cfg               Config
	State             *State
	Error             string

	Manifest     *Manifest
	Frozen       []Verification
	FrozenBroken bool
	Billing      Reconciliation
	Prod         *Metrics
	Genres       *GenreReport
	Polished     []LabelDist
	LOFO         []LOFORow
}

// HistoryLine is one night's trend data.
type HistoryLine struct {
	Night    string        `json:"night"`
	Method   string        `json:"method,omitempty"` // "v2": checking-half, continuous-score measurement (2026-10-29 on)
	Prod     map[string]AR `json:"prod"`             // frozen sets and slices, production model
	Accepted int           `json:"accepted"`
	Spent    float64       `json:"spent"`
}

func (l *Loop) historyPath() string { return filepath.Join(l.Cfg.RunRoot, "history.jsonl") }

func (l *Loop) readHistory() []HistoryLine {
	b, err := os.ReadFile(l.historyPath())
	if err != nil {
		return nil
	}
	var out []HistoryLine
	for _, line := range strings.Split(string(b), "\n") {
		var h HistoryLine
		if json.Unmarshal([]byte(line), &h) == nil && h.Night != "" && h.Night != l.Night {
			out = append(out, h)
		}
	}
	return out
}

func pct(v float64) string { return fmt.Sprintf("%.1f%%", v*100) }

func (l *Loop) writeReport(ctx context.Context, rep *Report) (string, error) {
	var b strings.Builder
	w := func(format string, a ...any) { fmt.Fprintf(&b, format+"\n", a...) }
	st := rep.State
	entries, _ := ReadLedger(l.Cfg.Ledger)
	spend := SpendFor(entries, l.Night)

	title := "Overnight report"
	if l.Cfg.DryRun {
		title = "Overnight dry run"
	}
	w("# %s, night of %s", title, l.Night)
	w("")
	w("Run %s to %s. Night's spending cap $%.2f (-budget-usd %.2f).", rep.Started.Format("15:04"), rep.Finished.Format("15:04"), st.Limit, l.Cfg.BudgetUSD)
	if rep.Error != "" {
		w("")
		w("**The run stopped early:** %s", rep.Error)
	}
	w("")

	// Recommendation first: it's what Joe reads.
	w("## Recommendation")
	w("")
	w("%s", recommendation(rep))
	w("")

	if len(st.Notes) > 0 {
		w("## Flags and notes")
		w("")
		for _, n := range st.Notes {
			w("- %s", n)
		}
		w("")
	}

	w("## Spend")
	w("")
	w("Ledger total for the night: **$%.2f** (batch jobs count in the night they were submitted; later corrections adjust that night).", spend.Total)
	w("")
	w("| Model | Spend |")
	w("|---|---:|")
	for _, m := range sortedKeys(spend.ByModel) {
		w("| %s | $%.2f |", m, spend.ByModel[m])
	}
	w("")
	w("Billing reconciliation: %s", rep.Billing.Note)
	for _, n := range rep.Billing.Nights {
		w("- %s: ledger $%.2f, billed $%.2f", n.Night, n.Ledger, n.Billed)
	}
	w("")
	w("### Tokens per call and local throughput")
	w("")
	w("| Model / mode | Calls | Input | Output | Thinking | Cache read | Output tok/s |")
	w("|---|---:|---:|---:|---:|---:|---:|")
	for _, k := range sortedKeys(spend.Usage) {
		u, n := spend.Usage[k], spend.Calls[k]
		tps := "-"
		if u.Seconds > 0 {
			tps = fmt.Sprintf("%.1f", float64(u.Output+u.Thinking)/u.Seconds)
		}
		w("| %s | %d | %d | %d | %d | %d | %s |", k, n, u.Input/max(n, 1), u.Output/max(n, 1), u.Thinking/max(n, 1), u.CacheRead/max(n, 1), tps)
	}
	w("")
	for _, s := range st.Local {
		if s.Skipped != "" {
			w("- %s: skipped (%s)", s.Gen, s.Skipped)
		} else {
			w("- %s: %d written, %d failed, %.1f output tok/s (all workers, over wall-clock time)", s.Gen, s.Written, s.Failed, s.TokensPerSec())
		}
	}
	w("")

	w("## Batch jobs")
	w("")
	w("Collected %d finished jobs; %d still running and carried to tomorrow.", st.Batches.Collected, st.Batches.Carried)
	if len(st.Batches.Submitted) == 0 {
		w("Nothing submitted tonight.")
	}
	for _, s := range st.Batches.Submitted {
		w("- submitted %s", s)
	}
	if len(st.Batches.Shares) > 0 {
		var parts []string
		for _, g := range sortedKeys(st.Batches.Shares) {
			parts = append(parts, fmt.Sprintf("%s %s", g, pct(st.Batches.Shares[g])))
		}
		w("")
		w("Paid shares after the bandit: %s.", strings.Join(parts, ", "))
	}
	w("")

	w("## Samples")
	w("")
	w("Human false positives promoted (train split, main pools, per-source cap %d): %s.", l.Cfg.SourceCap, kv(st.Mined))
	w("")
	w("AI misses (generated text scored under 60 by production):")
	w("")
	for _, g := range sortedKeys(st.Misses) {
		v := st.Misses[g]
		w("- %s: %d of %d (%s)", g, v[0], v[1], pct(float64(v[0])/float64(max(v[1], 1))))
	}
	w("")
	w("### Cells the bandit favors")
	w("")
	w("%s", topCells(l.bandit, 10, st.Batches.Shares))
	w("")

	w("## Gate")
	w("")
	if len(st.Rounds) == 0 {
		w("No retraining rounds ran.")
	}
	for _, r := range st.Rounds {
		verdict := "rejected"
		if r.Accepted {
			verdict = "**accepted**"
		}
		w("### Round %d: %s", r.N, verdict)
		w("")
		w("Candidate in `%s`, fit on %d training files.", r.Dir, r.Samples)
		if r.Err != "" {
			w("")
			w("Error: %s", r.Err)
			w("")
			continue
		}
		w("")
		w("| Check | Production | Candidate | Allowed | Pass |")
		w("|---|---:|---:|---:|:---:|")
		for _, c := range r.Checks {
			mark := "yes"
			if !c.Pass {
				mark = "**NO**"
			}
			allow := fmt.Sprintf("-%.3f", c.Tolerance)
			if !c.HigherBetter {
				allow = fmt.Sprintf("+%.3f", c.Tolerance)
			}
			w("| %s | %.4f | %.4f | %s | %s |", c.Name, c.Before, c.After, allow, mark)
		}
		w("")
	}

	if rep.Prod != nil {
		w("## Monitoring set")
		w("")
		w("%s", monitoringChanges(l, rep))
		w("")
		w("## Argon and Gemma")
		w("")
		w("%s", trends(l, rep))
		w("")
	}

	if g := rep.Genres; g != nil {
		w("## Per-genre cutoffs (measured only; nothing changed)")
		w("")
		w("| Genre | Shared likely/AI | Own likely/AI | Human FP shared (likely, AI) | Human FP own | AI recall shared | AI recall own | Humans / AI |")
		w("|---|---|---|---|---|---:|---:|---|")
		for _, c := range g.Cutoffs {
			w("| %s | %d/%d | %d/%d | %s, %s | %s, %s | %s | %s | %d / %d |", c.Genre, c.Shared.LikelyAI, c.Shared.AI, c.Own.LikelyAI, c.Own.AI,
				pct(c.FPRShared[0]), pct(c.FPRShared[1]), pct(c.FPROwn[0]), pct(c.FPROwn[1]), pct(c.RecallShared), pct(c.RecallOwn), c.Human, c.AI)
		}
		w("")
		rt := g.Routing
		w("Genre detection (Naive Bayes on words): accuracy %s; %s fell back to the strictest cutoff (posterior under 0.9).", pct(rt.Accuracy), pct(rt.Fallback))
		w("")
		w("| Cutoffs used | Human FP at likely-AI | AI recall at likely-AI |")
		w("|---|---:|---:|")
		for _, s := range []string{"shared", "true genre", "detected genre"} {
			w("| %s | %s | %s |", s, pct(rt.FPR[s]), pct(rt.Recall[s]))
		}
		w("")
		w("Cost of detection errors (detected minus true genre): FP %+.1f points, recall %+.1f points.",
			(rt.FPR["detected genre"]-rt.FPR["true genre"])*100, (rt.Recall["detected genre"]-rt.Recall["true genre"])*100)
		w("")
		w("Confusion (true → predicted): %s", confusion(rt.Confusion))
		w("")
	}

	if len(rep.Polished) > 0 {
		w("## Polished and humanized sets (eval only; measured only)")
		w("")
		w("| Set | N | human | likely human | inconclusive | likely AI | AI |")
		w("|---|---:|---:|---:|---:|---:|---:|")
		for _, d := range rep.Polished {
			c := func(l string) string { return pct(float64(d.Labels[l]) / float64(d.N)) }
			w("| %s | %d | %s | %s | %s | %s | %s |", d.Set, d.N, c("human"), c("likely human"), c("inconclusive"), c("likely AI"), c("AI"))
		}
		w("")
	}

	if len(rep.LOFO) > 0 {
		w("## Leave one family out")
		w("")
		w("| Kind | Family held out | AUROC held out | AUROC production | Recall@1%%FP held out | Recall@1%%FP production | Note |")
		w("|---|---|---:|---:|---:|---:|---|")
		for _, r := range rep.LOFO {
			w("| %s | %s | %.4f | %.4f | %s | %s | %s |", r.Kind, r.Family, r.Held.AUROC, r.Full.AUROC, pct(r.Held.Recall1), pct(r.Full.Recall1), r.Err)
		}
		w("")
	}

	w("## Frozen-set checksums")
	w("")
	if len(rep.Frozen) == 0 {
		w("Not verified (no manifest).")
	}
	for _, v := range rep.Frozen {
		status := "ok"
		if !v.OK() {
			status = fmt.Sprintf("**CHANGED**: %d changed, %d missing, %d added", v.Changed, v.Missing, v.Added)
		}
		w("- %s (%d files): %s", v.Set, v.Files, status)
	}
	w("- final-holdout: not read by the loop (verify by hand with `corpus overnight verify -holdout`)")
	w("")

	w("## Sonnet misses")
	w("")
	w("%s", l.sonnetAnalysis(ctx))
	w("")

	path := filepath.Join(l.Cfg.ReportDir, l.Night+".md")
	if l.Cfg.DryRun {
		path = filepath.Join(l.Cfg.ReportDir, l.Night+"-dry-run.md")
	}
	if err := os.MkdirAll(l.Cfg.ReportDir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return "", err
	}
	if rep.Prod != nil && !l.Cfg.DryRun {
		h := HistoryLine{Night: l.Night, Method: "v2", Prod: map[string]AR{}, Accepted: st.Accepted, Spent: spend.Total}
		for k, v := range rep.Prod.Frozen {
			h.Prod[k] = v
		}
		for k, v := range rep.Prod.Slices {
			h.Prod[k] = v
		}
		for k := range h.Prod { // the history doesn't need per-sample scores
			v := h.Prod[k]
			v.S = nil
			h.Prod[k] = v
		}
		// Count runs in a row without a candidate worth promoting (the
		// paid-generation stopping rule).
		streak := l.readStreak()
		if promotable(rep) {
			streak.Runs = 0
		} else {
			streak.Runs++
		}
		streak.Updated = time.Now().Format(time.RFC3339)
		if sb, err := json.Marshal(streak); err == nil {
			os.WriteFile(l.streakPath(), sb, 0o644)
		}
		hb, _ := json.Marshal(h)
		if f, err := os.OpenFile(l.historyPath(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err == nil {
			f.Write(append(hb, '\n'))
			f.Close()
		}
	}
	return path, nil
}

func kv(m map[string]int) string {
	var parts []string
	for _, k := range sortedKeys(m) {
		parts = append(parts, fmt.Sprintf("%s %d", k, m[k]))
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, ", ")
}

func confusion(c map[string]map[string]int) string {
	var parts []string
	for _, t := range sortedKeys(c) {
		for _, p := range sortedKeys(c[t]) {
			if p != t {
				parts = append(parts, fmt.Sprintf("%s→%s %d", t, p, c[t][p]))
			}
		}
	}
	if len(parts) == 0 {
		return "no errors"
	}
	return strings.Join(parts, ", ")
}

// topCells lists the cells with the highest miss rates among those the
// bandit can still pick: training styles and tonight's generators (when
// known), so retired cells (edits, paused models) don't clutter it.
func topCells(b *Bandit, n int, active map[string]float64) string {
	type row struct {
		key string
		s   *Stat
	}
	var rows []row
	for k, s := range b.Cells {
		parts := strings.Split(k, "|")
		if s.N < 3 || len(parts) != 4 || !slices.Contains(TrainStyles, Style(parts[2])) {
			continue
		}
		if _, ok := active[parts[3]]; len(active) > 0 && !ok {
			continue
		}
		rows = append(rows, row{k, s})
	}
	if len(rows) == 0 {
		return "Not enough scored samples yet (cells need 3+)."
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].s.Rate() > rows[j].s.Rate() })
	var out []string
	out = append(out, "| Genre · band · style · generator | Miss rate | Scored |", "|---|---:|---:|")
	for _, r := range rows[:min(n, len(rows))] {
		out = append(out, fmt.Sprintf("| %s | %s | %d |", strings.ReplaceAll(r.key, "|", " · "), pct(r.s.Rate()), r.s.N))
	}
	return strings.Join(out, "\n")
}

// promotable reports whether a round is worth promoting under the rule in
// recommendation.
func promotable(rep *Report) bool {
	return strings.HasPrefix(recommendation(rep), "**Consider promoting")
}

func recommendation(rep *Report) string {
	st := rep.State
	if rep.FrozenBroken {
		return "**Keep production.** A sealed frozen set changed, so tonight's gate isn't trustworthy. Check the checksum section."
	}
	// Passing the gate means "no worse"; promotion also needs a real gain,
	// and no detection figure may fall by more than its margin even when the
	// drop isn't statistically significant. A gain is at least 1 point of
	// test recall at 1% false positives in some kind, or a net reduction in
	// human documents at 60+ of at least 0.25% of all pool documents (at
	// least 3), counted in documents so small pools don't dominate.
	var best *Round
	bestScore := 0.0
	var bestRecall, bestDocs float64
	poolN := map[string]int{}
	if rep.Prod != nil {
		for p, n := range rep.Prod.PoolN {
			poolN[p] = n[0]
		}
	}
	totalDocs := 0
	for _, n := range poolN {
		totalDocs += n
	}
	needDocs := max(3, 0.0025*float64(totalDocs))
	for i := range st.Rounds {
		r := &st.Rounds[i]
		if !r.Accepted {
			continue
		}
		recall, docs, dropped := 0.0, 0.0, false
		for _, c := range r.Checks {
			worse := c.Before - c.After
			if !c.HigherBetter {
				worse = c.After - c.Before
			}
			if c.HigherBetter && worse > c.Tolerance+1e-9 {
				dropped = true
			}
			switch {
			case strings.HasPrefix(c.Name, "test recall@1%FP"):
				recall = max(recall, c.After-c.Before)
			case strings.HasPrefix(c.Name, "doc FPR 60+ "):
				docs += (c.Before - c.After) * float64(poolN[strings.TrimPrefix(c.Name, "doc FPR 60+ ")])
			}
		}
		if dropped || (recall < 0.01 && docs < needDocs) {
			continue
		}
		if score := recall*100 + docs; best == nil || score > bestScore {
			best, bestScore, bestRecall, bestDocs = r, score, recall, docs
		}
	}
	if best == nil {
		accepted := 0
		for _, r := range st.Rounds {
			if r.Accepted {
				accepted++
			}
		}
		if accepted > 0 {
			return fmt.Sprintf("**Keep production.** Tonight's candidate passed the gate (no significant regression) but isn't worth promoting: it needs at least 1 point more test recall or %.0f fewer human documents at 60+, with no detection figure down by more than its margin. New data stays in the corpus.", needDocs)
		}
		return "**Keep production.** No candidate passed every gate check tonight. New data stays in the corpus for future rounds."
	}
	return fmt.Sprintf("**Consider promoting round %d** (`%s`): it passed every gate check, gained %+.1f points of test recall at 1%% false positives (best kind), and flags %.0f fewer human documents at 60+ across the pools. "+
		"Before promoting: one logged look at the final holdout (`corpus overnight holdout -name ...`), the fixtures comparison, and the texts it newly flags (`flips`). Then copy its model files into internal/engine/models/, run `go test ./...`, and commit. The loop never does this itself.",
		best.N, best.Dir, bestRecall*100, bestDocs)
}

func monitoringChanges(l *Loop, rep *Report) string {
	var cand *Metrics
	for i := len(rep.State.Rounds) - 1; i >= 0; i-- {
		r := rep.State.Rounds[i]
		if b, err := os.ReadFile(filepath.Join(r.Dir, "metrics.json")); err == nil {
			var m Metrics
			if json.Unmarshal(b, &m) == nil {
				cand = &m
				break
			}
		}
	}
	prod := map[string]string{}
	for _, row := range rep.Prod.Fixtures.Rows {
		if row.Monitoring {
			prod[row.ID] = fmt.Sprintf("%d (%s)", row.Doc, row.Label)
		}
	}
	if cand == nil {
		return fmt.Sprintf("%d monitoring fixtures scored with production; no candidate to compare.", len(prod))
	}
	var changes []string
	for _, row := range cand.Fixtures.Rows {
		if !row.Monitoring {
			continue
		}
		now := fmt.Sprintf("%d (%s)", row.Doc, row.Label)
		if p, ok := prod[row.ID]; ok && !labelOf(p, now) {
			changes = append(changes, fmt.Sprintf("- %s: %s → %s", row.ID, p, now))
		}
	}
	if len(changes) == 0 {
		return "No monitoring-set label changes in the last candidate."
	}
	return "Label changes in the last candidate:\n\n" + strings.Join(changes, "\n")
}

func labelOf(a, b string) bool {
	la := a[strings.Index(a, "("):]
	lb := b[strings.Index(b, "("):]
	return la == lb
}

func trends(l *Loop, rep *Report) string {
	hist := append(l.readHistory(), HistoryLine{Night: l.Night, Prod: mergeAR(rep.Prod.Frozen, rep.Prod.Slices)})
	keys := []string{"gemini-4-argon/prose", "gemini-4-argon/pr", GemmaModel + "/prose", GemmaModel + "/pr"}
	var out []string
	out = append(out, "Production model on the frozen Argon and Gemma sets (Argon is in training since 2026-10-29; Gemma is evaluation-only):", "",
		"| Night | "+strings.Join(keys, " | ")+" |", "|---"+strings.Repeat("|---", len(keys))+"|")
	start := max(0, len(hist)-14)
	for _, h := range hist[start:] {
		row := "| " + h.Night
		for _, k := range keys {
			if v, ok := h.Prod[k]; ok {
				row += fmt.Sprintf(" | %.4f / %s", v.AUROC, pct(v.Recall1))
			} else {
				row += " | -"
			}
		}
		out = append(out, row+" |")
	}
	out = append(out, "", "Cells are AUROC / recall at 1% false positives.")
	return strings.Join(out, "\n")
}

func mergeAR(ms ...map[string]AR) map[string]AR {
	out := map[string]AR{}
	for _, m := range ms {
		for k, v := range m {
			out[k] = v
		}
	}
	return out
}

// sonnetAnalysis asks Sonnet (one call) what the lowest-scoring Sonnet
// texts have in common.
func (l *Loop) sonnetAnalysis(ctx context.Context) string {
	dirs := l.Cfg.Data.frozenDirs(detect.KindProse)
	split := ""
	keep := onlyGens(func(g string) bool { return g == "claude-sonnet-5-5" })
	r, err := eval.RunFiltered(ctx, l.prod, dirs, detect.KindProse, split, keep)
	if err != nil { // frozen set not generated yet: use the test split
		r, err = eval.RunFiltered(ctx, l.prod, l.Cfg.Data.TrainDirs(detect.KindProse), detect.KindProse, "test", keep)
	}
	if err != nil {
		return "No Sonnet samples to analyze: " + err.Error()
	}
	var ai []eval.Sample
	for _, s := range r.Samples {
		if s.IsAI {
			ai = append(ai, s)
		}
	}
	slices.SortFunc(ai, func(a, b eval.Sample) int { return a.Score - b.Score })
	var ex []string
	for i, s := range ai[:min(6, len(ai))] {
		words := strings.Fields(readText(s.Path))
		ex = append(ex, fmt.Sprintf("Text %d (score %d of 100, %d words):\n%s", i+1, s.Score, len(words), strings.Join(words[:min(160, len(words))], " ")))
	}
	if len(ex) == 0 {
		return "No Sonnet samples to analyze."
	}
	gens, err := corpus.NewGenerators(ctx, l.Cfg.Analyst)
	if err != nil {
		return "Analysis skipped: " + err.Error()
	}
	prompt := "These texts were written by Claude Sonnet, but a stylometric AI-text detector scored them as human-like (low scores). " +
		"The detector uses word-choice markers, sentence-length variation, punctuation habits, and word n-grams. " +
		"In at most 10 short lines, say what these texts have in common that the detector likely misses, and which kind of training text would help. No preamble.\n\n" +
		strings.Join(ex, "\n\n")
	out, err := gens[0].Generate(ctx, prompt)
	if err != nil {
		return "Analysis skipped: " + err.Error()
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	return strings.Join(lines[:min(10, len(lines))], "\n")
}

// localAvailable checks that the local server is up and (for LM Studio)
// lists the model.
func localAvailable(ctx context.Context, s corpus.Spec) error {
	model := s.Model
	base, def := os.Getenv("LMSTUDIO_URL"), "http://127.0.0.1:1234/v1"
	if s.Provider == "mlx" {
		base, def = os.Getenv("MLX_URL"), "http://127.0.0.1:8081/v1"
	}
	if base == "" {
		base = def
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", strings.TrimRight(base, "/")+"/models", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("LM Studio not reachable: %v", err)
	}
	defer resp.Body.Close()
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return err
	}
	if s.Provider == "mlx" {
		return nil // serves whatever it was started with
	}
	for _, m := range list.Data {
		if m.ID == model || strings.HasSuffix(m.ID, "/"+model) {
			return nil
		}
	}
	return fmt.Errorf("LM Studio doesn't list %s", model)
}
