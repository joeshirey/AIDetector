package corpus

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/joeshirey/AIDetector/internal/text"
)

// Cutoff is the default "definitely human" boundary: ChatGPT launched
// 2022-11-30, so anything before November 2022 is treated as human-written.
var Cutoff = time.Date(2022, 11, 1, 0, 0, 0, 0, time.UTC)

// Sample is one harvested human text plus whatever context a generator
// needs to write its AI counterpart.
type Sample struct {
	ID      string            `json:"id"`
	Kind    string            `json:"kind"`
	Source  string            `json:"source"`
	Text    string            `json:"text"`
	Context map[string]string `json:"context,omitempty"` // diff, title, tags, style examples…
}

const maxDiffBytes = 12_000

var (
	botRE     = regexp.MustCompile(`(?i)\[bot\]|dependabot|renovate|k8s-ci-robot|github-actions|greenkeeper|snyk|copybara|homu|bors`)
	htmlCmtRE = regexp.MustCompile(`(?s)<!--.*?-->`)
)

// NormalizePR removes HTML comments, which are template boilerplate.
func NormalizePR(body string) string {
	return strings.TrimSpace(htmlCmtRE.ReplaceAllString(strings.ReplaceAll(body, "\r\n", "\n"), ""))
}

func gh(ctx context.Context, args ...string) ([]byte, error) {
	out, err := exec.CommandContext(ctx, "gh", args...).Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("gh %s: %s", strings.Join(args[:min(2, len(args))], " "), strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, err
	}
	return out, nil
}

func truncateDiff(d string) string {
	if len(d) <= maxDiffBytes {
		return d
	}
	return d[:maxDiffBytes] + "\n[diff truncated]\n"
}

// HarvestPRs pulls merged, non-bot PRs with real descriptions.
func HarvestPRs(ctx context.Context, repo string, before time.Time, n int, lazy bool, seen func(id string) bool) ([]Sample, error) {
	raw, err := gh(ctx, "pr", "list", "--repo", repo, "--state", "merged",
		"--search", "merged:<"+before.Format("2006-01-02"), "--limit", fmt.Sprint(n*4+40),
		"--json", "number,title,body,author")
	if err != nil {
		return nil, err
	}
	var prs []struct {
		Number int    `json:"number"`
		Title  string `json:"title"`
		Body   string `json:"body"`
		Author struct {
			Login string `json:"login"`
			IsBot bool   `json:"is_bot"`
		} `json:"author"`
	}
	if err := json.Unmarshal(raw, &prs); err != nil {
		return nil, err
	}
	var out []Sample
	for _, pr := range prs {
		if len(out) >= n {
			break
		}
		body := NormalizePR(pr.Body)
		id := fmt.Sprintf("%s-%d", sanitize(repo), pr.Number)
		if pr.Author.IsBot || botRE.MatchString(pr.Author.Login) || len(strings.Fields(body)) < 25 || seen(id) {
			continue
		}
		sm := Sample{
			ID: id, Kind: "pr", Source: "github:" + repo, Text: text.Normalize("pr", body),
			Context: map[string]string{"repo": repo, "title": pr.Title, "number": fmt.Sprint(pr.Number)},
		}
		if !lazy && FetchDiff(ctx, &sm) != nil {
			continue
		}
		out = append(out, sm)
	}
	return out, nil
}

// FetchDiff fills in the diff for a PR sample harvested lazily.
func FetchDiff(ctx context.Context, sm *Sample) error {
	if sm.Context["diff"] != "" || sm.Kind != "pr" {
		return nil
	}
	repo := sm.Context["repo"]
	var diff []byte
	var err error
	diff, err = gh(ctx, "pr", "diff", sm.Context["number"], "--repo", repo)
	if err != nil {
		return err
	}
	if len(diff) == 0 {
		return fmt.Errorf("%s: empty diff", sm.ID)
	}
	sm.Context["diff"] = truncateDiff(string(diff))
	return nil
}

const mediumURL = "https://huggingface.co/datasets/fabiochiu/medium-articles/resolve/main/medium_articles.csv"

var workTags = map[string]bool{
	"programming": true, "software development": true, "software engineering": true, "technology": true,
	"tech": true, "startup": true, "startups": true, "leadership": true, "management": true, "productivity": true,
	"data science": true, "machine learning": true, "javascript": true, "python": true, "web development": true,
	"devops": true, "product management": true, "engineering": true, "business": true, "entrepreneurship": true,
	"work": true, "careers": true, "agile": true, "design": true, "ux": true, "cloud computing": true,
}

// HarvestMedium streams the Medium articles CSV and keeps work/tech posts of
// a readable length. It stops reading as soon as it has enough.
// phase picks which every-7th row to sample, so the pool (phase 3) and the
// original harvest (phase 0) draw different articles.
func HarvestMedium(ctx context.Context, before time.Time, n, phase int, seen func(id string) bool) ([]Sample, error) {
	return harvestMedium(ctx, before, n, phase, "medium", anyWorkTag, seen)
}

// HarvestMediumTech is HarvestMedium restricted to engineering tags.
func HarvestMediumTech(ctx context.Context, before time.Time, n, phase int, seen func(id string) bool) ([]Sample, error) {
	return harvestMedium(ctx, before, n, phase, "medium-tech", anyTechTag, seen)
}

func harvestMedium(ctx context.Context, before time.Time, n, phase int, source string, keep func([]string) bool, seen func(id string) bool) ([]Sample, error) {
	return harvestMediumFiltered(ctx, before, n, phase, source, keep, 250, nil, seen)
}

func harvestMediumFiltered(ctx context.Context, before time.Time, n, phase int, source string, keep func([]string) bool, minWords int, skip map[string]bool, seen func(id string) bool) ([]Sample, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", mediumURL, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	r := csv.NewReader(bufio.NewReaderSize(resp.Body, 1<<20))
	r.LazyQuotes = true
	if _, err := r.Read(); err != nil { // header: title,text,url,authors,timestamp,tags
		return nil, err
	}
	var out []Sample
	row := 0
	for len(out) < n {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil || len(rec) < 6 {
			continue
		}
		row++
		// Take every 7th match so we sample across the file, not one author's run.
		ts, err := time.Parse("2006-01-02", rec[4][:min(10, len(rec[4]))])
		if err != nil || !ts.Before(before) || row%7 != phase {
			continue
		}
		words := len(strings.Fields(rec[1]))
		tags := parseTags(rec[5])
		if words < minWords || words > 1500 || !keep(tags) || skip[mediumArticleID(rec[2])] {
			continue
		}
		if source == "medium-nontech" && !isEnglish(rec[1]) {
			continue
		}
		id := fmt.Sprintf("medium-%07d", row)
		if seen(id) {
			continue
		}
		out = append(out, Sample{
			ID: id, Kind: "blog", Source: source, Text: cleanMedium(rec[1], rec[0]),
			Context: map[string]string{"title": rec[0], "tags": strings.Join(tags, ", "), "words": fmt.Sprint(words)},
		})
	}
	return out, nil
}

var (
	blankRunRE    = regexp.MustCompile(`\n{3,}`)
	doubleSpaceRE = regexp.MustCompile(` {2,}`)
)

var photoCreditRE = regexp.MustCompile(`(?im)^\s*(photo|image|illustration|picture|gif|source)s?\b.{0,80}\b(by|via|from|courtesy|credit|unsplash|pexels|pixabay|shutterstock|giphy)\b.*$`)

// cleanMedium removes scraping artifacts the AI mirrors can't have: the title
// line (mirrors get the title separately) and image credits.
func cleanMedium(text, title string) string {
	text = strings.TrimSpace(text)
	if first, rest, ok := strings.Cut(text, "\n"); ok && strings.TrimSpace(first) == strings.TrimSpace(title) {
		text = rest
	}
	text = photoCreditRE.ReplaceAllString(text, "")
	return strings.TrimSpace(blankRunRE.ReplaceAllString(text, "\n\n"))
}

// mediumArticleID is the trailing hex ID on a Medium article URL.
func mediumArticleID(u string) string {
	u = strings.TrimRight(strings.TrimSpace(u), "/")
	if i := strings.LastIndex(u, "-"); i >= 0 {
		return u[i+1:]
	}
	return u
}

func parseTags(s string) []string {
	var tags []string
	for _, t := range strings.Split(strings.Trim(s, "[]"), ",") {
		if t = strings.Trim(strings.TrimSpace(t), `'"`); t != "" {
			tags = append(tags, t)
		}
	}
	return tags
}

func anyWorkTag(tags []string) bool {
	for _, t := range tags {
		if workTags[strings.ToLower(t)] {
			return true
		}
	}
	return false
}

// HarvestEnron pulls business emails (Enron, 2000–2002) from the AESLC
// dataset as a stand-in for internal memos.
func HarvestEnron(ctx context.Context, n int, seen func(id string) bool) ([]Sample, error) {
	var out []Sample
	for offset := 0; len(out) < n && offset < 14000; offset += 100 {
		u := "https://datasets-server.huggingface.co/rows?dataset=Yale-LILY/aeslc&config=default&split=train&length=100&offset=" + fmt.Sprint(offset)
		req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return out, err
		}
		var page struct {
			Rows []struct {
				RowIdx int `json:"row_idx"`
				Row    struct {
					Body    string `json:"email_body"`
					Subject string `json:"subject_line"`
				} `json:"row"`
			} `json:"rows"`
			Error string `json:"error"`
		}
		err = json.NewDecoder(resp.Body).Decode(&page)
		resp.Body.Close()
		if err != nil {
			return out, err
		}
		if page.Error != "" {
			return out, fmt.Errorf("aeslc: %s", page.Error)
		}
		for _, r := range page.Rows {
			body := cleanAESLC(r.Row.Body)
			words := len(strings.Fields(body))
			id := fmt.Sprintf("enron-%05d", r.RowIdx)
			// Longer emails read more like memos; forwarded chains aren't one author.
			if words < 80 || words > 800 || strings.Contains(body, "-----Original Message") ||
				strings.Contains(body, "Forwarded by") || seen(id) || len(out) >= n {
				continue
			}
			out = append(out, Sample{
				ID: id, Kind: "memo", Source: "aeslc", Text: text.Normalize("memo", body),
				Context: map[string]string{"subject": r.Row.Subject, "words": fmt.Sprint(words)},
			})
		}
	}
	return out, nil
}

// cleanAESLC undoes the dataset's preprocessing, which put each sentence on
// its own line and turned paragraph breaks into double spaces.
func cleanAESLC(body string) string {
	body = strings.ReplaceAll(strings.TrimSpace(body), "\n", " ")
	body = doubleSpaceRE.ReplaceAllString(body, "\n\n")
	return strings.TrimSpace(body)
}

func sanitize(s string) string {
	return strings.NewReplacer("/", "_", ".", "_").Replace(s)
}

func lastN(xs []string, n int) []string {
	if len(xs) > n {
		return xs[len(xs)-n:]
	}
	return xs
}
