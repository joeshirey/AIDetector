package corpus

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"
)

var (
	frontMatterRE = regexp.MustCompile(`(?s)\A---\n(.*?)\n---\n`)
	goDateRE      = regexp.MustCompile(`(?m)^date:\s*(\d{4}-\d{2}-\d{2})`)
	goTitleRE     = regexp.MustCompile(`(?m)^title:\s*"?(.+?)"?\s*$`)
	goDirectiveRE = regexp.MustCompile(`(?m)^\s*(?:\{\{.*\}\}|\.(?:code|play|image|html|iframe|video|caption)\b.*)$`)
	indentCodeRE  = regexp.MustCompile(`(?m)(?:^(?:\t| {4}).*\n?)+`)
	mdLinkRE      = regexp.MustCompile(`\[([^\]]+)\]\([^)]+\)`)
	htmlTagRE     = regexp.MustCompile(`<[^>\n]+>`)
)

// reflow joins the Go blog's one-clause-per-line source formatting back into
// paragraphs, so line structure can't give the source away.
func reflow(s string) string {
	var paras []string
	for _, p := range paraSplitRE.Split(s, -1) {
		if p = strings.Join(strings.Fields(p), " "); p != "" {
			paras = append(paras, p)
		}
	}
	return strings.Join(paras, "\n\n")
}

var paraSplitRE = regexp.MustCompile(`\n\s*\n`)

// HarvestGoBlog pulls pre-cutoff posts from the official Go blog: technical
// prose by many authors, from a source the corpus otherwise lacks.
func HarvestGoBlog(ctx context.Context, before time.Time, n int, seen func(id string) bool) ([]Sample, error) {
	raw, err := gh(ctx, "api", "repos/golang/website/contents/_content/blog", "--jq", ".[].name")
	if err != nil {
		return nil, err
	}
	var out []Sample
	for _, name := range strings.Fields(string(raw)) {
		if len(out) >= n || ctx.Err() != nil {
			break
		}
		id := "goblog-" + sanitize(strings.TrimSuffix(name, ".md"))
		if !strings.HasSuffix(name, ".md") || seen(id) {
			continue
		}
		b64, err := gh(ctx, "api", "repos/golang/website/contents/_content/blog/"+name, "--jq", ".content")
		if err != nil {
			continue
		}
		body, err := base64.StdEncoding.DecodeString(strings.ReplaceAll(strings.TrimSpace(string(b64)), "\\n", ""))
		if err != nil {
			continue
		}
		doc := string(body)
		fm := frontMatterRE.FindStringSubmatch(doc)
		if fm == nil {
			continue
		}
		d := goDateRE.FindStringSubmatch(fm[1])
		if d == nil {
			continue
		}
		if ts, err := time.Parse("2006-01-02", d[1]); err != nil || !ts.Before(before) {
			continue
		}
		title := ""
		if t := goTitleRE.FindStringSubmatch(fm[1]); t != nil {
			title = t[1]
		}
		prose := doc[len(fm[0]):]
		prose = goDirectiveRE.ReplaceAllString(prose, "")
		prose = indentCodeRE.ReplaceAllString(prose, "")
		prose = mdLinkRE.ReplaceAllString(prose, "$1")
		prose = htmlTagRE.ReplaceAllString(prose, "")
		prose = reflow(stripMarkdown(prose))
		words := len(strings.Fields(prose))
		if words < 250 || words > 2500 {
			continue
		}
		out = append(out, Sample{ID: id, Kind: "blog", Source: "goblog", Text: prose,
			Context: map[string]string{"title": title, "tags": "Go, programming", "words": fmt.Sprint(words)}})
	}
	return out, nil
}

// hfRows pages through a Hugging Face dataset via the datasets-server API.
func hfRows(ctx context.Context, dataset, config string, offset, length int) ([]map[string]any, error) {
	u := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=train&offset=%d&length=%d", dataset, config, offset, length)
	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var page struct {
		Rows []struct {
			Row map[string]any `json:"row"`
		} `json:"rows"`
		Error string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&page); err != nil {
		return nil, err
	}
	if page.Error != "" {
		return nil, fmt.Errorf("%s: %s", dataset, page.Error)
	}
	rows := make([]map[string]any, len(page.Rows))
	for i, r := range page.Rows {
		rows[i] = r.Row
	}
	return rows, nil
}

// HarvestLearnerEssays pulls essays by English-language learners: the
// Feedback Prize ELL corpus (US grades 8–12) and ICNALE (Asian college
// students). These guard against the well-known failure of detectors that
// flag non-native writing as AI.
func HarvestLearnerEssays(ctx context.Context, n int, seen func(id string) bool) ([]Sample, error) {
	sources := []struct{ name, dataset, field, idField string }{
		{"ell", "tcapelle/feedback-prize-english-language-learning-fluency", "full_text", "text_id"},
		{"icnale", "srky/ICNALE_writing_score", "OriginalText", "__index_level_0__"},
	}
	var out []Sample
	perSource := []int{n * 3 / 4, n - n*3/4}
	for si, src := range sources {
		got := 0
		for offset := 0; got < perSource[si]; offset += 100 {
			rows, err := hfRows(ctx, src.dataset, "default", offset, 100)
			if err != nil {
				return out, err
			}
			if len(rows) == 0 {
				break
			}
			for _, r := range rows {
				body, _ := r[src.field].(string)
				body = strings.TrimSpace(strings.ReplaceAll(body, "\r\n", "\n"))
				id := fmt.Sprintf("%s-%v", src.name, r[src.idField])
				words := len(strings.Fields(body))
				if words < 150 || words > 1200 || seen(id) || got >= perSource[si] {
					continue
				}
				out = append(out, Sample{ID: id, Kind: "essay", Source: src.name, Text: body,
					Context: map[string]string{"words": fmt.Sprint(words)}})
				got++
			}
		}
	}
	return out, nil
}
