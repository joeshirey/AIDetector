package corpus

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"html"
	"net/http"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/joeshirey/AIDetector/internal/split"
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

// Short, casual human prose from before ChatGPT, to balance the short
// human training text (mostly Enron emails): Reddit self-posts from the
// Webis TL;DR corpus (2006-2016) and personal blog posts from the Blog
// Authorship Corpus (2004). Only train-split ids are kept so the frozen
// test split doesn't change.

var (
	urlLinkRE   = regexp.MustCompile(`\burlLink\b`)
	headerRE    = regexp.MustCompile(`(?m)^\s*(From|Subject|To|Newsgroups|Date):`)
	shortStopRE = regexp.MustCompile(`(?i)\b(the|and|to|of|a|i|in|it|is|that|you|was|for|my|on|but|with|this|have|be|not|so|me|at|as|are|just|like|what|they|we)\b`)
)

// shortProseOK filters a candidate post: 80-350 words, English, at least
// three sentences, no mail or news headers, not shouting.
func shortProseOK(t string) bool {
	w := strings.Fields(t)
	if len(w) < 80 || len(w) > 350 || headerRE.MatchString(t) {
		return false
	}
	if float64(len(shortStopRE.FindAllString(t, -1)))/float64(len(w)) < 0.25 {
		return false // probably not English
	}
	if strings.Count(t, ".")+strings.Count(t, "!")+strings.Count(t, "?") < 3 {
		return false
	}
	upper, letters := 0, 0
	for _, r := range t {
		if unicode.IsLetter(r) {
			letters++
			if unicode.IsUpper(r) {
				upper++
			}
		}
	}
	return letters > 0 && float64(upper)/float64(letters) < 0.2
}

// HarvestShortHuman takes n short posts from a datasets-server dataset,
// sampling pages spread across it.
func HarvestShortHuman(ctx context.Context, source string, n int, seen func(string) bool) ([]Sample, error) {
	var dataset, textField, idField string
	var total int
	switch source {
	case "reddit":
		dataset, textField, idField, total = "webis/tldr-17", "content", "id", 1300000
	case "blogauth":
		dataset, textField, idField, total = "tasksource/blog_authorship_corpus", "text", "id", 680000
	default:
		return nil, fmt.Errorf("unknown short-human source %q", source)
	}
	var out []Sample
	stride := total / 400 // up to 400 pages of 100 rows, spread across the dataset
	for page := 0; page < 400 && len(out) < n && ctx.Err() == nil; page++ {
		offset := page*stride + (page*7919)%max(stride-100, 1)
		u := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=train&offset=%d&length=100", dataset, offset)
		var rows struct {
			Rows []struct {
				RowIdx int            `json:"row_idx"`
				Row    map[string]any `json:"row"`
			} `json:"rows"`
		}
		var err error
		for attempt := range 5 {
			if attempt > 0 {
				time.Sleep(time.Duration(attempt*attempt) * 5 * time.Second)
			}
			var body string
			if body, err = getBody(ctx, u); err == nil {
				if err = json.Unmarshal([]byte(body), &rows); err == nil {
					break
				}
			}
		}
		if err != nil {
			continue // skip a page that won't load
		}
		taken := 0
		for _, r := range rows.Rows {
			if taken >= 4 || len(out) >= n { // a few per page, for variety
				break
			}
			t, _ := r.Row[textField].(string)
			t = strings.ReplaceAll(html.UnescapeString(urlLinkRE.ReplaceAllString(t, "")), "\u00a0", " ")
			t = strings.TrimSpace(spaceRunRE.ReplaceAllString(t, " "))
			if !shortProseOK(t) {
				continue
			}
			id := fmt.Sprintf("%s-%v", source, r.Row[idField])
			if source == "blogauth" {
				id = fmt.Sprintf("%s-%d", source, r.RowIdx)
			}
			if split.Of(id) != "train" || seen(id) {
				continue
			}
			out = append(out, Sample{ID: id, Kind: "blog", Source: dataset, Text: t,
				Context: map[string]string{"title": "", "tags": "personal", "words": fmt.Sprint(len(strings.Fields(t)))}})
			taken++
		}
	}
	return out, nil
}
