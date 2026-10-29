// Package fixtures downloads and caches the regression fixtures listed in
// internal/engine/testdata/fixtures.json. Article text belongs to its
// authors, so it is fetched on demand into a gitignored cache rather than
// committed.
package fixtures

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type Fixture struct {
	ID       string `json:"id"`
	Source   string `json:"source"` // seroter, medium-csv, or medium-rss
	URL      string `json:"url"`
	Reported int    `json:"reported,omitempty"` // score the user saw before this round of changes
}

type List struct {
	KnownHuman []Fixture `json:"known_human"`
	Monitoring []Fixture `json:"monitoring"`
}

func Load(path string) (*List, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var l List
	return &l, json.Unmarshal(b, &l)
}

// Paths returns the cached files for a fixture: the paste-style plain text
// (always) and, for HTML sources, a Markdown rendering.
func Paths(dir string, f Fixture) (txt, md string) {
	return filepath.Join(dir, f.ID+".txt"), filepath.Join(dir, f.ID+".md")
}

// Fetch downloads every fixture missing from dir.
func Fetch(ctx context.Context, dir string, fs []Fixture, log io.Writer) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var csvWanted []Fixture
	for _, f := range fs {
		txt, md := Paths(dir, f)
		if _, err := os.Stat(txt); err == nil {
			continue
		}
		var page string
		var err error
		switch f.Source {
		case "seroter":
			page, err = seroterHTML(ctx, f.URL)
		case "medium-rss":
			page, err = mediumRSSHTML(ctx, f.URL)
		case "medium-csv":
			csvWanted = append(csvWanted, f)
			continue
		default:
			err = fmt.Errorf("unknown source %q", f.Source)
		}
		if err != nil {
			fmt.Fprintf(log, "%s: %v\n", f.ID, err)
			continue
		}
		if err := os.WriteFile(txt, []byte(ToPaste(page)), 0o644); err != nil {
			return err
		}
		if err := os.WriteFile(md, []byte(ToMarkdown(page)), 0o644); err != nil {
			return err
		}
		fmt.Fprintf(log, "%s: cached\n", f.ID)
	}
	if len(csvWanted) > 0 {
		return fetchMediumCSV(ctx, dir, csvWanted, log)
	}
	return nil
}

func get(ctx context.Context, u string) (string, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (aidetector fixtures; github.com/joeshirey/AIDetector)")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("%s: HTTP %d", u, resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	return string(b), err
}

// seroterHTML fetches a post's body HTML through the WordPress.com API.
func seroterHTML(ctx context.Context, postURL string) (string, error) {
	slug := path(postURL)
	body, err := get(ctx, "https://public-api.wordpress.com/rest/v1.1/sites/seroter.com/posts/slug:"+url.PathEscape(slug)+"?fields=title,content")
	if err != nil {
		return "", err
	}
	var p struct{ Title, Content string }
	if err := json.Unmarshal([]byte(body), &p); err != nil {
		return "", err
	}
	return p.Content, nil
}

var (
	rssItemRE    = regexp.MustCompile(`(?s)<item>(.*?)</item>`)
	rssLinkRE    = regexp.MustCompile(`(?s)<link>(.*?)</link>`)
	rssContentRE = regexp.MustCompile(`(?s)<content:encoded><!\[CDATA\[(.*?)\]\]></content:encoded>`)
)

// mediumRSSHTML finds a post's full HTML in its author's RSS feed. Feeds only
// carry recent posts, so fetch soon after publication.
func mediumRSSHTML(ctx context.Context, postURL string) (string, error) {
	u, err := url.Parse(postURL)
	if err != nil {
		return "", err
	}
	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	if len(parts) < 2 || !strings.HasPrefix(parts[0], "@") {
		return "", fmt.Errorf("not a medium.com/@author/slug URL")
	}
	feed, err := get(ctx, "https://medium.com/feed/"+parts[0])
	if err != nil {
		return "", err
	}
	for _, item := range rssItemRE.FindAllStringSubmatch(feed, -1) {
		link := rssLinkRE.FindStringSubmatch(item[1])
		content := rssContentRE.FindStringSubmatch(item[1])
		if link != nil && content != nil && strings.Contains(link[1], parts[1]) {
			return content[1], nil
		}
	}
	return "", fmt.Errorf("post not in the author's feed (feeds only keep recent posts)")
}

const mediumCSV = "https://huggingface.co/datasets/fabiochiu/medium-articles/resolve/main/medium_articles.csv"

// fetchMediumCSV streams the Medium dataset and saves the rows whose URL ends
// in a wanted article ID. Medium itself returns 403 to scripts.
func fetchMediumCSV(ctx context.Context, dir string, want []Fixture, log io.Writer) error {
	byID := map[string]Fixture{}
	for _, f := range want {
		byID[articleID(f.URL)] = f
	}
	req, _ := http.NewRequestWithContext(ctx, "GET", mediumCSV, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	r := csv.NewReader(bufio.NewReaderSize(resp.Body, 1<<20))
	r.LazyQuotes = true
	r.Read() // header: title,text,url,authors,timestamp,tags
	for len(byID) > 0 {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil || len(rec) < 3 {
			continue
		}
		f, ok := byID[articleID(rec[2])]
		if !ok {
			continue
		}
		txt, _ := Paths(dir, f)
		if err := os.WriteFile(txt, []byte(strings.TrimSpace(rec[1])+"\n"), 0o644); err != nil {
			return err
		}
		fmt.Fprintf(log, "%s: cached from dataset\n", f.ID)
		delete(byID, articleID(rec[2]))
	}
	for _, f := range byID {
		fmt.Fprintf(log, "%s: not found in dataset\n", f.ID)
	}
	return nil
}

// articleID is the trailing hex ID Medium appends to every article slug.
func articleID(u string) string {
	p := path(u)
	if i := strings.LastIndex(p, "-"); i >= 0 {
		return p[i+1:]
	}
	return p
}

func path(u string) string {
	pu, err := url.Parse(strings.TrimSpace(u))
	if err != nil {
		return u
	}
	parts := strings.Split(strings.Trim(pu.Path, "/"), "/")
	return parts[len(parts)-1]
}

var (
	preRE       = regexp.MustCompile(`(?is)<pre\b[^>]*>(.*?)</pre>`)
	tableRE     = regexp.MustCompile(`(?is)<table\b[^>]*>(.*?)</table>`)
	rowRE       = regexp.MustCompile(`(?is)<tr\b[^>]*>(.*?)</tr>`)
	cellRE      = regexp.MustCompile(`(?is)<t[hd]\b[^>]*>(.*?)</t[hd]>`)
	quoteRE     = regexp.MustCompile(`(?is)<blockquote\b[^>]*>(.*?)</blockquote>`)
	headRE      = regexp.MustCompile(`(?is)<h([1-6])\b[^>]*>(.*?)</h[1-6]>`)
	liRE        = regexp.MustCompile(`(?is)<li\b[^>]*>`)
	scriptRE    = regexp.MustCompile(`(?is)<(script|style|noscript|iframe)\b.*?</(script|style|noscript|iframe)>`)
	blockRE     = regexp.MustCompile(`(?i)</?(p|div|br|ul|ol|section|article|figure|figcaption|hr)\b[^>]*>`)
	tagRE       = regexp.MustCompile(`<[^>]+>`)
	blankRunsRE = regexp.MustCompile(`\n{3,}`)
	hspaceRE    = regexp.MustCompile(`[ \t\x{00a0}]+`)
)

// ToMarkdown renders article HTML roughly as Markdown: fenced code, piped
// tables, quoted blockquotes, headings, and bullet lists.
func ToMarkdown(h string) string { return render(h, true) }

// ToPaste renders article HTML the way it looks when copied out of a
// browser: code as plain lines, table rows as cells run together, quotes as
// ordinary paragraphs.
func ToPaste(h string) string { return render(h, false) }

func render(h string, markdown bool) string {
	h = scriptRE.ReplaceAllString(h, "")
	h = preRE.ReplaceAllStringFunc(h, func(m string) string {
		code := html.UnescapeString(tagRE.ReplaceAllString(preRE.FindStringSubmatch(m)[1], ""))
		if markdown {
			return "\n\n```\n" + strings.Trim(code, "\n") + "\n```\n\n"
		}
		return "\n\n" + strings.Trim(code, "\n") + "\n\n"
	})
	h = tableRE.ReplaceAllStringFunc(h, func(m string) string {
		var rows []string
		for i, r := range rowRE.FindAllStringSubmatch(m, -1) {
			var cells []string
			for _, c := range cellRE.FindAllStringSubmatch(r[1], -1) {
				cells = append(cells, strings.TrimSpace(hspaceRE.ReplaceAllString(html.UnescapeString(tagRE.ReplaceAllString(c[1], " ")), " ")))
			}
			if markdown {
				rows = append(rows, "| "+strings.Join(cells, " | ")+" |")
				if i == 0 {
					rows = append(rows, "|"+strings.Repeat("---|", len(cells)))
				}
			} else {
				rows = append(rows, strings.Join(cells, " "))
			}
		}
		sep := "\n"
		if !markdown {
			sep = " " // browsers often paste a table as one run-on line
		}
		return "\n\n" + strings.Join(rows, sep) + "\n\n"
	})
	if markdown {
		h = quoteRE.ReplaceAllStringFunc(h, func(m string) string {
			inner := strings.TrimSpace(tagRE.ReplaceAllString(blockRE.ReplaceAllString(quoteRE.FindStringSubmatch(m)[1], "\n"), ""))
			return "\n\n> " + strings.ReplaceAll(inner, "\n", "\n> ") + "\n\n"
		})
		h = headRE.ReplaceAllString(h, "\n\n## $2\n\n")
		h = liRE.ReplaceAllString(h, "\n- ")
	} else {
		h = headRE.ReplaceAllString(h, "\n\n$2\n\n")
		h = liRE.ReplaceAllString(h, "\n")
	}
	h = blockRE.ReplaceAllString(h, "\n\n")
	h = html.UnescapeString(tagRE.ReplaceAllString(h, ""))
	var lines []string
	for _, l := range strings.Split(h, "\n") {
		// Keep leading indentation (code); collapse runs of spaces after it.
		body := strings.TrimLeft(l, " \t")
		lead := l[:len(l)-len(body)]
		lines = append(lines, lead+strings.TrimRight(hspaceRE.ReplaceAllString(body, " "), " "))
	}
	return strings.TrimSpace(blankRunsRE.ReplaceAllString(strings.Join(lines, "\n"), "\n\n")) + "\n"
}

// Scorer scores a text as prose and returns the document score and the
// highest segment score (-1 if none).
type Scorer func(text string) (doc, maxSeg int, label string, err error)

// CheckResult is the regression check over known-human fixtures.
type CheckResult struct {
	KnownOver60, SegOver80, Missing int
	Rows                            []CheckRow
}

type CheckRow struct {
	ID, Label   string
	Doc, MaxSeg int
	Monitoring  bool
	Reported    int
}

// Pass is the rule: at most one known-human fixture at 60+, and no
// known-human segment at 80+.
func (r CheckResult) Pass() bool { return r.KnownOver60 <= 1 && r.SegOver80 == 0 }

// Check scores every cached fixture (paste-style text) with score.
func Check(l *List, dir string, score Scorer) (CheckResult, error) {
	var res CheckResult
	run := func(fs []Fixture, monitoring bool) error {
		for _, f := range fs {
			txt, _ := Paths(dir, f)
			b, err := os.ReadFile(txt)
			if err != nil {
				res.Missing++
				continue
			}
			doc, seg, label, err := score(string(b))
			if err != nil {
				return err
			}
			res.Rows = append(res.Rows, CheckRow{ID: f.ID, Label: label, Doc: doc, MaxSeg: seg, Monitoring: monitoring, Reported: f.Reported})
			if !monitoring {
				if doc >= 60 {
					res.KnownOver60++
				}
				if seg >= 80 {
					res.SegOver80++
				}
			}
		}
		return nil
	}
	if err := run(l.KnownHuman, false); err != nil {
		return res, err
	}
	return res, run(l.Monitoring, true)
}
