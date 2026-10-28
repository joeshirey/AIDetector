package corpus

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
	"unicode"
)

// Technical blogs by individual engineers, for human writing that's
// technical in the way READMEs, design docs, and PR descriptions are.

var (
	dropBlockRE = regexp.MustCompile(`(?is)<(script|style|pre|code|blockquote|figure|table|noscript|iframe)\b[^>]*>.*?</(script|style|pre|code|blockquote|figure|table|noscript|iframe)>`)
	blockTagRE  = regexp.MustCompile(`(?i)</?(p|div|br|li|ul|ol|h[1-6]|section|article|hr|tr)\b[^>]*>`)
	anyTagRE    = regexp.MustCompile(`<[^>]+>`)
	spaceRunRE  = regexp.MustCompile(`[ \t\x{00a0}]+`)
	blogNoiseRE = regexp.MustCompile(`(?i)^(sponsor:|sponsored by|about scott|scott hanselman is a|share this:|like this:|related|loading\.\.\.|advertisement)`)
)

// htmlToText turns article HTML into plain paragraphs. Code, quotes, and
// tables are dropped: they aren't the author's prose.
func htmlToText(s string) string {
	s = dropBlockRE.ReplaceAllString(s, " ")
	s = blockTagRE.ReplaceAllString(s, "\n\n")
	s = anyTagRE.ReplaceAllString(s, "")
	s = html.UnescapeString(s)
	var paras []string
	for _, p := range paraSplitRE.Split(s, -1) {
		p = strings.TrimSpace(spaceRunRE.ReplaceAllString(strings.ReplaceAll(p, "\n", " "), " "))
		if p == "" || blogNoiseRE.MatchString(p) {
			continue
		}
		paras = append(paras, p)
	}
	return strings.Join(paras, "\n\n")
}

func getBody(ctx context.Context, url string) (string, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	req.Header.Set("User-Agent", "aidetector-corpus/1.0 (personal research; github.com/joeshirey/AIDetector)")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("%s: HTTP %d", url, resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return string(b), err
}

func blogSample(id, source, title, text string) (Sample, bool) {
	words := len(strings.Fields(text))
	if words < 250 || words > 2500 {
		return Sample{}, false
	}
	return Sample{ID: id, Kind: "blog", Source: source, Text: text,
		Context: map[string]string{"title": title, "tags": "software, technology", "words": fmt.Sprint(words)}}, true
}

// HarvestSeroter pulls posts from Richard Seroter's blog (WordPress.com)
// published before `before`.
func HarvestSeroter(ctx context.Context, before time.Time, n int, seen func(string) bool) ([]Sample, error) {
	var out []Sample
	for page := 1; len(out) < n && page <= 20; page++ {
		url := fmt.Sprintf("https://public-api.wordpress.com/rest/v1.1/sites/seroter.com/posts/?before=%s&number=100&page=%d&fields=ID,date,title,content",
			before.Format("2006-01-02T15:04:05"), page)
		body, err := getBody(ctx, url)
		if err != nil {
			return out, err
		}
		var res struct {
			Posts []struct {
				ID      int    `json:"ID"`
				Title   string `json:"title"`
				Content string `json:"content"`
			} `json:"posts"`
		}
		if err := json.Unmarshal([]byte(body), &res); err != nil {
			return out, err
		}
		if len(res.Posts) == 0 {
			break
		}
		for _, p := range res.Posts {
			id := fmt.Sprintf("seroter-%d", p.ID)
			if len(out) >= n || seen(id) {
				continue
			}
			if sm, ok := blogSample(id, "seroter", html.UnescapeString(p.Title), htmlToText(p.Content)); ok {
				out = append(out, sm)
			}
		}
	}
	return out, nil
}

var (
	hanselPostRE  = regexp.MustCompile(`href="https://www\.hanselman\.com/blog/([a-z0-9-]{8,})"`)
	hanselBodyRE  = regexp.MustCompile(`(?s)class="containerInner blogBody">(.*?)<div class="about-scott">`)
	hanselTitleRE = regexp.MustCompile(`(?s)<title>(.*?)</title>`)
)

// HarvestHanselman samples posts from Scott Hanselman's blog by year,
// spreading n across the given years (all before the cutoff).
func HarvestHanselman(ctx context.Context, years []int, n int, seen func(string) bool) ([]Sample, error) {
	var out []Sample
	per := (n + len(years) - 1) / len(years)
	polite := time.NewTicker(time.Second)
	defer polite.Stop()
	for _, y := range years {
		page, err := getBody(ctx, fmt.Sprintf("https://www.hanselman.com/blog/archive/%d", y))
		if err != nil {
			return out, err
		}
		got := 0
		slugs := map[string]bool{}
		for _, m := range hanselPostRE.FindAllStringSubmatch(page, -1) {
			slugs[m[1]] = true
		}
		for slug := range slugs { // map order gives a cheap random sample
			id := "hanselman-" + slug
			if got >= per || len(out) >= n || seen(id) {
				continue
			}
			select {
			case <-polite.C:
			case <-ctx.Done():
				return out, ctx.Err()
			}
			post, err := getBody(ctx, "https://www.hanselman.com/blog/"+slug)
			if err != nil {
				continue
			}
			m := hanselBodyRE.FindStringSubmatch(post)
			if m == nil {
				continue
			}
			body := m[1]
			title := ""
			if t := hanselTitleRE.FindStringSubmatch(post); t != nil {
				title = strings.TrimSpace(strings.Split(html.UnescapeString(t[1]), " - ")[0])
			}
			if sm, ok := blogSample(id, "hanselman", title, htmlToText(body)); ok {
				out = append(out, sm)
				got++
			}
		}
	}
	return out, nil
}

// NonTechBefore is the date cutoff for the non-tech Medium pool, well before
// ChatGPT and before the Medium dataset's later, more mixed years.
var NonTechBefore = time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC)

// HarvestMediumNonTech collects Medium posts dated before 2021 whose tags
// match neither workTags nor techTags: general-interest human writing for a
// held-out false-positive pool (not training). skip excludes article IDs,
// such as regression fixtures.
func HarvestMediumNonTech(ctx context.Context, n, phase int, skip map[string]bool, seen func(id string) bool) ([]Sample, error) {
	keep := func(tags []string) bool { return !anyWorkTag(tags) && !anyTechTag(tags) && !anyTechTagLoose(tags) }
	return harvestMediumFiltered(ctx, NonTechBefore, n, phase, "medium-nontech", keep, 300, skip, seen)
}

// anyTechTagLoose catches engineering tags that the stricter techTags set
// leaves out on purpose (broad ones like "technology").
func anyTechTagLoose(tags []string) bool {
	for _, t := range tags {
		if techTags[strings.ToLower(t)] {
			return true
		}
	}
	return false
}

var stopwordRE = regexp.MustCompile(`(?i)\b(the|and|of|to|a|in|is|that|it|for|you|was|with|on|as|have|but|be|are|this|not|i)\b`)

// isEnglish is a cheap filter: mostly ASCII letters and a normal share of
// common English function words.
func isEnglish(s string) bool {
	letters, ascii := 0, 0
	for _, r := range s {
		if unicode.IsLetter(r) {
			letters++
			if r < 128 {
				ascii++
			}
		}
	}
	words := len(strings.Fields(s))
	return letters > 0 && float64(ascii)/float64(letters) > 0.97 &&
		words > 0 && float64(len(stopwordRE.FindAllStringIndex(s, -1)))/float64(words) > 0.25
}

// techTags narrows Medium to engineering topics (the original harvest also
// took business, productivity, and careers posts).
var techTags = map[string]bool{
	"programming": true, "software development": true, "software engineering": true, "javascript": true,
	"python": true, "web development": true, "devops": true, "machine learning": true, "data science": true,
	"cloud computing": true, "coding": true, "aws": true, "kubernetes": true, "docker": true, "react": true,
	"java": true, "golang": true, "api": true, "microservices": true, "database": true, "sql": true,
	"cybersecurity": true, "open source": true, "github": true, "linux": true, "frontend": true,
	"backend": true, "nodejs": true, "typescript": true, "css": true, "android": true, "ios": true,
	"swift": true, "rust": true, "google cloud platform": true, "azure": true, "serverless": true,
}

// broadTags mark posts that are about science, business, or culture even if
// they also carry an engineering tag.
var broadTags = map[string]bool{
	"neuroscience": true, "psychology": true, "health": true, "politics": true, "business": true,
	"marketing": true, "self improvement": true, "life": true, "entrepreneurship": true, "startup": true,
}

func anyTechTag(tags []string) bool {
	tech := false
	for _, t := range tags {
		t = strings.ToLower(t)
		if broadTags[t] {
			return false
		}
		tech = tech || techTags[t]
	}
	return tech
}
