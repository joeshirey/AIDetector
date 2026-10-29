package overnight

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"github.com/joeshirey/AIDetector/internal/corpus"
)

// ReadLedger returns every ledger entry.
func ReadLedger(path string) ([]corpus.Entry, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var out []corpus.Entry
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var e corpus.Entry
		if json.Unmarshal(sc.Bytes(), &e) == nil {
			out = append(out, e)
		}
	}
	return out, sc.Err()
}

// NightSpend sums a night's ledger by model, and its token usage.
type NightSpend struct {
	Total   float64
	ByModel map[string]float64
	Usage   map[string]corpus.Usage // by model@mode
	Calls   map[string]int          // by model@mode
}

func SpendFor(entries []corpus.Entry, night string) NightSpend {
	s := NightSpend{ByModel: map[string]float64{}, Usage: map[string]corpus.Usage{}, Calls: map[string]int{}}
	for _, e := range entries {
		if e.Night != night {
			continue
		}
		s.Total += e.Cost
		s.ByModel[e.Model] += e.Cost
		if e.Usage == (corpus.Usage{}) {
			continue // reservations and corrections carry no tokens
		}
		k := e.Model + "@" + e.Mode
		u := s.Usage[k]
		u.Input += e.Usage.Input
		u.Output += e.Usage.Output
		u.Thinking += e.Usage.Thinking
		u.CacheRead += e.Usage.CacheRead
		u.Seconds += e.Usage.Seconds
		s.Usage[k] = u
		s.Calls[k] += max(e.N, 1)
	}
	return s
}

// Reconciliation compares billed Vertex AI cost with the ledger for nights
// old enough that the billing export is complete.
type Reconciliation struct {
	Enabled bool
	Note    string
	Nights  []NightBill
	// Over is true when any night billed more than 15% above the ledger;
	// the loop then lowers its cap by the worst ratio.
	Over     bool
	Worst    float64
	NewLimit float64
}

type NightBill struct {
	Night          string
	Ledger, Billed float64
}

// BillingTable is the BigQuery billing export table
// (project.dataset.gcp_billing_export_v1_XXXX), from AIDETECTOR_BILLING_TABLE.
func BillingTable() string { return os.Getenv("AIDETECTOR_BILLING_TABLE") }

// Reconcile checks the nights 2 to 7 days before tonight (the export lags
// by up to a day). Batch costs land in the night the job ran, while the
// ledger counts them in the night they were submitted, so small
// night-to-night differences are expected; the check uses the sum.
func Reconcile(ctx context.Context, entries []corpus.Entry, tonight time.Time, limit float64) Reconciliation {
	r := Reconciliation{NewLimit: limit}
	table := BillingTable()
	if table == "" {
		r.Note = "billing export not configured (set AIDETECTOR_BILLING_TABLE after enabling the Cloud Billing export to BigQuery)"
		return r
	}
	r.Enabled = true
	from := tonight.AddDate(0, 0, -7)
	to := tonight.AddDate(0, 0, -1) // exclusive: nights before yesterday
	q := fmt.Sprintf("SELECT FORMAT_DATE('%%F', DATE(TIMESTAMP_SUB(usage_start_time, INTERVAL 12 HOUR), '%s')) AS night, "+
		"SUM(cost) + SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS billed "+
		"FROM `%s` WHERE service.description = 'Vertex AI' AND project.id = '%s' "+
		"AND usage_start_time >= TIMESTAMP('%s') AND usage_start_time < TIMESTAMP('%s') GROUP BY night",
		tonight.Location().String(), table, corpus.VertexProject(),
		from.Add(12*time.Hour).UTC().Format(time.RFC3339), to.Add(12*time.Hour).UTC().Format(time.RFC3339))
	rows, err := bqQuery(ctx, q)
	if err != nil {
		r.Note = "billing query failed: " + err.Error()
		return r
	}
	var ledgerSum, billedSum float64
	for d := from; d.Before(to); d = d.AddDate(0, 0, 1) {
		n := d.Format("2006-01-02")
		nb := NightBill{Night: n, Ledger: SpendFor(entries, n).Total, Billed: rows[n]}
		if nb.Ledger == 0 && nb.Billed == 0 {
			continue
		}
		r.Nights = append(r.Nights, nb)
		ledgerSum += nb.Ledger
		billedSum += nb.Billed
	}
	sort.Slice(r.Nights, func(i, j int) bool { return r.Nights[i].Night < r.Nights[j].Night })
	if ledgerSum > 1 && billedSum > ledgerSum*1.15 {
		r.Over, r.Worst = true, billedSum/ledgerSum
		r.NewLimit = limit / r.Worst
		r.Note = fmt.Sprintf("billed $%.2f vs ledger $%.2f (%.0f%% over); cap lowered to $%.2f", billedSum, ledgerSum, (r.Worst-1)*100, r.NewLimit)
	} else {
		r.Note = fmt.Sprintf("billed $%.2f vs ledger $%.2f over %d nights", billedSum, ledgerSum, len(r.Nights))
	}
	return r
}

// bqQuery runs a query through the BigQuery REST API with
// application-default credentials and returns night -> billed.
func bqQuery(ctx context.Context, q string) (map[string]float64, error) {
	ts, err := google.DefaultTokenSource(ctx, "https://www.googleapis.com/auth/bigquery.readonly", "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, err
	}
	project := corpus.VertexProject()
	body, _ := json.Marshal(map[string]any{"query": q, "useLegacySql": false, "timeoutMs": 60000})
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://bigquery.googleapis.com/bigquery/v2/projects/"+project+"/queries", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-user-project", project)
	resp, err := oauth2.NewClient(ctx, ts).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, data[:min(len(data), 300)])
	}
	var res struct {
		JobComplete bool `json:"jobComplete"`
		Rows        []struct {
			F []struct {
				V any `json:"v"`
			} `json:"f"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(data, &res); err != nil {
		return nil, err
	}
	if !res.JobComplete {
		return nil, fmt.Errorf("query did not finish in time")
	}
	out := map[string]float64{}
	for _, row := range res.Rows {
		if len(row.F) < 2 {
			continue
		}
		night, _ := row.F[0].V.(string)
		v, _ := row.F[1].V.(string)
		out[night], _ = strconv.ParseFloat(v, 64)
	}
	return out, nil
}
