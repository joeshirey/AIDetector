package stylometric_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/stylometric"
	"github.com/joeshirey/AIDetector/internal/text"
)

func runStylometric(t *testing.T, body string, kind detect.Kind) map[string]detect.Signal {
	t.Helper()
	d := stylometric.New()
	if got := d.Name(); got != "stylometric" {
		t.Fatalf("Name() = %q, want \"stylometric\"", got)
	}
	sigs, err := d.Detect(context.Background(), detect.Input{Text: body, Kind: kind})
	if err != nil {
		t.Fatalf("Detect() error: %v", err)
	}
	out := make(map[string]detect.Signal, len(sigs))
	for _, s := range sigs {
		if s.Detector != "stylometric" {
			t.Errorf("signal %q Detector = %q, want \"stylometric\"", s.Name, s.Detector)
		}
		out[s.Name] = s
	}
	return out
}

func TestDetectTooShortAndStripsCodeBlocks(t *testing.T) {
	if got := runStylometric(t, "Too few words here to score reliably.", detect.KindProse); len(got) != 0 {
		t.Fatalf("expected nil signals under 15 words, got %+v", got)
	}

	// Code block has >30 tokens, but after StripCode only 6 prose words remain (< 15).
	withCode := "Short note before code block:\n```go\nfunc main() {\n\tfmt.Println(\"one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\")\n}\n```\nDone."
	if got := runStylometric(t, withCode, detect.KindProse); len(got) != 0 {
		t.Fatalf("expected fenced code to be stripped before word-count check, got %+v", got)
	}
}

func TestDetectSignalsAndKindGating(t *testing.T) {
	// Rich multi-paragraph fixture hitting every stylometric feature:
	// - AI stems: delve, tapestry, robust, seamless
	// - AI phrases: "It's worth noting", "At its core"
	// - Em dashes: 2 ("—" and " -- ")
	// - Bullets: 3 total, 2 with **Bold** lead -> ratio 2/3
	// - Triplets ("fast, reliable, and secure"; "clear, concise, and testable")
	// - 4 paragraphs >= 8 words -> paragraph_uniformity emitted for KindProse only, not KindPR
	doc := strings.Join([]string{
		"It's worth noting that teams delve into a rich tapestry of distributed storage engines — building fast, reliable, and secure pipelines. At its core, the architecture keeps state simple, predictable, and observable.",
		"Engineers value clear, concise, and testable interfaces -- especially when deployments span multiple regions, clouds, and clusters. Careful capacity planning avoids sudden latency spikes, noisy alerts, and cascading retries.",
		"- **Throughput:** sustains steady write amplification under heavy load\n- **Recovery:** replays compacted segments within seconds\n- plain bullet without bold prefix for ratio check",
		"Robust automation enables seamless rollouts across staging environments, production clusters, and edge nodes. Every release records deterministic metrics, structured traces, and audit logs.",
	}, "\n\n")

	totalWords := len(text.Words(text.StripCode(doc)))
	if totalWords < 80 {
		t.Fatalf("expected >=80 words, got %d", totalWords)
	}
	per1k := 1000.0 / float64(totalWords)

	proseSigs := runStylometric(t, doc, detect.KindProse)
	for _, name := range []string{
		"ai_vocabulary",
		"ai_phrases",
		"sentence_burstiness",
		"prose_burstiness",
		"clause_uniformity",
		"em_dashes",
		"bold_lead_bullets",
		"rule_of_three",
		"paragraph_uniformity",
	} {
		if _, ok := proseSigs[name]; !ok {
			t.Errorf("KindProse missing expected signal %q", name)
		}
	}

	// Experimental rhythm signals carry heuristic Weight == 0.
	if proseSigs["prose_burstiness"].Weight != 0 || proseSigs["clause_uniformity"].Weight != 0 {
		t.Errorf("expected Weight=0 on prose_burstiness (%g) and clause_uniformity (%g)",
			proseSigs["prose_burstiness"].Weight, proseSigs["clause_uniformity"].Weight)
	}

	// Exact em-dash rate: 1 Unicode em dash + 1 spaced " -- " = 2 per doc.
	wantEmRate := 2.0 * per1k
	if math.Abs(proseSigs["em_dashes"].Value-wantEmRate) > 1e-9 {
		t.Errorf("em_dashes.Value = %g, want %g", proseSigs["em_dashes"].Value, wantEmRate)
	}

	// Exact bold-lead bullet ratio: 2 bold-lead bullets out of 3 list items.
	if math.Abs(proseSigs["bold_lead_bullets"].Value-(2.0/3.0)) > 1e-9 {
		t.Errorf("bold_lead_bullets.Value = %g, want 2/3 (%g)", proseSigs["bold_lead_bullets"].Value, 2.0/3.0)
	}

	// Exact AI vocabulary stem hits: delve, tapestry, robust, seamless = 4 hits.
	wantVocabRate := 4.0 * per1k
	if math.Abs(proseSigs["ai_vocabulary"].Value-wantVocabRate) > 1e-9 {
		t.Errorf("ai_vocabulary.Value = %g, want %g (detail=%q)", proseSigs["ai_vocabulary"].Value, wantVocabRate, proseSigs["ai_vocabulary"].Detail)
	}

	// Exact AI phrase hits: "it's worth noting" + "at its core" = 2 hits.
	wantPhraseRate := 2.0 * per1k
	if math.Abs(proseSigs["ai_phrases"].Value-wantPhraseRate) > 1e-9 {
		t.Errorf("ai_phrases.Value = %g, want %g (detail=%q)", proseSigs["ai_phrases"].Value, wantPhraseRate, proseSigs["ai_phrases"].Detail)
	}

	if !(proseSigs["rule_of_three"].Value > 0.3) {
		t.Errorf("rule_of_three.Value = %g, want > 0.3", proseSigs["rule_of_three"].Value)
	}

	// KindPR must omit paragraph_uniformity on the exact same text.
	prSigs := runStylometric(t, doc, detect.KindPR)
	if _, ok := prSigs["paragraph_uniformity"]; ok {
		t.Errorf("KindPR unexpectedly emitted paragraph_uniformity")
	}
}

func TestBurstinessUniformVsVariedSentences(t *testing.T) {
	uniform := strings.Join([]string{
		"Ten steady words fill every single sentence in this block.",
		"Each line repeats the exact cadence without changing the pace.",
		"Uniform rhythm makes machine prose feel mechanical to most readers.",
		"Every clause lands on the tenth word with clockwork precision.",
		"Nothing short or sprawling interrupts the flat rhythm across lines.",
		"Six even sentences establish near zero variance across word counts.",
	}, " ")

	varied := strings.Join([]string{
		"Wait.",
		"Why did it break?",
		"Yesterday afternoon I opened the dusty cabinet behind the boiler room, traced three tangled copper pipes along the ceiling joist, and found a hairline crack leaking warm water onto the concrete floor.",
		"Easy fix.",
		"Once the valve cooled down, I wrapped fresh PTFE tape around the brass threads, tightened the union nut by hand, and gave it a quarter turn.",
		"Dry since Tuesday.",
	}, " ")

	uSigs := runStylometric(t, uniform, detect.KindProse)
	vSigs := runStylometric(t, varied, detect.KindProse)
	if !(uSigs["sentence_burstiness"].Value < vSigs["sentence_burstiness"].Value) {
		t.Errorf("uniform burstiness CV (%g) should be lower than varied (%g)",
			uSigs["sentence_burstiness"].Value, vSigs["sentence_burstiness"].Value)
	}
	if !(uSigs["sentence_burstiness"].AIProb > vSigs["sentence_burstiness"].AIProb) {
		t.Errorf("uniform burstiness AIProb (%g) should exceed varied (%g)",
			uSigs["sentence_burstiness"].AIProb, vSigs["sentence_burstiness"].AIProb)
	}
}
