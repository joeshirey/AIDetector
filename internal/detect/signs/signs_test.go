package signs

import (
	"context"
	"strings"
	"testing"

	"github.com/joeshirey/AIDetector/internal/detect"
)

func signals(t *testing.T, text string) map[string]detect.Signal {
	t.Helper()
	got, err := New().Detect(context.Background(), detect.Input{Text: text, Kind: detect.KindProse})
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]detect.Signal{}
	for _, s := range got {
		m[s.Name] = s
	}
	return m
}

// Before/after pairs from the humanizer skill: each "before" should trip the
// named signal more than its "after".
func TestHumanizerExamples(t *testing.T) {
	cases := []struct{ signal, before, after string }{
		{"negative_parallelism",
			"It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere. It's not merely a song, it's a statement about the times we live in.",
			"The heavy beat adds to the aggressive tone, and the song says something about the times we live in, at least to me."},
		{"deep_sayings",
			"The real question is whether teams can adapt. At its core, what really matters is organizational readiness for the work ahead of them.",
			"The question is whether teams can adapt. That mostly depends on whether the organization is ready to change its habits."},
		{"inflated_significance",
			"The Statistical Institute of Catalonia was officially established in 1989, marking a pivotal moment in the evolution of regional statistics in Spain, setting the stage for later work.",
			"The Statistical Institute of Catalonia was established in 1989, part of a wider decentralization of administrative functions in Spain."},
		{"copula_avoidance",
			"Gallery 825 serves as the exhibition space for contemporary art. The gallery features four separate spaces and boasts over 3,000 square feet of room.",
			"Gallery 825 is the exhibition space for contemporary art. The gallery has four rooms totaling 3,000 square feet of room for shows."},
		{"ing_riders",
			"The temple is painted blue, green, and gold, symbolizing Texas bluebonnets and the Gulf of Mexico, reflecting the community's deep connection to the land around it.",
			"The temple is painted blue, green, and gold, colors meant to evoke Texas bluebonnets and the Gulf of Mexico and the land around it."},
	}
	for _, c := range cases {
		b, a := signals(t, c.before)[c.signal], signals(t, c.after)[c.signal]
		if b.Value <= a.Value {
			t.Errorf("%s: before %.2f should exceed after %.2f", c.signal, b.Value, a.Value)
		}
	}
}

func TestChatbotResidueIsDecisive(t *testing.T) {
	s := signals(t, "The French Revolution began in 1789 when a financial crisis and food shortages led to unrest. I hope this helps! Let me know if you'd like me to expand on any section.")
	if r, ok := s["chatbot_residue"]; !ok || !r.Decisive {
		t.Fatalf("chatbot residue not flagged as decisive: %+v", r)
	}
}

func TestPunctuationVariety(t *testing.T) {
	plain := signals(t, strings.Repeat("We shipped it. It works. The team is happy, and the users are happy. We will add more soon, and we will test it. The plan is set, and it is good. Next week we start again. ", 2))
	varied := signals(t, strings.Repeat("We shipped it (finally). Does it work? Mostly; the cache is flaky. The team is relieved: users stopped filing bugs. Next week — assuming nothing breaks — we start on search. ", 2))
	if plain["punctuation_variety"].Value >= varied["punctuation_variety"].Value {
		t.Errorf("plain %.2f should be below varied %.2f", plain["punctuation_variety"].Value, varied["punctuation_variety"].Value)
	}
}
