package devtext_test

import (
	"context"
	"testing"

	"github.com/joeshirey/AIDetector/internal/detect"
	"github.com/joeshirey/AIDetector/internal/detect/devtext"
)

func detectSignals(t *testing.T, body string, kind detect.Kind) map[string]detect.Signal {
	t.Helper()
	d := devtext.New()
	if got := d.Name(); got != "devtext" {
		t.Fatalf("Detector.Name() = %q, want %q", got, "devtext")
	}
	list, err := d.Detect(context.Background(), detect.Input{Text: body, Kind: kind})
	if err != nil {
		t.Fatalf("Detect() unexpected error: %v", err)
	}
	out := make(map[string]detect.Signal, len(list))
	for _, s := range list {
		if s.Detector != "devtext" {
			t.Errorf("signal %q has Detector = %q, want %q", s.Name, s.Detector, "devtext")
		}
		if _, dup := out[s.Name]; dup {
			t.Errorf("duplicate signal %q emitted: %+v", s.Name, list)
		}
		out[s.Name] = s
	}
	return out
}

func TestAgentPRTemplateMonotonicOrdering(t *testing.T) {
	const fullAgentTemplate = `## Summary

Fix a nil pointer dereference when loading an empty configuration file.

## Test plan

- [x] Run unit tests for empty config files
- [ ] Smoke check CLI startup with default path
`
	const summaryTestingChecklist = `## Summary

Guard config loader against empty input files.

## Testing

- [x] go test ./internal/config/...
`
	const twoPlainHeadings = `## Motivation

Empty config files caused a startup panic in local dev environments.

## Changes

Return an initialized zero-value struct when the file length is zero.
`
	const threePlainHeadings = `## Overview

Handle empty configuration files cleanly.

## Motivation

New developer setups often touch an empty file before populating settings.

## Changes

Skip YAML decoding when the byte slice is empty.
`
	const fourPlainHeadings = `## Overview

Handle empty configuration files cleanly.

## Motivation

New developer setups often touch an empty file before populating settings.

## Key changes

Return defaults early in LoadConfig.

## What changed

Added an empty-slice check in loader.go.
`
	const fourHeadingsSummaryTestPlanNoChecks = `## Summary

Handle empty configuration files cleanly.

## Motivation

Avoid startup crash on empty file.

## Changes

Return defaults early in LoadConfig.

## Test plan

Verified manually by passing an empty file on the command line.
`
	const twoPlainHeadingsWithChecklist = `## Motivation

Avoid startup crash on empty file.

## Changes

- [x] Return defaults early in LoadConfig
- [x] Keep existing error path for malformed files
`

	full := detectSignals(t, fullAgentTemplate, detect.KindPR)
	fullTesting := detectSignals(t, summaryTestingChecklist, detect.KindPR)
	plain2 := detectSignals(t, twoPlainHeadings, detect.KindPR)
	plain3 := detectSignals(t, threePlainHeadings, detect.KindPR)
	plain4 := detectSignals(t, fourPlainHeadings, detect.KindPR)
	sumTestNoChk := detectSignals(t, fourHeadingsSummaryTestPlanNoChecks, detect.KindPR)
	plainWithChk := detectSignals(t, twoPlainHeadingsWithChecklist, detect.KindPR)

	gotFull, ok := full["agent_pr_template"]
	if !ok {
		t.Fatal("full Summary + Test plan + checklist PR missing agent_pr_template signal")
	}
	if gotFull.Value != 3 {
		t.Errorf("full Summary + Test plan + checklist agent_pr_template.Value = %v, want 3", gotFull.Value)
	}
	if gotFull.AIProb != 0.85 || gotFull.Weight != 1.2 {
		t.Errorf("full template AIProb/Weight = %v/%v, want 0.85/1.2", gotFull.AIProb, gotFull.Weight)
	}

	if got := fullTesting["agent_pr_template"]; got.Value != 3 {
		t.Errorf("Summary + Testing + checklist agent_pr_template.Value = %v, want 3", got.Value)
	}

	for _, tc := range []struct {
		label string
		sigs  map[string]detect.Signal
		want  float64
	}{
		{"2 plain headings (no checklist)", plain2, 1},
		{"3 plain headings (no checklist)", plain3, 1},
		{"4 plain headings (no checklist)", plain4, 1},
		{"4 headings including Summary + Test plan (no checklist)", sumTestNoChk, 2},
		{"2 plain headings with checklist", plainWithChk, 2},
	} {
		s, ok := tc.sigs["agent_pr_template"]
		if !ok {
			t.Errorf("%s: missing agent_pr_template signal", tc.label)
			continue
		}
		if s.Value != tc.want {
			t.Errorf("%s: agent_pr_template.Value = %v, want %v", tc.label, s.Value, tc.want)
		}
		if !(gotFull.Value > s.Value) {
			t.Errorf("monotonicity violated for %s: Summary+Test plan+checklist Value (%v) must exceed %v",
				tc.label, gotFull.Value, s.Value)
		}
		if !(gotFull.AIProb > s.AIProb) {
			t.Errorf("AIProb ordering violated for %s: full template AIProb (%v) must exceed %v",
				tc.label, gotFull.AIProb, s.AIProb)
		}
	}
}

func TestPRStructureSignalsSeparatedAndPresenceInvariant(t *testing.T) {
	t.Run("plain paragraph PR omits agent_pr_template and emits zero-weight structure counts", func(t *testing.T) {
		sigs := detectSignals(t, "Fixes a race condition in worker shutdown when the parent context cancels early.", detect.KindPR)
		if s, ok := sigs["agent_pr_template"]; ok {
			t.Errorf("plain 0-heading PR emitted agent_pr_template = %+v; want omitted for shipped pr.json Missing parity", s)
		}
		sec, ok := sigs["pr_section_count"]
		if !ok || sec.Value != 0 || sec.Weight != 0 {
			t.Errorf("pr_section_count = %+v (present=%v), want Value=0 Weight=0", sec, ok)
		}
		chk, ok := sigs["pr_checklist"]
		if !ok || chk.Value != 0 || chk.Weight != 0 {
			t.Errorf("pr_checklist = %+v (present=%v), want Value=0 Weight=0", chk, ok)
		}
	})

	t.Run("single heading with checklist omits agent_pr_template when test plan absent", func(t *testing.T) {
		const body = `## Summary

 Guard the config loader against empty files.

- [x] Unit test added
- [X] Linter clean
* [ ] Release notes updated
`
		sigs := detectSignals(t, body, detect.KindPR)
		if s, ok := sigs["agent_pr_template"]; ok {
			t.Errorf("1-heading PR without Test plan emitted agent_pr_template = %+v; want omitted", s)
		}
		sec, ok := sigs["pr_section_count"]
		if !ok || sec.Value != 1 {
			t.Errorf("pr_section_count = %+v (present=%v), want Value=1", sec, ok)
		}
		chk, ok := sigs["pr_checklist"]
		if !ok || chk.Value != 3 || chk.Weight != 0 {
			t.Errorf("pr_checklist = %+v (present=%v), want Value=3 Weight=0", chk, ok)
		}
	})

	t.Run("bold Summary and Test plan with checklist but zero markdown headings keeps agent_pr_template present", func(t *testing.T) {
		const body = `**Summary**
Fix nil pointer on empty config.

**Test plan**
- [x] go test ./...
`
		sigs := detectSignals(t, body, detect.KindPR)
		tmpl, ok := sigs["agent_pr_template"]
		if !ok || tmpl.Value != 3 {
			t.Errorf("bold Summary+Test plan+checklist agent_pr_template = %+v (present=%v), want Value=3", tmpl, ok)
		}
		sec, ok := sigs["pr_section_count"]
		if !ok || sec.Value != 0 || sec.Weight != 0 {
			t.Errorf("pr_section_count = %+v (present=%v), want Value=0 Weight=0", sec, ok)
		}
		chk, ok := sigs["pr_checklist"]
		if !ok || chk.Value != 1 || chk.Weight != 0 {
			t.Errorf("pr_checklist = %+v (present=%v), want Value=1 Weight=0", chk, ok)
		}
	})

	t.Run("four headings without checklist separates pr_section_count=4 from agent_pr_template", func(t *testing.T) {
		const body = `## Summary

Short summary.

## Motivation

Short motivation.

## Changes

Short changes.

## Testing

Ran test suite locally.
`
		sigs := detectSignals(t, body, detect.KindPR)
		sec, ok := sigs["pr_section_count"]
		if !ok || sec.Value != 4 {
			t.Errorf("pr_section_count = %+v (present=%v), want Value=4", sec, ok)
		}
		tmpl, ok := sigs["agent_pr_template"]
		if !ok || tmpl.Value != 2 {
			t.Errorf("agent_pr_template = %+v (present=%v), want Value=2 (not section count 4)", tmpl, ok)
		}
		chk, ok := sigs["pr_checklist"]
		if !ok || chk.Value != 0 || chk.Weight != 0 {
			t.Errorf("pr_checklist = %+v (present=%v), want Value=0 Weight=0", chk, ok)
		}
	})
}

func TestKindProseIsolationAndExistingSignals(t *testing.T) {
	const prLikeProse = `## 🚀 Summary

- Add validation for empty configuration files in startup
- Add regression coverage for empty byte slices in loader
- Keep existing YAML parse error reporting unchanged
- Return zero-value config when file has zero bytes

## Test plan

- [x] Run go test ./...

🤖 Generated with [Claude Code](https://claude.com/claude-code)
`
	proseSigs := detectSignals(t, prLikeProse, detect.KindProse)
	for _, forbidden := range []string{"agent_pr_template", "pr_section_count", "pr_checklist", "emoji_headings", "bullet_uniformity"} {
		if s, ok := proseSigs[forbidden]; ok {
			t.Errorf("KindProse unexpectedly emitted PR-only signal %q: %+v", forbidden, s)
		}
	}
	attr, ok := proseSigs["ai_attribution"]
	if !ok || !attr.Decisive || attr.Value != 1 {
		t.Errorf("KindProse ai_attribution = %+v (present=%v), want present Decisive=true Value=1", attr, ok)
	}

	prSigs := detectSignals(t, prLikeProse, detect.KindPR)
	for _, want := range []string{"ai_attribution", "agent_pr_template", "pr_section_count", "pr_checklist", "emoji_headings", "bullet_uniformity"} {
		if _, ok := prSigs[want]; !ok {
			t.Errorf("KindPR missing expected signal %q; got %v", want, prSigs)
		}
	}
}
