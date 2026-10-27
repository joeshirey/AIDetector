# aidetector

Scores English text from 0 (human) to 100 (AI-written). Tuned for blog posts,
internal memos and reports, commit messages, and PR descriptions. Ships as a
CLI and an MCP server. Both are pure-Go static binaries (`CGO_ENABLED=0`), so
they cross-compile anywhere.

## Install

```sh
go build -o bin/aidetector ./cmd/aidetector
go build -o bin/aidetector-mcp ./cmd/aidetector-mcp

# Register with Claude Code (user scope = available in every project)
claude mcp add --scope user aidetector -- "$PWD/bin/aidetector-mcp"
```

## MCP tools

| Tool | Purpose |
|---|---|
| `detect_ai_text` | Score a string. Optional `kind`: `auto`, `prose`, `commit`, `pr`. |
| `detect_ai_file` | Score a local `.txt`/`.md` file. |
| `detect_ai_commits` | Score commit messages in a repo range (e.g. `main..HEAD`). |

Each result includes `score`, `label`, `confidence`, ranked `reasons` (each
with a direction, an impact level, its contribution to the fitted score, and the
typical human and AI values from the training corpus), the raw `signals`, and,
for long prose, per-segment scores plus `ai_fraction`. Before scoring, text is
sanitized: zero-width characters and look-alike letters (e.g. Cyrillic "а") are
removed and reported as a warning, since they're used for evasion and also make
detectors falsely accuse human writers.

## CLI

```sh
aidetector post.md                  # score a file
pbpaste | aidetector -kind memo     # score stdin
aidetector commits -n 50 main..HEAD # score commits
aidetector eval ./corpus            # measure accuracy on labeled data
```

## How it works

```
internal/detect/            Detector interface + kind detection
internal/detect/stylometric rhythm, AI vocabulary, stock phrases, em dashes, formatting
internal/detect/signs       Wikipedia "Signs of AI writing" / humanizer patterns, era-specific AI vocabulary
internal/detect/ngram       word-pattern classifier (word + punctuation 1–2-grams) learned from the corpus
internal/detect/devtext     AI co-author trailers, agent PR templates, commit structure
internal/engine             combines signals (log-odds), length confidence, segments
internal/eval               AUROC / false-positive report over a labeled corpus
```

Signals are combined in log-odds space and shrunk toward 50 for short text.
An explicit AI attribution (e.g. `Co-Authored-By: Claude`) sets a floor of 97.

Kinds with a fitted model in `internal/engine/models/` are scored by that
model. Others fall back to hand-set weights, which are rough guesses.

### Held-out accuracy (2026-10-27, 2,793 human/AI pairs)

Test split (20%, includes hard negatives chosen by the previous model):

| Kind | AUROC | AI caught at ≤1% false positives |
|---|---|---|
| Prose (blogs, memos, essays) | 0.989 | 89% |
| PRs | 0.985 | 85% |
| Commits | 0.856 | 40% |

False-positive rate on a fresh human pool, never trained on or used for
mining (commits and PRs from 10 repos the corpus never used):

| Human texts | Scored 50+ | Scored 80+ |
|---|---|---|
| Learner essays (600) | 0.0% | 0.0% |
| Blogs (600) | 0.2% | 0.0% |
| Memos (600) | 3.7% | 0.3% |
| PRs (300) | 5.7% | 0.0% |
| Commits (600) | 17.8% | 1.8% |

Eval-only sets: after a humanizer rewrite, 87–98% of AI texts still score 50+.
Human text polished by AI lands in between (blog and memo median ~47, essays
66, PRs 8). Commits are the weak spot: treat a commit score below ~80 as weak
evidence.

### Where the rules come from

`internal/detect/signs` encodes the patterns in Wikipedia's
[Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) and the
[humanizer](https://github.com/blader/humanizer) skill built from it. Fitting showed which
still matter for 2026 models: AI vocabulary drifts by era (2023 words like "delve" now carry no
weight, mid-2026 words do), current models keep sentences in an 8–20 word band, avoid dashes,
semicolons, and parentheses, and vary word choice more than people. Em dashes now point
toward human writing.

## Building the corpus

`cmd/corpus` harvests human text written before Nov 2022 and generates an AI
"mirror" of each sample: the same diff, PR title, or content brief, written by a model.

| Kind | Human source | AI mirror prompt |
|---|---|---|
| commit | GitHub commits from 20 popular repos | the same diff (some prompts include the repo's recent subjects) |
| pr | merged GitHub PRs | the same title + diff |
| blog | Medium posts tagged tech/business | a brief extracted from the post |
| memo | Enron emails (AESLC) | a brief extracted from the email |
| essay | English-learner essays (Feedback Prize ELL, ICNALE) | a brief extracted from the essay |

```sh
go build -o bin/corpus ./cmd/corpus
bin/corpus ping                 # check generator credentials
bin/corpus harvest              # needs gh auth; writes corpus/<kind>/human + src
bin/corpus mirror -j 12         # resumable; writes corpus/<kind>/ai/<model>/
bin/corpus stats
```

### Human pool and hard-negative mining

Growth follows a hard-negative active-learning loop, driven by the model's own
mistakes rather than random sampling:

```sh
bin/corpus harvest -pool                 # human-only reserve (adds Go blog posts and learner essays)
bin/aidetector dist -kind prose corpus/essay/pool   # false-positive rate at scale
bin/corpus promote -kind blog -n 300 -hard 0.5      # half hardest-to-classify human texts, half random
bin/corpus mirror -j 12
```

The pool is never trained on, so what remains in it is a large, free
false-positive test set, including non-native English writing.

### Eval-only variant sets

```sh
bin/corpus variant -type polished  -kind blog,memo,pr,essay -n 60   # human test text polished by a model
bin/corpus variant -type humanized -kind blog,memo,pr,essay -n 60   # AI test text rewritten with the humanizer skill
bin/aidetector dist -kind prose corpus/blog/ai_polished
```

These live in `ai_polished/` and `ai_humanized/`, which `fit` and `eval` never
read. They measure how the detector handles AI-assisted and evasion-rewritten
text without training on either.

Generators rotate across Claude Opus 5.5, Sonnet 5.5, and Haiku 5.5 (Vertex AI
when `ANTHROPIC_VERTEX_PROJECT_ID` is set, otherwise the Claude API) and Gemini 3.8
Flash and 3.1 Pro (`GEMINI_API_KEY`). Override with `-gens provider:model,...`.

To keep the detector from learning dataset artifacts rather than AI style,
both sides are normalized the same way (`text.Normalize`, also applied at detection time):
commit trailers, `(#123)` suffixes, URLs, issue references, and @mentions are stripped from
commits and PRs; email greetings and sign-offs are stripped from memos; Medium image credits
are removed; AESLC line breaks are repaired; and AI prose has its Markdown stripped (the human
prose sources are plain text). `corpus renormalize` re-applies this to an existing corpus.

Generation prompts must ask for the artifact plainly. `checkPrompts` refuses to run if a
prompt asks the model to sound human or avoid AI tells, so the training data reflects
normal use. Humanized text, if wanted, belongs in a separate adversarial set that is never
trained on.

## Calibrating

Samples are split 80/20 by id (a human text and its mirror always share a split). `fit`
trains the word-pattern model, then the calibration model on all signals, using 5-fold
out-of-fold word-pattern scores so the calibrator doesn't trust an overfit classifier.

```sh
bin/aidetector fit -kind commit corpus/commit
bin/aidetector fit -kind pr corpus/pr
bin/aidetector fit -kind prose corpus/blog corpus/memo corpus/essay
go build -o bin/aidetector ./cmd/aidetector       # embeds internal/engine/models/*.json
bin/aidetector eval -split test -kind commit corpus/commit
bin/aidetector eval -split test -kind commit -heuristic corpus/commit   # compare to hand-set weights
```

## Ideas adopted from "Forensics for AI Generated Text"

Adopted: hard-negative mining and a human-only false-positive pool, Unicode sanitization, explanations with direction, impact, and
reference ranges, signpost and corporate-verb features, Yule's K and MTLD, and an
eval-only AI-polished set (the report's "AI-assisted" middle category).

Not adopted: hand-set thresholds and fusion weights (the report's burstiness
ranges don't match 2026 models in our corpus; fitted weights replace them), and
LLM-as-a-judge. Deferred: Binoculars / Fast-DetectGPT (needs token
log-probabilities from an open model) and token-level span tagging.

## Roadmap

1. **Phase 1 (done):** heuristic detectors, MCP server, CLI, eval harness.
2. **Phase 2 (in progress):** corpus builder and logistic-regression
   calibration (done); next, a perplexity/Binoculars detector using an API that
   returns prompt logprobs (Together or Fireworks, `echo` + `logprobs`).
3. **Phase 3:** generate "mirror" AI texts with Claude/GPT/Gemini from human
   prompts and real diffs; fine-tune a ModernBERT classifier and serve it
   behind HTTP so the Go binary stays pure-Go.
4. **Phase 4:** mine hard negatives, retrain, refine segment-level scoring.
