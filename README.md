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

Each result includes `score`, `label`, `confidence`, the `signals` behind the
score, and, for long prose, per-segment scores plus `ai_fraction`.

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
internal/detect/devtext     AI co-author trailers, agent PR templates, commit structure
internal/engine             combines signals (log-odds), length confidence, segments
internal/eval               AUROC / false-positive report over a labeled corpus
```

Signals are combined in log-odds space and shrunk toward 50 for short text.
An explicit AI attribution (e.g. `Co-Authored-By: Claude`) sets a floor of 97.

Kinds with a fitted model in `internal/engine/models/` are scored by that
model. Others fall back to hand-set weights, which are rough guesses.

## Building the corpus

`cmd/corpus` harvests human text written before Nov 2022 and generates an AI
"mirror" of each sample: the same diff, PR title, or content brief, written by a model.

| Kind | Human source | AI mirror prompt |
|---|---|---|
| commit | GitHub commits from 20 popular repos | the same diff (some prompts include the repo's recent subjects) |
| pr | merged GitHub PRs | the same title + diff |
| blog | Medium posts tagged tech/business | a brief extracted from the post |
| memo | Enron emails (AESLC) | a brief extracted from the email |

```sh
go build -o bin/corpus ./cmd/corpus
bin/corpus ping                 # check generator credentials
bin/corpus harvest              # needs gh auth; writes corpus/<kind>/human + src
bin/corpus mirror -j 12         # resumable; writes corpus/<kind>/ai/<model>/
bin/corpus stats
```

Generators rotate across Claude Opus 5.5, Sonnet 5.5, and Haiku 5.5 (Vertex AI
when `ANTHROPIC_VERTEX_PROJECT_ID` is set, otherwise the Claude API) and Gemini 3.8
Flash and 3.1 Pro (`GEMINI_API_KEY`). Override with `-gens provider:model,...`.

To keep the detector from learning dataset artifacts rather than AI style,
both sides are normalized: commit trailers and `(#123)` suffixes are stripped,
Medium image credits are removed, AESLC line breaks are repaired, and AI prose
has its Markdown stripped (the human prose sources are plain text).

## Calibrating

Samples are split 80/20 by id (a human text and its mirror always share a split).

```sh
bin/aidetector fit -kind commit corpus/commit
bin/aidetector fit -kind pr corpus/pr
bin/aidetector fit -kind prose corpus/blog corpus/memo
go build -o bin/aidetector ./cmd/aidetector       # embeds internal/engine/models/*.json
bin/aidetector eval -split test -kind commit corpus/commit
bin/aidetector eval -split test -kind commit -heuristic corpus/commit   # compare to hand-set weights
```

## Roadmap

1. **Phase 1 (done):** heuristic detectors, MCP server, CLI, eval harness.
2. **Phase 2 (in progress):** corpus builder and logistic-regression
   calibration (done); next, a perplexity/Binoculars detector using an API that
   returns prompt logprobs (Together or Fireworks, `echo` + `logprobs`).
3. **Phase 3:** generate "mirror" AI texts with Claude/GPT/Gemini from human
   prompts and real diffs; fine-tune a ModernBERT classifier and serve it
   behind HTTP so the Go binary stays pure-Go.
4. **Phase 4:** mine hard negatives, retrain, refine segment-level scoring.
