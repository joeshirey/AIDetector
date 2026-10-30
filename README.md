# aidetector

I wanted a way to check if some piece of English text was written by AI. This
gives it a score from 0 to 100 and tells you why. You can run it as an MCP
server (Claude Code, Codex, and Antigravity all work) or from the command line.

I tuned it for the writing I deal with day to day: blog posts, memos,
reports, emails, essays, and pull request descriptions. Everything
runs on the local machine from one Go binary with the models baked in. No API
calls. Nothing gets downloaded.

It's a personal project and not a product. Have a look at the
[limits](#limits) first if you're going to put much weight on a score.

## Install

You'll need Go 1.27 or newer.

```sh
go install github.com/joeshirey/AIDetector/cmd/aidetector@latest
go install github.com/joeshirey/AIDetector/cmd/aidetector-mcp@latest

# register the MCP server and install the skill in Claude Code, Codex, and Antigravity
aidetector setup
```

`setup` calls each client's own CLI (`claude mcp add`, `codex mcp add`, or
`agy mcp add`). You can run it again later and it'll update the existing
entry. If one of the clients isn't installed it gets skipped. You'll see the
command to run by hand. Use `-client claude,codex` to pick clients and
`-dry-run` to see what it would do.

It also puts a copy of the `ai-text-detection` skill in the skills folder
for each client (`~/.claude/skills`, `~/.codex/skills`, `~/.gemini/config/skills`).
Restart the clients afterward.

To register by hand:

```sh
claude mcp add --scope user aidetector -- "$(go env GOPATH)/bin/aidetector-mcp"
codex mcp add aidetector -- "$(go env GOPATH)/bin/aidetector-mcp"
agy mcp add aidetector "$(go env GOPATH)/bin/aidetector-mcp"
```

## Using it

There are two MCP tools:

| Tool | What it does |
|---|---|
| `detect_ai_text` | Scores text. Pass `kind` (`prose` or `pr`) if you know it. |
| `detect_ai_file` | Scores a local `.txt` or `.md` file. |

From the command line:

```sh
aidetector post.md                    # score a file
pbpaste | aidetector -kind prose      # score the clipboard
```

The skill explains to agents how to read results (and what not to conclude
from them). The server also hands a short version of that guidance to each
client when it connects, in case the skill isn't installed.

### Reading a score

There's one scale for all kinds of text. I picked the cutoffs with human
writing the models had never seen. The idea is that a label should mean about
the same thing for a blog post as it does for a PR description:

| Score | Label | Share of human writing that scores this high |
|---|---|---|
| 80 to 100 | AI | about 0.5% |
| 60 to 79 | likely AI | about 2% |
| 40 to 59 | inconclusive | about 5% |
| 20 to 39 | likely human | |
| 0 to 19 | human | |

You also get:

- `reasons`: what moved the score, strongest first. Each one has a
  direction and a size. Most also show typical values for human and AI text.
- `segments` and `ai_fraction` for long prose. These score the text in
  chunks of about 200 words (handy when only part of a document is AI).
  Short chunks swing more than whole documents, so segments have their own
  cutoffs: a human segment reaches 60 about as often as a human document does.
- A `warning` reason if there were hidden characters or look-alike letters in
  the text. They get stripped out first. People use them to fool detectors
  (they also trip up detectors on perfectly innocent text). There's also a
  warning when most of the AI evidence comes from one embedded or quoted block,
  like an agent-written file pasted into a post. It gives the score without
  that block.
- An `info` reason when code, tables, YAML front matter, or headings and
  captions were left out. Only the prose gets scored.

Prose has to be at least 50 words and PR descriptions at least 25. Anything
shorter comes back as "too short to judge", because the models never saw text
that short. There are a few tells that count at any length, like "Generated
with Claude Code" in a PR description. They push the score to 97 or higher.

## How accurate is it?

I measured this on 2026-10-29 on text that was held out from training.

The test split is 20% of the corpus:

| Kind | AUROC | AI caught with 1% of human text flagged |
|---|---|---|
| Prose (blogs, technical blogs, memos, essays) | 0.996 | 96% |
| PR descriptions | 0.994 | 93% |

For false positives I used fresh human text. None of it was in training and
this half wasn't used to pick the cutoffs either. The PRs are from 105 repos
the training corpus had never touched:

| Human texts | Labeled inconclusive | Labeled likely AI or AI | Labeled AI | Segments at 60+ |
|---|---|---|---|---|
| Learner essays (293) | 0.3% | 0% | 0% | 0.4% |
| Blog posts (263) | 1.1% | 0.8% | 0% | 0.3% |
| Technical blog posts (321)* | 0.9% | 0% | 0% | 0.3% |
| Non-tech Medium, pre-2021 (656) | 3.8% | 0.8% | 0.3% | 0.8% |
| PR descriptions (980) | 2.9% | 1.7% | 0.8% | |
| Memos and emails (287) | 6.6% | 2.1% | 0.3% | |

A separate 10% of each pool (the final holdout) was checked once for this
release: 3 of 398 human prose documents and 4 of 225 PRs scored 60+, against
7 and 16 for the previous models.

\* Posts by Richard Seroter, Scott Hanselman, and technical Medium authors.
Other posts by the same authors are in training, so this row is optimistic.
Expect more false positives for an engineering writer the model has never
seen. When I left Seroter out of training entirely, 15% of his posts landed
in the middle band (that was under the older, wider band).

Prose uses one set of cutoffs for all its genres. That's why memos run a bit
high and blogs a bit low.

A few other results:

- I ran AI text past the [humanizer](https://github.com/blader/humanizer)
  skill. Between 86% and 99% of it still got labeled likely AI or AI.
- Human writing that's been polished by AI is all over the map. For blogs and
  memos, 13 to 18% of it comes out "inconclusive" and around 40% comes out
  "AI". A full polish changes most of the wording.
- AI models it hasn't seen can be a weak spot. When I trained with all the
  Claude models left out (and then all the Gemini ones), prose AUROC on the
  missing family dropped to 0.84 to 0.87. Qwen3.6 went the other way: before
  any Qwen text was in training, 98% to 100% of its prose already scored as AI.
  GPT landed in between. Before any GPT text was in training, 93% to 100% of
  GPT blog posts and essays already scored as AI, but only 68% to 84% of memos
  and 54% of gpt-6-luna's PR descriptions did. Those were the gaps the GPT
  training data closed (per-model test AUROC is now 0.96 to 1.00 for prose
  and PRs).
- Technical writing used to drift toward the middle of the scale. Adding engineering blogs to
  training cut that by about a third for authors it had never seen, and AI
  detection on technical text went up slightly.
- When I pasted about 400 words of AI text into the middle of a human post,
  a segment caught it 87.5% of the time. Segments are tuned to be as cautious
  as whole documents. Looser segment cutoffs would catch 96% of those inserts
  but would also flag 2 to 3% of segments in purely human posts.

## Limits

A score isn't proof. About 2 in 100 human texts get "likely AI" or higher.
I wouldn't base a grade, a disciplinary action, or a hiring call on it alone.

It can't tell a light AI edit from a full AI rewrite (polished text tends to
read as AI). Formal writing and templates (form letters, PR templates) also
score higher than casual writing does.

It's trained on Claude, Gemini, GPT, and Qwen. Text from other model families
(Llama, Grok, DeepSeek) may score lower than it should. English only. No code.

## How it works

The final score comes from a logistic regression. Its inputs are about 35
measured features plus one learned word-pattern score. That output gets mapped
onto the scale above. There's no neural network involved in scoring (a
700-word document takes about 14 ms).

Before any of that, prose gets cleaned up. Code (fenced, indented, or pasted
plain), tables (including ones pasted as one long line), YAML front matter,
and short heading or caption lines are left out. Sentences glued together by
lost line breaks ("why.In 1938") get split, with guards for URLs, file names,
versions, abbreviations, and dotted names like `Microsoft.BizTalk.Pipeline`.
Training and detection use the same cleanup.

The features come from four detectors:

| Detector | Measures |
|---|---|
| `ngram` | A learned word-pattern score: logistic regression over word and punctuation 1- and 2-grams (about 40,000 terms for prose). |
| `signs` | Patterns from Wikipedia's [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), like "not X but Y" constructions, AI vocabulary by model era, punctuation variety, and lexical diversity (MTLD, Yule's K). |
| `stylometric` | Sentence and paragraph rhythm, em dashes, lists of three, bold-label bullets. |
| `devtext` | PR structure (agent PR templates, uniform bullet lists) and AI tool attributions ("Generated with Claude Code"). |

`aidetector fit` trains one model per kind. The word-pattern model goes first.
Its score is then a feature in the final model. I use 5-fold out-of-fold
predictions for that, otherwise the final model would trust it too much.
Features are standardized. Each one also gets a weight for when it's
missing (lots of them can't be measured on short text). The word-pattern
score does most of the work. The hand-built features adjust it. They're also
what make the reasons readable.

`aidetector thresholds` scores a separate pool of human text. Half of it sets
the cutoffs (the points where 10%, 2%, and 0.5% of human text lands) and the
other half checks them. Those cutoffs get mapped to 40, 60, and 80. The order
of scores never changes. You can still see the model's own probability in
`raw_score`.

Some things I learned on the way. The hand-tuned rules from AI-writing guides
were not much better than a coin flip on 2026 models. Fitting their weights
helped a little. The learned word patterns did most of the work. Em dashes now
suggest a human writer (the newer models avoid them). And 2023 favorites
like "delve" have no weight at all anymore.

## Rebuilding the models

The corpus isn't checked in. `cmd/corpus` builds it from scratch.

The human text all predates ChatGPT's launch in November 2022:

| Kind | Source |
|---|---|
| pr | Merged GitHub PRs from 30 open-source repos |
| blog | Medium posts tagged tech or business ([fabiochiu/medium-articles](https://huggingface.co/datasets/fabiochiu/medium-articles)), plus Go blog posts for the human pool |
| blog (technical) | Posts by [Richard Seroter](https://seroter.com) and [Scott Hanselman](https://www.hanselman.com), and Medium posts with engineering tags (`harvest -source seroter`, `hanselman`, or `medium-tech`) |
| blog (non-tech pool) | Medium posts from before 2021 with no tech or business tags (`harvest -pool -source medium-nontech`). Used only to measure false positives and set cutoffs, never for training. |
| memo | Enron emails ([AESLC](https://huggingface.co/datasets/Yale-LILY/aeslc)) |
| essay | English-learner essays (Feedback Prize ELL and ICNALE) |

These sources each have their own terms (the blog posts belong to their
authors). The corpus isn't redistributed. Check the terms if you plan to
redistribute anything built from them.

Each human sample gets an AI "mirror". I hand a model the same diff (or PR
title, or a brief of the same post) and have it write its own version. The
topic stays the same on the human side and the AI side. The classifier has
to pick up on style because the subject gives nothing away. I've used Claude Opus, Sonnet, and Haiku 5.5 (on Vertex AI or
the Claude API), Gemini 3.8 Flash and 3.1 Pro, GPT-5.6 Sol, GPT-5 Mini, and
GPT-6 Luna (OpenAI API), and Qwen3.6 in LM Studio. Any OpenAI-compatible API
should work. Paid OpenAI runs track estimated spend from the token counts in
each response and stop at `-budget` dollars (default 20).

Human and AI text get the same cleanup. Otherwise the model could learn
where a sample came from in place of who wrote it. Links get stripped. So do issue
numbers, email greetings and sign-offs, Markdown, and image credits. The
prompts just ask for the thing itself (write a PR description for this
diff). There's also a check that refuses to run any
prompt asking for "human-sounding" output.

```sh
go build -o bin/corpus ./cmd/corpus
go build -o bin/aidetector ./cmd/aidetector

bin/corpus ping                      # check generator credentials
bin/corpus harvest                   # human samples (needs gh auth)
bin/corpus mirror -j 12              # AI mirrors (can be resumed)
bin/corpus harvest -pool             # human-only reserve that's never trained on
bin/corpus promote -kind blog -n 300 # move the pool's hardest human texts into training
bin/corpus mirror -add -gens lmstudio:qwen3.6-35b-a3b -limit 100   # add a model family

# technical blogs: harvest into their own pool, move a random 300 per source into training, mirror
bin/corpus harvest -pool -out corpus_tech -kind blog -source seroter -exclude corpus,corpus_fresh
bin/corpus promote -out corpus_tech -kind blog -n 300 -hard 0 -prefix seroter-
bin/corpus mirror -out corpus_tech -kind blog -j 12

bin/aidetector fit -kind prose corpus/blog corpus/memo corpus/essay corpus_tech/blog
bin/aidetector fit -kind pr corpus/pr
go build -o bin/aidetector ./cmd/aidetector   # embeds the new models

# set label cutoffs on fresh human text the models haven't seen
bin/corpus harvest -pool -out corpus_fresh -exclude corpus
bin/aidetector thresholds -kind prose -only documents corpus_fresh/blog/pool corpus_fresh/memo/pool corpus_fresh/essay/pool
bin/aidetector thresholds -kind prose -only segments corpus_fresh/blog/pool corpus_fresh/memo/pool corpus_fresh/essay/pool corpus_tech/blog/pool corpus_nontech/blog/pool
go build -o bin/aidetector ./cmd/aidetector
bin/aidetector fixtures      # regression check: known-human fixtures plus a report-only monitoring set

bin/aidetector eval -split test -kind prose corpus/blog corpus/memo corpus/essay corpus_tech/blog
bin/aidetector dist -kind prose corpus_fresh/blog/pool   # false-positive rate
```

Segment cutoffs use a wider pool than document cutoffs because they need a
lot of long text. They're matched to the documents the segments came from:
at each level, a human segment reaches the cutoff about as often as its
document reaches the document cutoff.

`aidetector fixtures` downloads the regression fixtures listed in
`internal/engine/testdata/fixtures.json` into a gitignored cache (the articles
belong to their authors). It fails if more than one known-human article
scores 60+ or any of their segments hits 80. The monitoring set is 2026 posts
of unknown authorship. It's report-only and never used for training or
calibration.

`promote` is the hard-negative step. It finds the human texts the current
model gets most wrong (plus some random ones) and moves those into training.
New data goes where the model is weakest.

`bin/corpus variant -type polished` (or `-type humanized`) builds a test-only
set from the test split. Those files live outside `human/` and `ai/`. That
keeps `fit` and `eval` from seeing them. Run `aidetector dist` on the folder
to see how they score.

## Overnight loop

`corpus overnight run` is an unattended improvement loop meant to run on
one machine overnight. It covers prose and PR descriptions. Each night it:

1. collects batch jobs submitted on earlier nights (results file into the corpus)
2. promotes the human pool texts production scores as most AI-like
   (train split only, main `corpus/*/pool` only, at most `-source-cap` per source)
3. scores the AI text that came in and feeds the misses (AI scored under 60) to a bandit
4. plans tomorrow's paid generation with Thompson sampling over
   genre × length band × prompt style × generator, keeping 20% uniform exploration.
   The training styles are plain brief, brief plus persona or audience, and
   continue a human opening. Edits of human drafts are eval-only, like the
   polished set: they're mostly human text and raised false positives when
   trained on, so they live in `ai_edited/`, outside `ai/`. Every prompt passes the same
   "no human-sounding" check as `mirror`.
5. generates locally with Qwen and Gemma (free) until it's time to retrain,
   one model at a time since both are served on this machine.
   Qwen is paused for now: its 2026-10-28 samples raised human false positives
   and were moved to `excluded/` (pass `-local` to opt back in)
6. fits a candidate (`fit` plus `thresholds`, the same procedure as above)
   into `runs/overnight/<night>/round-N/`. Cutoffs are fit on the
   calibration half of each pool only, with the final holdout left out.
7. gates the candidate against a baseline: production's own models with
   their cutoffs refit by that same procedure, so both are measured alike.
   Each check compares the two on the same texts:
   - test-split AUROC and recall at ≤1% false positives, per kind
     (allowed margin: 0.002 AUROC, 1 point of recall)
   - document false-positive rates at 60+ and 80+, and segment false
     positives at 60+, on the checking half of every fresh pool (margin: 0)
   - the frozen Opus, Sonnet, Argon, and GPT sets (margin: 1 point of AUROC
     or recall)
   - the Gemini 3.8 Flash, Qwen, and Gemma slices (margin: 0.005 AUROC)
   - the fixtures, which must all pass with none missing

   A check fails only when a paired bootstrap shows the candidate worse
   than its margin, after a Holm correction across all checks. That keeps
   the chance of failing a candidate that is no worse at about 5%; the
   earlier "one extra text" rule failed neutral refits most of the time.
   AUROC and recall use the unrounded model score, so there are no
   integer-score ties to cliff on.
8. measures (without changing anything) per-genre cutoffs and how much a
   genre classifier's mistakes would cost, and the label mix on the polished
   and humanized sets. On Sunday nights it also runs leave-one-family-out.
9. writes `reports/overnight/<night>.md`

At most 3 rounds are accepted a night. Data from rejected rounds stays in the corpus.

The loop never commits, pushes, promotes a model, or changes labels or
cutoffs. Promoting a candidate is a manual copy into
`internal/engine/models/`, and the report says how.

### Frozen data

`corpus overnight init` runs once. It moves 450 human sources out of the main
pools into `corpus_frozen/` (100 each of blog, memo, and essay, plus 150 PRs)
and writes `corpus_frozen/manifest.json`, a SHA-256 for every file in:

- the test split
- the fresh pools
- the fixtures
- the frozen sources
- the final holdout (10% of each fresh pool, picked by SHA-256)

The loop never reads the final holdout. Check it by hand with
`corpus overnight verify -holdout`.

The holdout and the calibration/checking halves were first drawn with
FNV-1a, whose lowest bit is a parity of the input bytes. As a result the
whole holdout sat inside the calibration half. On 2026-10-29 both moved to
SHA-256 and the holdout was redrawn (`corpus overnight redraw-holdout`).
Production's shipped cutoffs were fit on all pool text, so the holdout is
untouched only for candidates. Every night starts by verifying the other
sets, and it skips retraining if anything changed.

New generations only ever go to train-split sources, and promotion only moves
train-split ids, so the test split stays fixed.

The frozen generator sets (Opus, Sonnet, Argon, and Gemma) mirror those 450
sources from briefs kept in `briefs/frozen/`, which nothing else uses:

```sh
bin/corpus overnight frozen-gen -plan -gens claude:claude-opus-5-5,claude:claude-sonnet-5-5,gemini:gemini-4-argon
bin/corpus overnight frozen-gen -budget-usd 20 -gens claude:claude-opus-5-5,claude:claude-sonnet-5-5,gemini:gemini-4-argon
bin/corpus overnight frozen-gen -gens mlx:gemma-4-26b-a4b-it-8bit     # local, free (start the server first)
bin/corpus overnight collect                       # later: file batch results
bin/corpus overnight seal -gen claude-opus-5-5     # once complete; repeat per generator
```

The GPT frozen set is the GPT mirrors in the test split; there's no new GPT
generation. `corpus overnight gpt-check` reports its size and looks for
source texts or diffs that appear on both sides of the split.

### Models and money

| Generator | Role |
|---|---|
| Haiku 5.5 | ~45% of paid samples |
| Gemini 3.8 Flash | ~25% |
| Sonnet 5.5 | ~15% |
| Argon (`gemini-4-argon`) | ~15% from week 2, with approval (`-argon-training`); eval-only before that |
| Opus 5.5 | frozen set only |
| Qwen3.6 35B A3B (LM Studio), Gemma 4 26B A4B (`mlx_lm.server`) | free, local |

Gemma 4 31B (8-bit) made 15.8 tokens/s alone but 13.2 next to Qwen, under
the 15 tokens/s floor, so the loop uses the 26B A4B. LM Studio wouldn't
register a model downloaded outside it, so Gemma runs in `mlx_lm.server`
(the `mlx:` provider, `MLX_URL`, default port 8081) with its default sampling
(temperature 1.0, top-p 0.95, top-k 64). Qwen keeps the LM Studio settings
earlier mirrors used.

The bandit shifts budget toward the generators whose text gets missed. No
paid generator drops below 5% or rises above twice its base share.

Paid models all run on Vertex AI. Claude uses `ANTHROPIC_VERTEX_PROJECT_ID`
and `CLOUD_ML_REGION=global`. Gemini uses application-default credentials;
`gemini-api:` with `GEMINI_API_KEY` is only a fallback.

Bulk generation goes through Vertex batch prediction:

- Requests are JSONL in `gs://dss-model-study-aidetector-batch/batch/<night>/`.
- Claude batches run in us-east5; Gemini batches use the global location.
- Jobs are pipelined: tonight's submissions are collected tomorrow night.
- A job's cost counts against the night it was submitted.
- A failed job is retried once, then dropped.

Every call is logged to `runs/ledger.jsonl` with input, output, thinking, and
cache-read tokens and an estimated cost from the price table in
`internal/corpus/prices.json`. The table is dated and can be overridden with
`-prices`. Claude batches run regionally and cost 0.55× the global online
price (half off, plus 10% for the region); Gemini batches run on the global
endpoint at 0.50×. Thinking is billed as output.

`-budget-usd` is a hard cap for the night (40 for full nights; Joe's ceiling
is 50), and `-model-caps` sets per-model caps on top. A run over $2 refuses
to start until `corpus overnight approve` has been run.

If the Cloud Billing export to BigQuery is enabled, set
`AIDETECTOR_BILLING_TABLE=project.dataset.gcp_billing_export_v1_XXXX`. The loop
then compares billed Vertex AI cost with the ledger for nights 2 to 7 days
back. If billing runs more than 15% over the ledger, it lowers that night's
cap by the same ratio and flags it in the report.

Price notes:

- Haiku 5.5 on Vertex is $0.10/$0.50 per million tokens (checked
  2026-10-29). The table first had $0.80/$4, a legacy Haiku rate, so ledger
  entries for Haiku before 2026-10-29 are about 8× too high.
- The paid shares above are shares of samples, not dollars. The loop works
  out how many samples of each fit the night's budget at each model's price.

Recommended Vertex quotas for a $40 night (project `dss-model-study`):

| Quota | Recommendation | Why |
|---|---|---|
| Batch prediction: concurrent jobs per region (us-east5 and global) | 10 | 4 paid generators, plus retries and frozen or effort jobs |
| Claude Haiku 5.5 online, us-east5/global | 200k input and 40k output tokens/min | briefs are written online, one at a time |
| Claude Sonnet 5.5 online | 20k input and 4k output tokens/min | morning analysis only |
| Gemini online | defaults | almost everything Gemini does is batch |

Batch jobs draw on their own batch quota, not the online tokens-per-minute quota.

### Running it

Run it from a worktree so the code under test doesn't move while you work
in the main checkout. The data stays in the main checkout:

```sh
git worktree add ../AIDetector-overnight -b overnight/$(date +%F)
cd ../AIDetector-overnight && go build -o bin/corpus ./cmd/corpus

# Gemma runs in mlx_lm.server (the loop uses Qwen and Gemma in turn, never together)
uv run --with mlx-lm mlx_lm.server --port 8081 \
  --model ~/.lmstudio/models/mlx-community/gemma-4-26b-a4b-it-8bit &

# dry run: about 20 requests per generator, one round, $2 cap
bin/corpus overnight run -data ~/Code/AIDetector -budget-usd 2 -until 23:30

# full night, after approval
bin/corpus overnight run -data ~/Code/AIDetector -budget-usd 40 -until 07:00
```

Ctrl-C stops it cleanly; it still writes the report. Restarting the same
night resumes after the last finished step, and the spend so far is restored
from the ledger. `corpus overnight status` lists batch jobs.

The effort experiment runs once. For each paid model it generates the same
50 sources at default and at low effort. `effort-report` compares the two
distributions; `effort-paired` compares each source's two versions directly
(mean paired difference with a 95% interval, and a Wilcoxon test), split by
prose and PR. Ten PR pairs per model are too few to show the two efforts
match, so treat low effort for PRs as unproven.

```sh
bin/corpus overnight effort -plan
bin/corpus overnight effort -budget-usd 10
bin/corpus overnight collect && bin/corpus overnight effort-report   # writes runs/effort/result.json
bin/corpus overnight effort-paired
```

### Diagnostics

These cost nothing (no API calls):

```sh
# fit a candidate from the corpus as it is and gate it against the baseline
bin/corpus overnight gate -name exp/try1
#   -exclude-since T [-exclude-gen PREFIX]   leave out AI text written after T
#   -exclude-human-since T                   leave out human files added after T
#   -exclude-style persona,continue          leave out generations of these styles
#   -weighting source                        weight each source and its mirrors equally
#   -keep-cutoffs                            keep production's cutoffs (no refit)
#   -candidate DIR                           gate an existing candidate instead
bin/corpus overnight flips -name exp/try1    # pool texts that cross 60, by id
```

## Experimental: Binoculars scoring

`scorer/server.py` is a small service you run yourself. It loads a base model
along with its instruct version (Qwen3-8B, using MLX on Apple Silicon). For
each text it reports log-perplexity, the Binoculars ratio, and Fast-DetectGPT
curvature. I was hoping for a signal that didn't care which company's model
wrote the text.

It didn't help much. AUROC on held-out model families went up by 0.02 to 0.03.
On the test split it didn't change at all. I haven't shipped any models
trained with it. The binaries ignore it unless you fit those yourself (the
script's docstring has more).

```sh
uv run scorer/server.py --observer mlx-community/Qwen3-8B-Base-bf16 --performer mlx-community/Qwen3-8B-bf16
export AIDETECTOR_SCORER_URL=http://127.0.0.1:8765
bin/aidetector fit -lm -kind prose corpus/blog corpus/memo corpus/essay   # writes prose.lm.json
```

## Layout

```
cmd/aidetector        CLI: detect, eval, fit, thresholds, dist, setup, fixtures
cmd/aidetector-mcp    MCP server (stdio)
cmd/corpus            corpus builder: harvest, mirror, promote, variant, overnight
internal/detect/      detectors (stylometric, signs, devtext, ngram, perplexity)
internal/engine       scoring, the calibrated scale, reasons, segments, embedded models
internal/calibrate    logistic regression for the final model
internal/train        fitting and cutoff calibration (shared by the CLI and the overnight loop)
internal/overnight    the overnight loop: bandit, batch jobs, frozen sets, gate, report
internal/setup        client setup and the embedded skill
scorer/               experimental Binoculars service (Python, MLX)
```

## Credits

The patterns come from Wikipedia's Signs of AI writing (WikiProject AI
Cleanup) and the humanizer skill. Thanks to Richard Seroter and Scott
Hanselman, whose years of engineering posts made the technical training data. The zero-shot methods are from Binoculars
(Hans et al., 2024) and Fast-DetectGPT (Bao et al., 2024). The mirrored
training data and hard-negative mining follow published work on supervised
AI-text detection.

## License

Apache 2.0 (see [LICENSE](LICENSE)).
