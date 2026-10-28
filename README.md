# aidetector

I wanted a way to check if some piece of English text was written by AI. This
gives it a score from 0 to 100 and tells you why. You can run it as an MCP
server (Claude Code, Codex, and Antigravity all work) or from the command line.

I tuned it for the writing I deal with day to day: blog posts, memos,
reports, emails, essays, pull request descriptions, commit messages. Everything
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

There are three MCP tools:

| Tool | What it does |
|---|---|
| `detect_ai_text` | Scores text. Pass `kind` (`prose`, `pr`, or `commit`) if you know it. |
| `detect_ai_file` | Scores a local `.txt` or `.md` file. |
| `detect_ai_commits` | Scores the commit messages in a range like `main..HEAD`. |

From the command line:

```sh
aidetector post.md                    # score a file
pbpaste | aidetector -kind prose      # score the clipboard
aidetector commits -n 50 main..HEAD   # score recent commits
```

The skill explains to agents how to read results (and what not to conclude
from them). The server also hands a short version of that guidance to each
client when it connects, in case the skill isn't installed.

### Reading a score

There's one scale for all kinds of text. I picked the cutoffs with human
writing the models had never seen. The idea is that a label should mean about
the same thing for a blog post as it does for a commit:

| Score | Label | Share of human writing that scores this high |
|---|---|---|
| 80 to 100 | AI | about 0.5% |
| 60 to 79 | likely AI | about 2% |
| 40 to 59 | mixed (possibly AI-assisted) | about 10% |
| 20 to 39 | likely human | |
| 0 to 19 | human | |

You also get:

- `reasons`: what moved the score, strongest first. Each one has a
  direction and a size. Most also show typical values for human and AI text.
- `segments` and `ai_fraction` for long prose. These score the text in
  chunks of about 200 words (handy when only part of a document is AI).
- A `warning` reason if there were hidden characters or look-alike letters in
  the text. They get stripped out first. People use them to fool detectors
  (they also trip up detectors on perfectly innocent text).

Prose has to be at least 50 words and PR descriptions at least 25. Anything
shorter comes back as "too short to judge", because the models never saw text
that short. Commit messages have no minimum. There are a few tells that count
at any length, like an AI co-author trailer on a commit. They push the score
to 97 or higher.

## How accurate is it?

I measured this in October 2026 on text that was held out from training.

The test split is 20% of the corpus:

| Kind | AUROC | AI caught with 1% of human text flagged |
|---|---|---|
| Prose (blogs, technical blogs, memos, essays) | 0.996 | 95% |
| PR descriptions | 0.990 | 87% |
| Commit messages | 0.862 | 35% |

For false positives I used fresh human text. None of it was in training and
this half wasn't used to pick the cutoffs either. The commits and PRs are from
10 repos the corpus had never touched:

| Human texts | Labeled mixed | Labeled likely AI or AI | Labeled AI |
|---|---|---|---|
| Learner essays (275) | 3% | 0% | 0% |
| Blog posts (289) | 6% | 1% | 0% |
| Technical blog posts (693)* | 4% | under 1% | 0% |
| Commit messages (282) | 4% | 1% | 0% |
| PR descriptions (157) | 4% | 2% | 0% |
| Memos and emails (293) | 17% | 6% | 3% |

\* Posts by Richard Seroter, Scott Hanselman, and technical Medium authors.
Other posts by the same authors are in training, so this row is optimistic.
For an engineering writer the model has never seen, expect closer to 15%
"mixed" (that's what I got when I left Seroter out of training entirely).

Prose uses one set of cutoffs for all its genres. That's why memos run a bit
high and blogs a bit low.

A few other results:

- I ran AI text past the [humanizer](https://github.com/blader/humanizer)
  skill. Between 86% and 99% of it still got labeled likely AI or AI.
- Human writing that's been polished by AI is all over the map. For blogs and
  memos, about a quarter of it comes out "mixed" and around 40% comes out
  "AI". A full polish changes most of the wording.
- AI models it hasn't seen can be a weak spot. When I trained with all the
  Claude models left out (and then all the Gemini ones), prose AUROC on the
  missing family dropped to 0.84 to 0.87. Qwen3.6 went the other way: before
  any Qwen text was in training, 98% to 100% of its prose already scored as AI.
- Technical writing used to drift toward "mixed". Adding engineering blogs to
  training cut that by about a third for authors it had never seen, and AI
  detection on technical text went up slightly.

## Limits

A score isn't proof. About 2 in 100 human texts get "likely AI" or higher.
I wouldn't base a grade, a disciplinary action, or a hiring call on it alone.

Commit messages are hard. They're short and there isn't much to go on. I
wouldn't read much into one commit scoring under 80 (look at a whole batch of
commits together).

It can't tell a light AI edit from a full AI rewrite (polished text tends to
read as AI). Formal writing and templates (form letters, PR templates) also
score higher than casual writing does.

It's trained on Claude, Gemini, and Qwen. Text from other model families (GPT,
Llama, Grok) may score lower than it should. English only. No code.

## How it works

The final score comes from a logistic regression. Its inputs are about 35
measured features plus one learned word-pattern score. That output gets mapped
onto the scale above. There's no neural network involved in scoring (a
700-word document takes about 14 ms).

The features come from four detectors:

| Detector | Measures |
|---|---|
| `ngram` | A learned word-pattern score: logistic regression over word and punctuation 1- and 2-grams (about 40,000 terms for prose). |
| `signs` | Patterns from Wikipedia's [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), like "not X but Y" constructions, AI vocabulary by model era, punctuation variety, and lexical diversity (MTLD, Yule's K). |
| `stylometric` | Sentence and paragraph rhythm, em dashes, lists of three, bold-label bullets. |
| `devtext` | Commit and PR structure (change-list bullets, agent PR templates, AI co-author trailers). |

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
| commit, pr | GitHub commits and merged PRs from 30 open-source repos |
| blog | Medium posts tagged tech or business ([fabiochiu/medium-articles](https://huggingface.co/datasets/fabiochiu/medium-articles)), plus Go blog posts for the human pool |
| blog (technical) | Posts by [Richard Seroter](https://seroter.com) and [Scott Hanselman](https://www.hanselman.com), and Medium posts with engineering tags (`harvest -source seroter`, `hanselman`, or `medium-tech`) |
| memo | Enron emails ([AESLC](https://huggingface.co/datasets/Yale-LILY/aeslc)) |
| essay | English-learner essays (Feedback Prize ELL and ICNALE) |

These sources each have their own terms (the blog posts belong to their
authors). The corpus isn't redistributed. Check the terms if you plan to
redistribute anything built from them.

Each human sample gets an AI "mirror". I hand a model the same diff (or PR
title, or a brief of the same post) and have it write its own version. The
topic stays the same on the human side and the AI side. The classifier has
to pick up on style because the subject gives nothing away. I've used Claude Opus, Sonnet, and Haiku 5.5 (on Vertex AI or
the Claude API), Gemini 3.8 Flash and 3.1 Pro, and Qwen3.6 in LM Studio. Any
OpenAI-compatible API should work.

Human and AI text get the same cleanup. Otherwise the model could learn
where a sample came from in place of who wrote it. Links get stripped. So do issue
numbers, commit trailers, email greetings and sign-offs, Markdown, and image
credits. The prompts just ask for the thing itself (write a
commit message for this diff). There's also a check that refuses to run any
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
bin/aidetector fit -kind commit corpus/commit
go build -o bin/aidetector ./cmd/aidetector   # embeds the new models

# set label cutoffs on fresh human text the models haven't seen
bin/corpus harvest -pool -out corpus_fresh -exclude corpus
bin/aidetector thresholds -kind prose corpus_fresh/blog/pool corpus_fresh/memo/pool corpus_fresh/essay/pool
go build -o bin/aidetector ./cmd/aidetector

bin/aidetector eval -split test -kind prose corpus/blog corpus/memo corpus/essay corpus_tech/blog
bin/aidetector dist -kind prose corpus_fresh/blog/pool   # false-positive rate
```

`promote` is the hard-negative step. It finds the human texts the current
model gets most wrong (plus some random ones) and moves those into training.
New data goes where the model is weakest.

`bin/corpus variant -type polished` (or `-type humanized`) builds a test-only
set from the test split. Those files live outside `human/` and `ai/`. That
keeps `fit` and `eval` from seeing them. Run `aidetector dist` on the folder
to see how they score.

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
cmd/aidetector        CLI: detect, commits, eval, fit, thresholds, dist, setup
cmd/aidetector-mcp    MCP server (stdio)
cmd/corpus            corpus builder: harvest, mirror, promote, variant
internal/detect/      detectors (stylometric, signs, devtext, ngram, perplexity)
internal/engine       scoring, the calibrated scale, reasons, segments, embedded models
internal/calibrate    logistic regression for the final model
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
