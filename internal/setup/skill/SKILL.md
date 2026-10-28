---
name: ai-text-detection
description: Check whether text was written by AI using the aidetector MCP tools (detect_ai_text, detect_ai_file). Use when the user asks if something is AI-written or AI-generated, wants a blog post, memo, report, email, essay, or PR description checked for AI authorship, or asks for an AI-detection score.
---

# AI text detection

The `aidetector` MCP server scores English text from 0 (human) to 100 (AI) and
explains why. Scores are probabilistic evidence, never proof.

## Running a check

1. **Pick the tool.**
   - Text in the conversation: `detect_ai_text`.
   - A local `.txt` or `.md` file: `detect_ai_file` with an absolute path.
2. **Set `kind` when you know it**: `prose` (blog posts, memos, reports,
   emails, essays) or `pr` (pull request descriptions). Without it, text with
   PR-style section headings is treated as a PR and everything else as prose.
   Commit messages aren't supported.
3. **Check the length.** Prose under 50 words and PR descriptions under 25
   words come back labeled "too short to judge". Tell the user it's too short
   instead of reporting the number. Results are most reliable from about 150
   words.
4. **For long documents, read `segments` and `ai_fraction`.** They show which
   parts read as AI, which matters for documents with mixed authorship.
   Segments are calibrated separately so a segment at 60+ is about as rare in
   human writing as a document at 60+. Even so, one hot segment in a long,
   otherwise-human document is weak evidence on its own.

## Reading a result

| Score | Label | Meaning |
|---|---|---|
| 80–100 | AI | about 0.5% of human writing of this kind scores this high |
| 60–79 | likely AI | about 2% of human writing reaches 60 |
| 40–59 | inconclusive | about 5% of human writing reaches 40 |
| 0–39 | likely human / human | |

- `reasons` lists what drove the score, strongest first. Each has a
  direction (AI or human), an impact, and, where available, typical values
  for human and AI text. Use the top two or three when explaining.
- A reason with direction `warning` flags something to tell the user about:
  hidden or look-alike characters were removed (someone may have altered the
  text), or most of the AI evidence comes from one embedded or quoted block
  (for example a pasted agent-generated document). For the embedded-block
  warning, report the score both with and without the block.
- A reason with direction `info` says how many words of code, tables, front
  matter, or headings were left out of scoring. Mention it when it's large.
- A reason with impact `decisive` (for example "Generated with Claude Code" in
  a PR description, or "as an AI language model") is near-certain evidence.
- Ignore `raw_score` and `cutoffs` unless the user asks how scoring works.

## Presenting results

- Lead with the label and score, then the two or three strongest reasons in
  plain words, then any caveats that apply.
- Describe the text, not the person: "this reads as likely AI-written (score
  72)", not "they used AI".
- Keep it short. Don't paste the full JSON or every signal.

## Limits to keep in mind

- **Not proof.** Even at "likely AI", about 2 in 100 human texts of that kind
  score as high. Don't present a result as the sole basis for a consequential
  decision such as grading, discipline, or hiring; suggest corroborating
  evidence instead.
- **AI-polished human writing** often scores "mixed" or higher. The detector
  can't separate light AI editing from AI generation.
- **Formal or templated human writing** (form letters, boilerplate PR
  templates) scores higher than casual writing.
- **English only.** Code is not supported.
- **Unfamiliar AI models and heavy human editing** can lower scores for
  genuinely AI-written text.
