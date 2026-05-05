# PromptLens

**Local-first AI usage analytics for your own exported conversations.**

PromptLens turns personal AI conversation exports into readable, local reports: topic clusters, prompt patterns, unresolved loops, and workflow signals.

```bash
python -m promptlens analyze conversations.json
```

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/runs-100%25%20local-orange" />
</p>

---


## Full writeup and example prompts

PromptLens is a work in progress. For the broader product thesis, example prompts, and why this category matters, see Ryan's Google Cloud NEXT writing-challenge article:

**[Google Just Unlocked Something Huge With Gemini Memory Import, Here’s How to Actually Profit From It](https://dev.to/vonb/google-just-unlocked-something-huge-with-gemini-memory-import-heres-how-to-actually-profit-from-2ckf)**

That piece explains the larger workflow: once a user can import or export AI history, the next step is turning that archive into practical intelligence, voice profiles, unfinished-idea mining, pattern recognition, personal SOPs, and decision archaeology. PromptLens is the local/open-source prototype layer for that same direction.

## Why

AI platforms already understand broad usage patterns: what people ask for, where conversations loop, which workflows keep recurring, and how prompt structure changes outcomes.

Individual users usually do not get that visibility into their own exports.

PromptLens is a small, readable prototype for personal AI-usage analytics. It processes exports locally and produces artifacts that help a user answer questions like:

- What do I keep asking AI systems to help with?
- Where do conversations loop without resolving?
- Which prompt structures do I use most?
- What workflows are becoming repeated operating patterns?

## Market signal

This repo sits in the same product direction as the AI-usage analytics and personal-workflow insight tools recently highlighted around Google Next / hackathon-style demos: users and teams want visibility into how they actually use AI, not just access to another chat box.

PromptLens is independent, unaffiliated, and deliberately local-first: no cloud account, no telemetry, no API calls, and no vendor lock-in.

## What It Finds

| Module | What It Does |
|--------|-------------|
| **Topic Clustering** | Groups conversations into discovered topics using TF-IDF |
| **Intent Classification** | Labels prompts: question, instruction, brainstorm, debug, creative, meta |
| **Loop Detection** | Finds conversations where similar requests recur without clear resolution |
| **Prompt Shapes** | Categorizes prompt structure: short command, medium instruction, essay, code paste, etc. |
| **Workflow Signals** | Summarizes vocabulary, prompt length distribution, activity patterns, and repeated workflows |

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, scikit-learn, and numpy. No GPU, API key, or network access required.

## Usage

### 1. Export your data

For ChatGPT: Settings → Data Controls → Export Data → download `conversations.json`.

### 2. Run analysis

```bash
python -m promptlens analyze conversations.json
```

Options:

```text
--output-dir DIR              Output directory (default: ./promptlens-output)
--topics N                    Number of topics to discover (default: 20)
--similarity-threshold F      Loop detection threshold (default: 0.4)
```

### 3. Read your report

```text
promptlens-output/
├── report.md      ← human-readable summary
├── topics.json    ← topic clusters with keywords and conversation IDs
├── intents.json   ← intent distribution across prompts
├── loops.json     ← detected recurring conversation loops
└── shapes.json    ← prompt-shape and workflow-signal summary
```

## Example Output

```text
PromptLens v0.1.0
Input: conversations.json
Output: ./promptlens-output/

[1/5] Parsing conversations...
  → conversations.json: 215 conversations (chatgpt)
  → Total: 215 conversations, 695 user messages
[2/5] Extracting topics...
  ✓ topics.json
[3/5] Classifying intents...
  ✓ intents.json
[4/5] Detecting loops...
  ✓ loops.json
[5/5] Analyzing prompt shapes...
  ✓ shapes.json
Generating report...
  ✓ report.md
```

## How It Works

- **No hosted model required.** Topic clustering uses TF-IDF + k-means from scikit-learn.
- **Rule-based classification.** Intent and prompt-shape labels are simple, inspectable heuristics.
- **Deterministic.** Same input, same output.
- **Readable code.** The prototype is intentionally small enough to audit quickly.

## Input Format

Currently accepts:

1. Official OpenAI export, `conversations.json` from Settings → Data Controls → Export.
2. Any export with the same conversation `mapping` structure.

See [`schema.json`](schema.json) for the input schema.

## Privacy

- **Zero network calls.** The pipeline does not contact external services.
- **Read-only input.** Source exports are never modified.
- **No telemetry.** No analytics, tracking, or usage collection.
- **Local output.** Reports stay in the directory you choose.

Do not commit real conversation exports or generated private reports to a public repository.

## Roadmap

- [x] HTML dashboard prototype
- [ ] Comparison mode across exports/time periods
- [ ] Prompt quality heuristics: specificity, context density, constraint clarity
- [ ] Time-series analysis: how prompting changes over time
- [ ] Claude / Gemini / Grok export adapters
- [ ] Plugin system for custom analyzers

## License

MIT

---

Built by [Ryan Vonbrubeck](https://github.com/dodge1218).
