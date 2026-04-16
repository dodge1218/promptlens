# PromptLens

**See how you actually use AI.** Topic patterns, conversation loops, prompt shapes, and usage DNA from your own ChatGPT history.

```
promptlens analyze conversations.json
```

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/runs-100%25%20local-orange" />
</p>

---

## Why

Every AI provider already has analytics on your usage — topic clusters, prompt patterns, conversation loops, engagement curves. They use it for model training, safety research, and product decisions.

You can't see any of it.

PromptLens gives you the same insights from your own data. Export your conversations, run the pipeline, see your patterns. No API calls. No cloud. Everything stays on your machine.

## What It Finds

| Module | What It Does |
|--------|-------------|
| **Topic Clustering** | Groups your conversations into discovered topics using TF-IDF |
| **Intent Classification** | Labels each prompt: question, instruction, brainstorm, debug, creative, meta |
| **Loop Detection** | Finds conversations where you asked the same thing twice (and didn't resolve it) |
| **Prompt Shapes** | Categorizes prompts by structure: short command, medium instruction, essay, code paste, etc. |
| **Usage DNA** | Your fingerprint: vocabulary richness, prompt length distribution, active hours, session frequency |

## Install

```bash
pip install -r requirements.txt
```

Requires: Python 3.10+, scikit-learn, numpy. No GPU. No API keys. No network access.

## Usage

### 1. Export your data

**ChatGPT:** Settings → Data Controls → Export Data → download `conversations.json`

### 2. Run analysis

```bash
python -m promptlens analyze conversations.json
```

Options:
```
--output-dir DIR     Output directory (default: ./promptlens-output)
--topics N           Number of topics to discover (default: 20)
--similarity-threshold F   Loop detection threshold (default: 0.4)
```

### 3. Read your report

```
promptlens-output/
├── report.md      ← Human-readable summary
├── topics.json    ← Topic clusters with keywords & conversation IDs
├── intents.json   ← Intent distribution across all prompts
├── loops.json     ← Detected conversation loops
└── shapes.json    ← Prompt shapes + usage DNA fingerprint
```

### Multiple exports

Merge exports from different platforms or time periods:

```bash
python -m promptlens analyze chatgpt-export.json claude-export.json grok-export.json
```

## Example Output

```
PromptLens v0.1.0
Input: conversations.json
Output: ./promptlens-output/

[1/5] Parsing conversations...
  → conversations.json: 215 conversations (chatgpt)
  → Total: 215 conversations, 695 user messages
[2/5] Extracting topics...
  ✓ topics.json (18,432 bytes)
[3/5] Classifying intents...
  ✓ intents.json (312 bytes)
[4/5] Detecting loops...
  ✓ loops.json (891 bytes)
[5/5] Analyzing prompt shapes...
  ✓ shapes.json (1,204 bytes)
Generating report...
  ✓ report.md

Done. 215 conversations → ./promptlens-output/
```

### Sample Usage DNA

```
Average prompt length: 39.5 words
Median prompt length: 23 words
Vocabulary richness: 0.18 (4,610 unique / 25,618 total)
Avg conversation length: 6.7 turns
Sessions/week: 43
```

### Sample Prompt Shapes

| Shape | % |
|-------|---|
| Medium instruction (16-50 words, directive) | 38.1% |
| Short command (≤15 words, imperative) | 19.7% |
| Long instruction (50+ words) | 16.3% |
| Ultra short ("yes", "continue") | 8.2% |
| Questions | 12.4% |

## How It Works

- **No ML models required.** Topic clustering uses TF-IDF + k-means from scikit-learn. Intent classification is rule-based (prompt structure analysis). Loop detection uses cosine similarity between conversation TF-IDF vectors.
- **~500 lines of Python.** Deliberately simple. You can read every line in 20 minutes.
- **Deterministic.** Same input → same output, every time. No randomness, no API calls.
- **Fast.** 215 conversations in under 10 seconds on a laptop.

## Input Format

Accepts:
1. **Official OpenAI export** — `conversations.json` from Settings → Data Controls → Export
2. **Any export with OpenAI's `mapping` structure** — same conversation tree format

See [`schema.json`](schema.json) for the full input schema.

## Privacy

- **Zero network calls.** The pipeline never touches the internet.
- **Read-only.** Input files are never modified.
- **No telemetry.** No analytics. No data collection. No phoning home.
- **Local only.** All processing happens on your machine. Output stays in your output directory.

Your conversations are yours. PromptLens just helps you see them clearly.

## Roadmap

- [x] HTML dashboard with visualizations
- [ ] Comparison mode: your patterns vs public datasets
- [ ] Prompt quality scoring (specificity, context density)
- [ ] Time-series analysis (how your prompting evolves)
- [ ] Claude / Gemini / Grok export support
- [ ] Plugin system for custom analyzers

## License

MIT

---

*Built by [Ryan](https://github.com/dodge1218) at [DreamSiteBuilders.com](https://dreamsitebuilders.com).*
