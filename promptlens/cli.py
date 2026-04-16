"""PromptLens CLI — analyze ChatGPT conversation exports."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import parser, topics, intents, loops, shapes


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  ✓ {path.name} ({path.stat().st_size:,} bytes)", file=sys.stderr)


def _generate_report(
    topics_data: dict,
    intents_data: dict,
    loops_data: dict,
    shapes_data: dict,
    n_convos: int,
) -> str:
    """Generate a human-readable Markdown report."""
    dna = shapes_data.get("usage_dna", {})
    tp = dna.get("time_patterns", {})

    lines = [
        "# PromptLens Analysis Report",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Conversations analyzed:** {n_convos}",
        "",
        "---",
        "",
        "## Usage DNA",
        f"- Average prompt length: **{dna.get('avg_prompt_length_words', '?')} words**",
        f"- Median prompt length: **{dna.get('median_prompt_length_words', '?')} words**",
        f"- Vocabulary richness: **{dna.get('vocabulary_richness', '?')}** ({dna.get('unique_words', '?')} unique / {dna.get('total_words', '?')} total)",
        f"- Avg conversation length: **{dna.get('avg_conversation_length_turns', '?')} turns**",
        f"- Prompt complexity trend: **{dna.get('prompt_complexity_trend', '?')}**",
        f"- Most active hour (UTC): **{tp.get('most_active_hour_utc', '?')}**",
        f"- Most active day: **{tp.get('most_active_day', '?')}**",
        f"- Sessions/week: **{tp.get('sessions_per_week_avg', '?')}**",
        "",
        "## Prompt Shapes",
        "| Shape | Count | % |",
        "|-------|------:|--:|",
    ]
    for s in shapes_data.get("prompt_shapes", []):
        lines.append(f"| {s['shape']} | {s['frequency']} | {s['pct']*100:.1f}% |")

    lines += [
        "",
        "## Intent Distribution",
        "| Intent | Count |",
        "|--------|------:|",
    ]
    for intent, count in sorted(
        intents_data.get("intent_distribution", {}).items(),
        key=lambda x: x[1], reverse=True
    ):
        lines.append(f"| {intent} | {count} |")

    lines += [
        "",
        f"## Topics ({len(topics_data.get('topics', []))} discovered)",
    ]
    for i, t in enumerate(topics_data.get("topics", [])[:15], 1):
        lines.append(
            f"**{i}. {t['name']}** — {t['conversation_count']} convos, "
            f"{t['message_count']} msgs | Keywords: {', '.join(t['keywords'][:5])}"
        )

    lines += [
        "",
        f"## Loops Detected: {loops_data.get('total_loops_detected', 0)}",
    ]
    for l in loops_data.get("loops", [])[:10]:
        lines.append(
            f"- **{l['topic']}** (sim={l['similarity']}) — "
            f"{l['titles'][0][:40]} ↔ {l['titles'][1][:40]}"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="promptlens", description="Analyze ChatGPT exports")
    sub = ap.add_subparsers(dest="command")

    analyze_p = sub.add_parser("analyze", help="Run full analysis pipeline")
    analyze_p.add_argument("input", nargs="+", help="Path(s) to conversations.json (supports multiple files from different platforms)")
    analyze_p.add_argument("--output-dir", default="./promptlens-output", help="Output directory")
    analyze_p.add_argument("--topics", type=int, default=20, help="Number of topics to discover")
    analyze_p.add_argument("--similarity-threshold", type=float, default=0.4, help="Loop detection threshold")

    args = ap.parse_args()

    if args.command != "analyze":
        ap.print_help()
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("PromptLens v0.1.0", file=sys.stderr)
    print(f"Input: {', '.join(args.input)}", file=sys.stderr)
    print(f"Output: {out}/", file=sys.stderr)
    print(file=sys.stderr)

    # 1. Parse (merge multiple files)
    print("[1/5] Parsing conversations...", file=sys.stderr)
    convos = []
    for inp in args.input:
        batch = parser.load(inp)
        platform = batch[0]["source_platform"] if batch else "unknown"
        print(f"  → {inp}: {len(batch)} conversations ({platform})", file=sys.stderr)
        convos.extend(batch)
    n = len(convos)
    n_msgs = sum(len(c["user_messages"]) for c in convos)
    print(f"  → Total: {n} conversations, {n_msgs} user messages", file=sys.stderr)

    # 2. Topics
    print("[2/5] Extracting topics...", file=sys.stderr)
    topics_data = topics.extract(convos, n_topics=args.topics)
    _write_json(topics_data, out / "topics.json")

    # 3. Intents
    print("[3/5] Classifying intents...", file=sys.stderr)
    intents_data = intents.classify(convos)
    _write_json(intents_data, out / "intents.json")

    # 4. Loops
    print("[4/5] Detecting loops...", file=sys.stderr)
    loops_data = loops.detect(convos, threshold=args.similarity_threshold)
    _write_json(loops_data, out / "loops.json")

    # 5. Shapes
    print("[5/5] Analyzing prompt shapes...", file=sys.stderr)
    shapes_data = shapes.analyze(convos)
    _write_json(shapes_data, out / "shapes.json")

    # Report
    print("Generating report...", file=sys.stderr)
    report = _generate_report(topics_data, intents_data, loops_data, shapes_data, n)
    report_path = out / "report.md"
    report_path.write_text(report)
    print(f"  ✓ report.md", file=sys.stderr)

    print(f"\nDone. {n} conversations → {out}/", file=sys.stderr)
