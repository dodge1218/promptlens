"""Prompt shape analyzer — usage patterns, vocabulary, time analysis."""
from __future__ import annotations
import re
from collections import Counter
from datetime import datetime, timezone


def _classify_shape(text: str) -> str:
    """Classify a prompt's shape."""
    wc = len(text.split())
    has_question = text.rstrip().endswith("?")
    has_code = bool(re.search(r"```|def |class |import |function |const |let |var ", text))
    has_url = bool(re.search(r"https?://", text))

    if has_code and wc > 50:
        return "code_paste"
    if has_code:
        return "code_snippet"
    if wc <= 5:
        return "ultra_short"
    if wc <= 15 and has_question:
        return "short_question"
    if wc <= 15:
        return "short_command"
    if wc <= 50 and has_question:
        return "medium_question"
    if wc <= 50:
        return "medium_instruction"
    if wc <= 150:
        return "long_instruction"
    return "essay_prompt"


def analyze(conversations: list[dict]) -> dict:
    """Analyze prompt shapes and usage DNA. Returns shapes.json structure."""
    all_user_msgs: list[dict] = []
    all_words: list[str] = []
    word_counts: list[int] = []
    convo_lengths: list[int] = []
    hours: list[int] = []
    days: list[str] = []
    shape_counter: Counter = Counter()

    for convo in conversations:
        um = convo["user_messages"]
        convo_lengths.append(len(convo["messages"]))
        for msg in um:
            all_user_msgs.append(msg)
            wc = msg["word_count"]
            word_counts.append(wc)
            words = re.findall(r"[a-zA-Z]{2,}", msg["text"].lower())
            all_words.extend(words)

            shape = _classify_shape(msg["text"])
            shape_counter[shape] = shape_counter.get(shape, 0) + 1

            ts = msg.get("create_time")
            if ts:
                try:
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    hours.append(dt.hour)
                    days.append(dt.strftime("%A"))
                except (ValueError, OSError):
                    pass

    total = len(all_user_msgs) or 1
    unique_words = set(all_words)
    vocab_richness = round(len(unique_words) / max(len(all_words), 1), 4)

    # Domain spread from topics (rough — count unique 2-grams)
    word_freq = Counter(all_words)
    top_domains_raw = word_freq.most_common(50)

    # Time patterns
    hour_counter = Counter(hours)
    day_counter = Counter(days)
    peak_hour = hour_counter.most_common(1)[0][0] if hour_counter else None
    peak_day = day_counter.most_common(1)[0][0] if day_counter else None

    # Weekly cadence
    dates_set = set()
    for convo in conversations:
        ct = convo.get("create_time")
        if ct:
            try:
                dates_set.add(datetime.fromtimestamp(ct, tz=timezone.utc).strftime("%Y-W%W"))
            except (ValueError, OSError):
                pass
    weeks = len(dates_set) or 1
    sessions_per_week = round(len(conversations) / weeks, 1)

    # Prompt complexity trend (compare first half vs second half avg word count)
    if len(word_counts) > 10:
        mid = len(word_counts) // 2
        first_half_avg = sum(word_counts[:mid]) / mid
        second_half_avg = sum(word_counts[mid:]) / (len(word_counts) - mid)
        if second_half_avg > first_half_avg * 1.15:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.85:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    # Build shape list
    shape_descs = {
        "ultra_short": "≤5 words, terse command",
        "short_question": "≤15 words, ends with ?",
        "short_command": "≤15 words, imperative",
        "medium_question": "16-50 words, ends with ?",
        "medium_instruction": "16-50 words, directive",
        "long_instruction": "51-150 words, detailed directive",
        "essay_prompt": ">150 words, extended context",
        "code_paste": ">50 words with code blocks",
        "code_snippet": "Code included, short context",
    }

    prompt_shapes = []
    for shape, count in shape_counter.most_common():
        prompt_shapes.append({
            "shape": shape,
            "pattern": shape_descs.get(shape, shape),
            "frequency": count,
            "pct": round(count / total, 3),
        })

    import numpy as np
    wc_arr = word_counts if word_counts else [0]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usage_dna": {
            "avg_prompt_length_words": round(float(np.mean(wc_arr)), 1),
            "median_prompt_length_words": round(float(np.median(wc_arr)), 1),
            "vocabulary_richness": vocab_richness,
            "unique_words": len(unique_words),
            "total_words": len(all_words),
            "avg_conversation_length_turns": round(float(np.mean(convo_lengths)), 1) if convo_lengths else 0,
            "domain_spread": min(len(unique_words) // 50, 50),
            "prompt_complexity_trend": trend,
            "time_patterns": {
                "most_active_hour_utc": peak_hour,
                "most_active_day": peak_day,
                "sessions_per_week_avg": sessions_per_week,
            },
        },
        "prompt_shapes": prompt_shapes,
    }
