"""Intent classifier — rule-based + keyword signals."""
from __future__ import annotations
import re
from datetime import datetime, timezone

_PATTERNS: list[tuple[str, list[re.Pattern], list[str], float]] = [
    # (intent, regex_patterns, keyword_signals, base_confidence)
    ("debug", [
        re.compile(r"\b(error|bug|fix|broken|crash|fail|traceback|exception|stacktrace|typeerror|syntaxerror|not working|doesn.t work)\b", re.I),
    ], ["debug", "issue", "wrong", "unexpected", "stderr"], 0.80),

    ("question", [
        re.compile(r"^(what|how|why|where|when|who|which|is|are|can|could|does|do|will|would|should)\b", re.I),
        re.compile(r"\?\s*$"),
    ], [], 0.75),

    ("instruction", [
        re.compile(r"^(write|create|build|make|generate|add|remove|update|change|set|implement|refactor|deploy|install|run|execute)\b", re.I),
    ], ["please", "now", "step by step"], 0.80),

    ("brainstorm", [
        re.compile(r"\b(idea|brainstorm|think about|explore|what if|imagine|vision|concept|strategy|approach|options?|alternatives?)\b", re.I),
    ], ["possibilities", "pros and cons", "tradeoff"], 0.70),

    ("creative", [
        re.compile(r"\b(write a (story|poem|song|essay|article|blog)|creative|narrative|fiction|draft me|tone|voice)\b", re.I),
    ], ["rewrite", "rephrase", "style"], 0.75),

    ("meta", [
        re.compile(r"\b(you are|your (role|instructions|system)|act as|pretend|persona|context window|token|prompt)\b", re.I),
    ], ["system prompt", "jailbreak", "rules"], 0.70),
]


def _classify_one(text: str) -> tuple[str, float]:
    """Return (intent, confidence) for a single message."""
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for intent, regexes, keywords, base_conf in _PATTERNS:
        score = 0.0
        for pat in regexes:
            if pat.search(text):
                score += base_conf
        for kw in keywords:
            if kw in text_lower:
                score += 0.15
        if score > 0:
            scores[intent] = min(score, 0.95)

    if not scores:
        return ("other", 0.50)

    best = max(scores, key=scores.get)  # type: ignore
    return (best, round(scores[best], 2))


def classify(conversations: list[dict]) -> dict:
    """Classify all user messages by intent. Returns intents.json structure."""
    dist: dict[str, int] = {
        "question": 0, "instruction": 0, "brainstorm": 0,
        "debug": 0, "creative": 0, "meta": 0, "other": 0,
    }
    messages: list[dict] = []

    for convo in conversations:
        for msg in convo["user_messages"]:
            intent, conf = _classify_one(msg["text"])
            dist[intent] = dist.get(intent, 0) + 1
            messages.append({
                "conversation_id": convo["id"],
                "message_id": msg["id"],
                "text_preview": msg["text"][:120],
                "intent": intent,
                "confidence": conf,
                "word_count": msg["word_count"],
            })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_messages_classified": len(messages),
        "intent_distribution": dist,
        "messages": messages,
    }
