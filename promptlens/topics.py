"""Topic extractor — TF-IDF + NMF clustering."""
from __future__ import annotations
import re
from datetime import datetime, timezone
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np


_STOP_EXTRAS = {
    "like", "just", "want", "need", "make", "use", "using", "get", "would",
    "could", "also", "one", "know", "think", "sure", "okay", "yes", "yeah",
    "please", "thanks", "thank", "hey", "hi", "hello", "code", "file",
    "something", "thing", "way", "good", "great", "right", "really",
    "let", "try", "look", "see", "take", "give", "tell", "going",
}


def _convo_text(convo: dict) -> str:
    """Combine all user messages into one document per conversation."""
    return " ".join(m["text"] for m in convo["user_messages"])


def _ts_to_date(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


def extract(conversations: list[dict], n_topics: int = 20) -> dict:
    """Run topic extraction. Returns topics.json structure."""
    docs = [_convo_text(c) for c in conversations]

    # TF-IDF
    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=2,
        max_df=0.85,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    try:
        tfidf = vec.fit_transform(docs)
    except ValueError:
        return _empty(conversations)

    feature_names = vec.get_feature_names_out()

    # Filter extra stop words
    keep_mask = np.array([f not in _STOP_EXTRAS for f in feature_names])
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) < n_topics:
        return _empty(conversations)
    tfidf = tfidf[:, keep_idx]
    feature_names = feature_names[keep_idx]

    actual_topics = min(n_topics, tfidf.shape[1], tfidf.shape[0])
    if actual_topics < 2:
        return _empty(conversations)

    # NMF
    nmf = NMF(n_components=actual_topics, random_state=42, max_iter=400)
    W = nmf.fit_transform(tfidf)  # doc-topic matrix
    H = nmf.components_            # topic-term matrix

    # Assign each conversation to its dominant topic
    assignments = np.argmax(W, axis=1)
    max_scores = np.max(W, axis=1)

    topics: list[dict] = []
    uncategorized = 0

    for t_idx in range(actual_topics):
        top_term_idx = H[t_idx].argsort()[-8:][::-1]
        keywords = [feature_names[i] for i in top_term_idx]

        members = np.where(assignments == t_idx)[0]
        if len(members) == 0:
            continue

        # Check if assignments are meaningful
        weak = [i for i in members if max_scores[i] < 0.01]
        uncategorized += len(weak)
        strong = [i for i in members if max_scores[i] >= 0.01]
        if not strong:
            continue

        convo_ids = [conversations[i]["id"] for i in strong]
        msg_count = sum(len(conversations[i]["user_messages"]) for i in strong)

        dates = [conversations[i].get("create_time") for i in strong]
        dates = [d for d in dates if d]
        first = _ts_to_date(min(dates)) if dates else None
        last = _ts_to_date(max(dates)) if dates else None

        # Name topic from top keywords
        name = " / ".join(keywords[:3]).title()

        topics.append({
            "name": name,
            "conversation_count": len(strong),
            "message_count": msg_count,
            "keywords": keywords,
            "conversation_ids": convo_ids,
            "first_seen": first,
            "last_seen": last,
            "trend": "stable",
        })

    topics.sort(key=lambda t: t["conversation_count"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_conversations": len(conversations),
        "total_user_messages": sum(len(c["user_messages"]) for c in conversations),
        "topics": topics,
        "uncategorized_count": uncategorized,
    }


def _empty(conversations: list[dict]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_conversations": len(conversations),
        "total_user_messages": sum(len(c["user_messages"]) for c in conversations),
        "topics": [],
        "uncategorized_count": len(conversations),
    }
