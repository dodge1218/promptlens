"""Loop detector — find repeated topic conversations via TF-IDF cosine similarity."""
from __future__ import annotations
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def detect(conversations: list[dict], threshold: float = 0.4) -> dict:
    """Detect conversation loops. Returns loops.json structure."""
    if len(conversations) < 2:
        return _empty()

    docs = [" ".join(m["text"] for m in c["user_messages"]) for c in conversations]

    vec = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        min_df=1,
        max_df=0.9,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    try:
        tfidf = vec.fit_transform(docs)
    except ValueError:
        return _empty()

    sim = cosine_similarity(tfidf)

    # Find pairs above threshold (upper triangle only)
    loops: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for i in range(len(conversations)):
        for j in range(i + 1, len(conversations)):
            if sim[i, j] >= threshold and (i, j) not in seen:
                seen.add((i, j))
                ci, cj = conversations[i], conversations[j]

                dates = []
                for c in (ci, cj):
                    ct = c.get("create_time")
                    if ct:
                        try:
                            dates.append(datetime.fromtimestamp(ct, tz=timezone.utc).strftime("%Y-%m-%d"))
                        except (ValueError, OSError):
                            pass

                # Guess topic from shared top terms
                shared_terms = _shared_keywords(tfidf[i], tfidf[j], vec.get_feature_names_out())

                loops.append({
                    "conversation_ids": [ci["id"], cj["id"]],
                    "titles": [ci["title"], cj["title"]],
                    "topic": " / ".join(shared_terms[:3]) if shared_terms else "unknown",
                    "similarity": round(float(sim[i, j]), 3),
                    "message_count": len(ci["user_messages"]) + len(cj["user_messages"]),
                    "date_range": sorted(dates) if dates else [],
                    "loop_type": "repeated_question",
                    "resolution": "unresolved",
                })

    loops.sort(key=lambda l: l["similarity"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_loops_detected": len(loops),
        "loops": loops[:100],  # cap output size
    }


def _shared_keywords(vec_a, vec_b, feature_names) -> list[str]:
    """Find keywords that are strong in both vectors."""
    a = vec_a.toarray().flatten()
    b = vec_b.toarray().flatten()
    combined = np.minimum(a, b)
    top_idx = combined.argsort()[-5:][::-1]
    return [feature_names[i] for i in top_idx if combined[i] > 0]


def _empty() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_loops_detected": 0,
        "loops": [],
    }
