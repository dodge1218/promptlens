"""Parser — Load and linearize ChatGPT conversation exports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def _text_from_parts(parts: list | None) -> str:
    """Extract text from content.parts, skipping images/dicts."""
    if not parts:
        return ""
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, str):
            chunks.append(p)
        elif isinstance(p, dict) and "text" in p:
            chunks.append(p["text"])
    return "\n".join(chunks).strip()


def _walk_tree(mapping: dict[str, Any]) -> list[dict]:
    """Walk the mapping tree root→leaf and return ordered messages."""
    # Find root node (no parent or parent not in mapping)
    root_id = None
    for nid, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = nid
            break
    if root_id is None:
        return []

    messages: list[dict] = []
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        node = mapping.get(nid, {})
        msg = node.get("message")
        if msg and msg.get("content"):
            role = msg.get("author", {}).get("role", "unknown")
            text = _text_from_parts(msg["content"].get("parts"))
            if text and role in ("user", "assistant"):
                messages.append({
                    "id": msg.get("id", nid),
                    "role": role,
                    "text": text,
                    "create_time": msg.get("create_time"),
                    "model": msg.get("metadata", {}).get("model_slug"),
                    "word_count": len(text.split()),
                })
        children = node.get("children", [])
        if children:
            queue.append(children[0])  # follow first branch
    return messages


def load(path: str | Path) -> list[dict]:
    """Load a ChatGPT/Claude/Grok/AIStudio export and return parsed conversations.

    Accepts any PromptLens-compatible JSON (ChatPort, ClaudeExport,
    GrokExport, AIStudioExport, or official OpenAI export).

    Each conversation dict has:
      id, title, create_time, update_time, messages: [{role, text, ...}],
      source_platform (str)
    """
    path = Path(path)
    raw = json.loads(path.read_text(errors="replace"))

    # Detect source platform
    source_platform = "chatgpt"
    if isinstance(raw, dict) and raw.get("metadata", {}).get("source_platform"):
        source_platform = raw["metadata"]["source_platform"]

    # Handle both formats
    if isinstance(raw, list):
        convos_raw = raw
    elif isinstance(raw, dict):
        convos_raw = raw.get("conversations", raw.get("data", []))
    else:
        raise ValueError(f"Unexpected top-level type: {type(raw)}")

    conversations: list[dict] = []
    for c in convos_raw:
        mapping = c.get("mapping")
        if not mapping:
            continue
        messages = _walk_tree(mapping)
        if not messages:
            continue
        conversations.append({
            "id": c.get("conversation_id", c.get("id", "")),
            "title": c.get("title") or "(untitled)",
            "create_time": c.get("create_time"),
            "update_time": c.get("update_time"),
            "messages": messages,
            "user_messages": [m for m in messages if m["role"] == "user"],
            "source_platform": source_platform,
        })
    return conversations
