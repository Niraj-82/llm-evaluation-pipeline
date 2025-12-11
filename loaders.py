"""Loaders: reading chat and vector JSONs and extracting messages."""
import json
from typing import Any, Dict, List, Tuple

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_latest_messages(chat_obj: Any) -> Tuple[str, str]:
    last_user = ""
    last_ai = ""
    turns = []
    if isinstance(chat_obj, dict):
        turns = chat_obj.get("conversation_turns") or chat_obj.get("turns") or []
    elif isinstance(chat_obj, list):
        turns = chat_obj
    for turn in turns:
        role = str(turn.get("role","")).lower()
        msg = turn.get("message","") or ""
        sender_id = turn.get("sender_id", None)
        if role.startswith("user") or role == "user" or (isinstance(sender_id,int) and sender_id != 1):
            last_user = msg
        if role.startswith("ai") or role.startswith("assistant") or (isinstance(sender_id,int) and sender_id == 1):
            last_ai = msg
    return last_user, last_ai

def normalize_vector_entries(vectors_obj: Any) -> List[Dict]:
    raw = vectors_obj
    if isinstance(vectors_obj, dict):
        raw = vectors_obj.get("vector_data") or vectors_obj.get("data") or vectors_obj.get("vectors") or vectors_obj.get("vectorData") or []
        if isinstance(raw, dict) and "vector_data" in raw:
            raw = raw["vector_data"]
    if raw is None:
        raw = []
    entries = []
    for entry in raw:
        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("content") or entry.get("snippet") or ""
            url = entry.get("source_url") or entry.get("source") or entry.get("url") or ""
            entries.append({"text": text, "source_url": url, "meta": entry})
        else:
            entries.append({"text": str(entry), "source_url": "", "meta": {}})
    return entries
