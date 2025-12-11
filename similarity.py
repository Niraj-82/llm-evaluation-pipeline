"""Similarity helpers and simple fallbacks."""
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple
import math

def simple_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 1
    return max(1, int(len(text) / 4))

def estimate_cost(tokens: int, price_per_1k: float) -> float:
    return (tokens / 1000.0) * float(price_per_1k)

def top_k_by_simple_similarity(query: str, sources: List[Dict[str,Any]], k:int=5) -> List[Tuple[float,int,Dict[str,Any]]]:
    sims = []
    for i,s in enumerate(sources):
        txt = s.get("text","") or ""
        sims.append((simple_similarity(query, txt), i, s))
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:k]
