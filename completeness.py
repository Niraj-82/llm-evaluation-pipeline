"""Keypoint extraction and coverage scoring."""
from typing import List, Dict, Any, Optional, Tuple
def safe_split_sentences(text: str):
    import re
    if not text:
        return []
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents or [text.strip()]

def extract_keypoints_from_sources(sources: List[Dict[str,Any]], nlp=None, max_k:int=10) -> List[str]:
    keypoints = []
    for s in sources:
        txt = s.get("text","") or ""
        if not txt:
            continue
        sents = safe_split_sentences(txt)
        if not sents:
            continue
        first = sents[0].strip()
        if nlp is not None:
            try:
                doc = nlp(first)
                nouns = [chunk.text.strip() for chunk in doc.noun_chunks][:3]
                kp = " ; ".join(nouns) if nouns else (first if len(first.split())<=30 else " ".join(first.split()[:30])+"...")
            except Exception:
                kp = first if len(first.split())<=30 else " ".join(first.split()[:30])+"..."
        else:
            kp = first if len(first.split())<=30 else " ".join(first.split()[:30])+"..."
        if kp not in keypoints:
            keypoints.append(kp)
        if len(keypoints) >= max_k:
            break
    return keypoints

from typing import Tuple
from similarity import simple_similarity
from embedder import Embedder
def compute_completeness(response: str, keypoints: List[str], embedder: Optional[Embedder]=None, threshold: float=0.55) -> Tuple[float, List[Dict[str,Any]]]:
    matches = []
    for kp in keypoints:
        sim = 0.0
        if embedder is not None and hasattr(embedder,'model') and embedder.model is not None:
            try:
                r_emb = embedder.embed(response)
                kp_emb = embedder.embed(kp)
                # use simple similarity if embeddings not available
                if r_emb is not None and kp_emb is not None:
                    # lazy cosine: rely on sentence-transformers util inside embedder env if available
                    from sentence_transformers import util as sbert_util  # type: ignore
                    sim = float(sbert_util.cos_sim(r_emb, kp_emb).cpu().numpy())
                else:
                    sim = simple_similarity(response, kp)
            except Exception:
                sim = simple_similarity(response, kp)
        else:
            sim = simple_similarity(response, kp)
        matches.append({"keypoint": kp, "similarity": float(sim), "covered": bool(sim >= threshold)})
    covered = sum(1 for m in matches if m["covered"])
    score = (covered / len(matches)) if matches else 1.0
    return float(score), matches
