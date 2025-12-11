"""NLI wrapper using transformers when available; fallback heuristics otherwise."""
from typing import Tuple, Optional
HAVE_TRANSFORMERS = False
try:
    from transformers import pipeline  # type: ignore
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

from similarity import simple_similarity

class NLIPipeline:
    def __init__(self, model_name: Optional[str] = None):
        self.pipe = None
        self.model_name = model_name
        if HAVE_TRANSFORMERS and model_name:
            try:
                self.pipe = pipeline("text-classification", model=model_name, return_all_scores=False)  # type: ignore
            except Exception:
                self.pipe = None

    def verify(self, claim: str, source_text: str) -> Tuple[str, float]:
        if not claim.strip() or not source_text.strip():
            return "unsupported", 0.0
        if self.pipe is None:
            if claim.lower() in source_text.lower():
                return "entailment", 0.8
            sim = simple_similarity(claim, source_text)
            if sim > 0.65:
                return "entailment", sim
            if sim < 0.2:
                return "contradiction", 1.0-sim
            return "neutral", sim
        try:
            out = None
            try:
                out = self.pipe({"premise": source_text, "hypothesis": claim})
            except Exception:
                out = self.pipe(f"{claim} </s></s> {source_text}")
            if isinstance(out, list) and out and isinstance(out[0], dict) and "label" in out[0]:
                label = out[0]["label"].lower()
                score = float(out[0].get("score",0.0))
                if label.startswith("entail"):
                    return "entailment", score
                if label.startswith("contrad"):
                    return "contradiction", score
                return "neutral", score
            return "neutral", 0.0
        except Exception:
            if claim.lower() in source_text.lower():
                return "entailment", 0.75
            sim = simple_similarity(claim, source_text)
            if sim > 0.65:
                return "entailment", sim
            if sim < 0.2:
                return "contradiction", 1.0-sim
            return "neutral", sim
