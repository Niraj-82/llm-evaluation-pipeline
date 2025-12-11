"""Embedding wrapper using sentence-transformers when available."""
from typing import List, Optional
HAVE_SBERT = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAVE_SBERT = True
except Exception:
    HAVE_SBERT = False

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        if HAVE_SBERT:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    def embed_texts(self, texts: List[str]):
        if self.model is not None:
            return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return None

    def embed(self, text: str):
        emb = self.embed_texts([text])
        return emb[0] if emb is not None else None
