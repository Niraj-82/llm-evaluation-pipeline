"""Claim extraction logic with optional spaCy support."""
from typing import List, Optional
HAVE_SPACY = False
try:
    import spacy  # type: ignore
    HAVE_SPACY = True
except Exception:
    HAVE_SPACY = False

def safe_split_sentences(text: str):
    import re
    if not text:
        return []
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents or [text.strip()]

def extract_claims(text: str, nlp=None) -> List[str]:
    if not text:
        return []
    sentences = []
    if nlp is not None:
        try:
            doc = nlp(text)
            for sent in doc.sents:
                s = sent.text.strip()
                if not s:
                    continue
                ents = list(sent.ents)
                has_number = any(tok.like_num for tok in sent)
                has_assertive_verb = any(tok.lemma_.lower() in ("is","are","was","were","will","have","has","may","must","can") for tok in sent)
                if ents or has_number or has_assertive_verb or len(s.split())>10:
                    sentences.append(s)
        except Exception:
            sentences = safe_split_sentences(text)
    else:
        sents = safe_split_sentences(text)
        for s in sents:
            low = s.lower()
            if any(ch.isdigit() for ch in s) or any(k in low for k in (" will "," is "," are "," has "," have "," may "," must "," can ")) or len(s.split())>10:
                sentences.append(s)
    uniq = []
    for s in sentences:
        if s not in uniq:
            uniq.append(s)
    return uniq
