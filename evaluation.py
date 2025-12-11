"""Orchestrator: ties modules together and produces the report."""
from typing import Dict, Any, Optional
from loaders import read_json, extract_latest_messages, normalize_vector_entries
from embedder import Embedder
from similarity import top_k_by_simple_similarity, simple_similarity
from claims import extract_claims
from nli_engine import NLIPipeline
from completeness import extract_keypoints_from_sources, compute_completeness
import time
import os

def evaluate(
    chat_json:str,
    vectors_json:str,
    ai_response_override:Optional[str]=None,
    embed_model_name:str="all-MiniLM-L6-v2",
    nli_model_name:Optional[str]="roberta-large-mnli",
    top_k_sources:int=5,
    price_per_1k:float=0.02
) -> Dict[str,Any]:
    t_start = time.time()
    timings = {}

    chat = read_json(chat_json)
    vectors = read_json(vectors_json)
    timings["load_inputs"] = time.time() - t_start

    t0 = time.time()
    user_msg, ai_msg = extract_latest_messages(chat)
    ai_response = (ai_response_override.strip() if ai_response_override else ai_msg or "")
    timings["extract_messages"] = time.time() - t0

    vector_entries = normalize_vector_entries(vectors)
    timings["build_sources"] = time.time() - t0

    t1 = time.time()
    embedder = Embedder(embed_model_name)
    source_embeddings = None
    try:
        texts = [s["text"] or "" for s in vector_entries]
        if texts and embedder.model is not None:
            source_embeddings = embedder.embed_texts(texts)
    except Exception:
        source_embeddings = None
    timings["embeddings_prep"] = time.time() - t1

    t2 = time.time()
    # relevance: user similarity and top source similarity
    if embedder.model is not None:
        try:
            resp_emb = embedder.embed(ai_response)
            user_sim = 0.0
            if user_msg:
                user_emb = embedder.embed(user_msg)
                if resp_emb is not None and user_emb is not None:
                    from sentence_transformers import util as sbert_util  # type: ignore
                    user_sim = float(sbert_util.cos_sim(resp_emb, user_emb).cpu().numpy())
                else:
                    user_sim = simple_similarity(ai_response, user_msg)
        except Exception:
            user_sim = simple_similarity(ai_response, user_msg)
    else:
        user_sim = simple_similarity(ai_response, user_msg)

    top_sources = top_k_by_simple_similarity(ai_response, vector_entries, k=top_k_sources)
    avg_top = sum(t[0] for t in top_sources) / (len(top_sources) or 1)
    relevance_score = 0.5 * float(user_sim) + 0.5 * float(avg_top)
    timings["relevance"] = time.time() - t2

    t3 = time.time()
    nlp = None
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
    except Exception:
        nlp = None
    claims = extract_claims(ai_response, nlp=nlp)
    timings["claim_extraction"] = time.time() - t3

    t4 = time.time()
    nli = NLIPipeline(nli_model_name if nli_model_name else None)
    claim_results = []
    unsupported = 0
    contrad = 0
    for claim in claims:
        candidates = top_k_by_simple_similarity(claim, vector_entries, k=3)
        evidence = []
        best_verdict = "unsupported"
        best_score = 0.0
        for sim_score, idx, src in candidates:
            verdict, vscore = nli.verify(claim, src.get("text","") or "")
            evidence.append({
                "source_index": idx,
                "source_url": src.get("source_url",""),
                "similarity": float(sim_score),
                "nli_verdict": verdict,
                "nli_score": float(vscore),
                "text_snippet": (src.get("text","") or "")[:400] + "..."
            })
            if verdict == "entailment" and vscore > best_score:
                best_verdict = "entailment"
                best_score = vscore
            if verdict == "contradiction" and vscore > best_score:
                best_verdict = "contradiction"
                best_score = vscore
        if best_verdict == "unsupported":
            unsupported += 1
        if best_verdict == "contradiction":
            contrad += 1
        claim_results.append({"claim": claim, "best_verdict": best_verdict, "best_score": float(best_score), "evidence": evidence})
    timings["claim_verification"] = time.time() - t4

    t5 = time.time()
    total_claims = len(claim_results)
    hallucination = float((unsupported + contrad) / total_claims) if total_claims>0 else 0.0
    timings["hallucination_calc"] = time.time() - t5

    t6 = time.time()
    top_source_dicts = [t[2] for t in top_sources]
    keypoints = extract_keypoints_from_sources(top_source_dicts, nlp=nlp, max_k=8)
    completeness_score, completeness_matches = compute_completeness(ai_response, keypoints, embedder if embedder.model is not None else None, threshold=0.55)
    timings["completeness"] = time.time() - t6

    t7 = time.time()
    from similarity import estimate_tokens_from_text, estimate_cost
    tokens = estimate_tokens_from_text(ai_response) + estimate_tokens_from_text(user_msg) + sum(estimate_tokens_from_text(s.get("text","")[:1000]) for s in top_source_dicts)
    cost = estimate_cost(tokens, price_per_1k)
    timings["costs_calc"] = time.time() - t7

    timings["total_elapsed"] = time.time() - t_start

    report = {
        "metadata": {
            "chat_file": os.path.abspath(chat_json),
            "vectors_file": os.path.abspath(vectors_json),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "embed_model_used": embed_model_name if embedder.model is not None else None,
            "nli_model_used": nli.model_name if getattr(nli,'pipe',None) is not None else None
        },
        "input": {"user_message": user_msg, "ai_response": ai_response},
        "relevance": {"score": float(relevance_score), "user_similarity": float(user_sim), "avg_top_source_similarity": float(avg_top), "top_sources": [{"score":float(t[0]), "index":int(t[1]), "source_url":t[2].get("source_url",""), "snippet": (t[2].get("text","") or "")[:400]} for t in top_sources]},
        "completeness": {"score": float(completeness_score), "keypoints": completeness_matches},
        "hallucination": {"score": float(hallucination), "total_claims": total_claims, "unsupported_claims": unsupported, "contradicted_claims": contrad, "claims": claim_results},
        "latency": timings,
        "costs": {"estimated_tokens": int(tokens), "price_per_1k": float(price_per_1k), "estimated_cost": float(cost)}
    }
    return report
