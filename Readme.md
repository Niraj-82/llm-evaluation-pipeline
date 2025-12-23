# LLM Evaluation Pipeline

A modular Python-based pipeline for evaluating LLM responses in real time.  
The system analyzes AI outputs across three major dimensions:

- **Relevance** — How well the response aligns with the user message and retrieved context.
- **Completeness** — Whether key information from the context is covered.
- **Hallucination Detection** — Identifies unsupported or contradicted claims using NLI or fallback logic.

This project was created as part of a technical assignment and is designed to be clean, scalable, and easy to extend.

---

## 🚀 Features

- Supports **sentence-transformers** for embeddings (optional).
- Uses **HuggingFace NLI models** when available.
- Works even without heavy ML dependencies (fallback modes included).
- Modular architecture for easy maintenance and expansion.
- Detailed evaluation report (JSON) with:
  - Relevance score
  - Hallucination analysis
  - Claim breakdown
  - Completeness metrics
  - Latency timings
  - Token cost estimation

---

## 📁 Project Structure

llm_evaluation_pipeline/
│
├── loaders.py # JSON loading and message extraction
├── embedder.py # SBERT wrapper
├── similarity.py # Similarity + cost heuristics
├── claims.py # Claim extraction
├── nli_engine.py # Entailment / contradiction checker
├── completeness.py # Keypoint extraction + coverage scoring
├── evaluation.py # Main evaluator orchestrator
├── utils.py # Small helpers
└── main.py # Internal CLI entrypoint

run_pipeline.py # Root-level CLI for end users

---


## 📦 Installation


### 1. Install dependencies


pip install -r requirements.txt
(You may create the file using your preferred dependency versions.)

Optional packages that improve evaluation accuracy:

sentence-transformers

transformers

torch

spacy

numpy

The pipeline will automatically fall back to lightweight heuristics if these packages are not installed.


▶️ Usage
Run the evaluation pipeline using:

python run_pipeline.py \
  --chat_json sample-chat-conversation-02.json \
  --vectors_json sample_context_vectors-02.json \
  --output eval_report.json
Optional arguments:
Argument	Description
--ai_response_file	Supply custom AI response instead of reading from chat JSON
--embed_model	SentenceTransformer model name
--nli_model	HuggingFace NLI model name
--top_k_sources	Number of context sources considered
--price_per_1k_tokens	Token cost estimation


🧪 Output
The script produces a JSON report containing:

Relevance metrics

Completeness score

Hallucination score

Extracted claims + evidence

Timing per pipeline stage

Cost estimate


Example snippet:

json
Copy code
{
  "relevance": {
    "score": 0.82,
    "user_similarity": 0.79,
    "avg_top_source_similarity": 0.84
  },
  "hallucination": {
    "score": 0.12,
    "claims": [...]
  }
}


🛠 Development Notes

The system is designed to run in constrained environments.

Each module can be extended independently.

You can swap out embedding/NLI models without altering the orchestration logic.

