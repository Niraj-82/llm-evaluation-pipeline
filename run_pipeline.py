"""User-facing entrypoint. Run from repository root."""
import argparse, sys, os
from evaluation import evaluate
from utils import write_json

def parse_args():
    p = argparse.ArgumentParser(description="Run the LLM evaluation pipeline")
    p.add_argument("--chat_json", required=True, help="Path to chat JSON")
    p.add_argument("--vectors_json", required=True, help="Path to vectors JSON")
    p.add_argument("--ai_response_file", required=False, help="Optional AI response override")
    p.add_argument("--output", required=False, default="eval_report.json", help="Output JSON path")
    p.add_argument("--embed_model", required=False, default="all-MiniLM-L6-v2")
    p.add_argument("--nli_model", required=False, default="roberta-large-mnli")
    p.add_argument("--top_k_sources", required=False, type=int, default=5)
    p.add_argument("--price_per_1k_tokens", required=False, type=float, default=0.02)
    return p.parse_args()

def main():
    args = parse_args()
    ai_override = None
    if args.ai_response_file:
        try:
            with open(args.ai_response_file, "r", encoding="utf-8") as fh:
                ai_override = fh.read()
        except Exception as e:
            print("Failed to read ai_response_file:", e, file=sys.stderr)
            sys.exit(2)
    report = evaluate(
        chat_json=args.chat_json,
        vectors_json=args.vectors_json,
        ai_response_override=ai_override,
        embed_model_name=args.embed_model,
        nli_model_name=args.nli_model,
        top_k_sources=args.top_k_sources,
        price_per_1k=args.price_per_1k_tokens
    )
    write_json(args.output, report)
    print("Saved report to", os.path.abspath(args.output))

if __name__ == "__main__":
    main()
