#!/usr/bin/env python3
"""
MacPro AI — CLI query script.

Usage:
    python query.py "Show blood test results for patient P001"
    python query.py "Find X-ray related to lung infection" --top-k 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="MacPro AI — Query medical documents")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=10, help="Number of sources to retrieve")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis; show raw retrieval")
    args = parser.parse_args()

    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.synthesizer import LLMSynthesizer

    pipeline = RetrievalPipeline()
    result = pipeline.retrieve(args.query)
    sources = result.sources[:args.top_k]

    print(f"\n── Query: {args.query!r}")
    if result.query_filters:
        print(f"── Filters extracted: {result.query_filters}")

    if not args.no_llm:
        synthesizer = LLMSynthesizer()
        try:
            answer = synthesizer.synthesize(args.query, sources)
            print(f"\n── Answer ──────────────────────────────────────────")
            print(answer)
        except Exception as e:
            print(f"\n── LLM Synthesis Failed ─────────────────────────────")
            print(f"Error: {e}")
            print("Showing raw retrieval results only.")

    print(f"\n── Sources ({len(sources)}) ─────────────────────────────────")
    for i, src in enumerate(sources, 1):
        loc = f"page {src.page}" if src.page else "n/a"
        star = " ★" if src.type.value == "image" else ""
        print(f"\n[{i}] {src.type.value.upper()}{star} | {src.filename} | {loc} | score={src.score}")
        if src.snippet:
            print(f"    {src.snippet[:200]}")
        if src.path_or_uri:
            print(f"    path: {src.path_or_uri}")

    print()


if __name__ == "__main__":
    main()
