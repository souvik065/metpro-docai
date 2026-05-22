import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval.pipeline import RetrievalPipeline
from config.settings import settings

async def debug():
    pipeline = RetrievalPipeline()
    query = "clinical management flowchart"
    
    print(f"Query: {query}")
    
    with open("debug_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")

        # ── Text search ───────────────────────────────────────────────────
        text_vec = pipeline.text_embedder.embed(query)
        text_hits = pipeline.vector_store.search_text(
            query_vector=text_vec,
            top_k=5,
        )
        f.write(f"\nText Hits ({len(text_hits)}):\n")
        for hit in text_hits:
            p = hit["payload"]
            f.write(f"  - [{p.get('asset_type')}] Score: {hit['score']:.4f}, Asset: {p.get('asset_id')}, File: {p.get('filename')}\n")
            if p.get('snippet'):
                f.write(f"    Snippet: {p.get('snippet')[:100]}\n")

        # ── Image search (CLIP text → image space) ────────────────────────
        clip_vec = pipeline.image_embedder.embed_text_for_image_search(query)
        image_hits = pipeline.vector_store.search_image(
            query_vector=clip_vec,
            top_k=5,
        )
        f.write(f"\nImage Hits ({len(image_hits)}):\n")
        for hit in image_hits:
            p = hit["payload"]
            f.write(f"  - [{p.get('asset_type')}] Score: {hit['score']:.4f}, Asset: {p.get('asset_id')}, File: {p.get('filename')}\n")
            if p.get('ocr_text'):
                f.write(f"    OCR: {p.get('ocr_text')[:100]}\n")
            if p.get('path_or_uri'):
                f.write(f"    Path: {p.get('path_or_uri')}\n")

if __name__ == "__main__":
    asyncio.run(debug())
