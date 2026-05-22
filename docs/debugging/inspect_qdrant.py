import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.indexing.vector_store import get_vector_store
from config.settings import settings

async def inspect():
    vs = get_vector_store()
    client = vs._get_client()
    
    print(f"Collection: {settings.qdrant_collection}")
    
    # Scroll to find image points
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    image_filter = Filter(must=[FieldCondition(key="asset_type", match=MatchValue(value="image"))])
    
    points, next_page = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=image_filter,
        limit=5,
        with_vectors=True,
        with_payload=True
    )
    
    print(f"\nFound {len(points)} image points via scroll:")
    for p in points:
        print(f"\nPoint ID: {p.id}")
        payload = p.payload
        print(f"  Asset Type: {payload.get('asset_type')}")
        print(f"  Filename: {payload.get('filename')}")
        
        vectors = p.vector
        if isinstance(vectors, dict):
            text_vec = vectors.get("text")
            image_vec = vectors.get("image")
            print(f"  Text Vector: {'Present' if text_vec else 'Missing'} (Sum: {sum(text_vec) if text_vec else 0:.4f})")
            print(f"  Image Vector: {'Present' if image_vec else 'Missing'} (Sum: {sum(image_vec) if image_vec else 0:.4f})")
            if image_vec:
                print(f"    First 5 Image dims: {image_vec[:5]}")
        else:
            print(f"  Vectors: {type(vectors)} (not named?)")

if __name__ == "__main__":
    asyncio.run(inspect())
