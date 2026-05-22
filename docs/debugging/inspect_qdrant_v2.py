from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config.settings import settings
import os

def inspect():
    path = "data/qdrant"
    client = QdrantClient(path=path)
    
    image_filter = Filter(must=[FieldCondition(key="asset_type", match=MatchValue(value="image"))])
    
    # Scroll to get image points with vectors
    res = client.scroll(
        collection_name="macpro_medical",
        scroll_filter=image_filter,
        limit=3,
        with_vectors=True,
        with_payload=True
    )
    
    points = res[0]
    print(f"Found {len(points)} images")
    
    for p in points:
        print(f"\nAsset ID: {p.payload.get('asset_id')}")
        print(f"File: {p.payload.get('filename')}")
        
        image_vec = p.vector.get("image")
        if image_vec:
            # Check if all zeros
            is_zero = all(v == 0.0 for v in image_vec)
            print(f"Image Vector: {'ALL ZEROS' if is_zero else 'NON-ZERO'}")
            print(f"  First 10: {image_vec[:10]}")
            print(f"  Sum: {sum(image_vec):.6f}")
        else:
            print("Image Vector: REMOVED/MISSING")

if __name__ == "__main__":
    inspect()
