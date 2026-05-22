from qdrant_client import QdrantClient
from config.settings import settings

def inspect():
    client = QdrantClient(path="data/qdrant")
    
    # Scroll through all points
    offset = None
    all_types = set()
    total_count = 0
    
    while True:
        res = client.scroll(
            collection_name="macpro_medical",
            limit=100,
            with_payload=True,
            offset=offset
        )
        points, offset = res
        for p in points:
            atype = p.payload.get('asset_type')
            all_types.add(atype)
            total_count += 1
        
        if offset is None:
            break
            
    print(f"Total points: {total_count}")
    print(f"Unique asset types: {all_types}")

if __name__ == "__main__":
    inspect()
