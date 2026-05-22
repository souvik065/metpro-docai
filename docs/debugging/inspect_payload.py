from qdrant_client import QdrantClient
from config.settings import settings

def inspect():
    client = QdrantClient(path="data/qdrant")
    res = client.scroll(collection_name="macpro_medical", limit=5, with_payload=True)
    points = res[0]
    
    print(f"Total points fetched: {len(points)}")
    for p in points:
        print(f"\nPoint ID: {p.id}")
        print(f"Payload keys: {list(p.payload.keys())}")
        print(f"Payload: {p.payload}")

if __name__ == "__main__":
    inspect()
