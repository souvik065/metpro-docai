from qdrant_client import QdrantClient
from config.settings import settings

def inspect():
    client = QdrantClient(path="data/qdrant")
    res = client.scroll(collection_name="macpro_medical", limit=5, with_payload=True)
    points = res[0]
    
    for p in points:
        atype = p.payload.get('asset_type')
        print(f"Point ID: {p.id}, asset_type: '{atype}'")

if __name__ == "__main__":
    inspect()
