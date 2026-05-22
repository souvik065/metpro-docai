from qdrant_client import QdrantClient
from config.settings import settings
import os

def check_qdrant():
    path = "data/qdrant"
    print(f"Checking Qdrant at: {path}")
    if not os.path.exists(path):
        print("Path does not exist!")
        return
        
    client = QdrantClient(path=path)
    collections = client.get_collections().collections
    print(f"\nCollections ({len(collections)}):")
    for c in collections:
        count = client.count(collection_name=c.name).count
        print(f"  - {c.name}: {count} points")
        
        # Check vectors config
        coll_info = client.get_collection(collection_name=c.name)
        print(f"    Vectors: {coll_info.config.params.vectors}")

if __name__ == "__main__":
    check_qdrant()
