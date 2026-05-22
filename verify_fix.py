import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.indexing.vector_store import get_vector_store
from src.embeddings.models import get_text_embedder
from config.settings import settings

def verify():
    print("Initializing embedder...")
    embedder = get_text_embedder()
    store = get_vector_store()
    
    query = "What is acute watery diarrhea?"
    print(f"Embedding query: {query}")
    vector = embedder.embed(query)
    
    print("Searching...")
    try:
        results = store.search_text(vector, top_k=2)
        print(f"Found {len(results)} results.")
        for i, res in enumerate(results):
            print(f"[{i}] Score: {res['score']}")
            print(f"    Payload: {str(res['payload'])[:100]}...")
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
