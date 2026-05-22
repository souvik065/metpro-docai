import os
import sys

# Set environment variables for the download phase to make sure they match
os.environ["HF_HOME"] = "/usr/share/model-cache/huggingface"
os.environ["EASYOCR_MODULE_PATH"] = "/usr/share/model-cache/easyocr"

try:
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPProcessor, CLIPModel
    import easyocr
except ImportError as e:
    print(f"Error importing libraries: {e}")
    sys.exit(1)

print("Downloading sentence-transformers model (all-MiniLM-L6-v2)...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Downloading CLIP model (openai/clip-vit-base-patch32)...")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Downloading EasyOCR model (english)...")
easyocr.Reader(["en"], gpu=False)

print("All models successfully cached!")
