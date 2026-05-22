import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

def test_clip():
    print("Testing CLIP loading...")
    model_id = "openai/clip-vit-base-patch32"
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        print("✓ Models loaded.")
        
        # Dummy image
        img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            print(f"✓ Embedding success! Shape: {feats.shape}")
            print(f"  Sample: {feats[0][:5].tolist()}")
    except Exception as e:
        print(f"✗ CLIP failed: {e}")

if __name__ == "__main__":
    test_clip()
