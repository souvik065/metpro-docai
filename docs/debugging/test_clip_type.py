from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

def test_type():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        print(f"Return type: {type(feats)}")
        if hasattr(feats, "shape"):
            print(f"Shape: {feats.shape}")
        else:
            print("No 'shape' attribute!")
            print(f"Attributes: {dir(feats)}")

if __name__ == "__main__":
    test_type()
