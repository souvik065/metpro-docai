from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

def test_fix():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        print(f"Original return type: {type(outputs)}")
        
        # Robust logic
        if not isinstance(outputs, torch.Tensor):
            feats = getattr(outputs, "image_embeds", None)
            if feats is None:
                feats = getattr(outputs, "pooler_output", outputs)
        else:
            feats = outputs
            
        print(f"Resolved feats type: {type(feats)}")
        if hasattr(feats, "shape"):
            print(f"Shape: {feats.shape}")
            # Normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
            print(f"✓ Success! First vector sum: {feats[0].sum():.4f}")
        else:
            print("✗ Still no shape attribute!")

if __name__ == "__main__":
    test_fix()
