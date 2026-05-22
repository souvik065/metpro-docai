from transformers import CLIPModel, CLIPProcessor
import torch

def inspect_clip():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    print(f"CLIPModel type: {type(model)}")
    print("\nAvailable methods/attributes (filtered):")
    for attr in dir(model):
        if "image" in attr.lower() or "feature" in attr.lower() or "encode" in attr.lower():
            print(f"  - {attr}")

if __name__ == "__main__":
    inspect_clip()
