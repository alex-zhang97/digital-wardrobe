import sys
import json
import numpy as np
import requests
from PIL import Image
from fashion_clip.fashion_clip import FashionCLIP

API_URL = "http://localhost:3000/clothing"  # Your Prisma API endpoint

def load_attributes(file_path=None):
    if file_path:
        with open(file_path, "r") as f:
            return json.load(f)
    return {
        "category": ["shirt", "pants", "shoes", "jacket", "dress"],
        "color": ["red", "blue", "black", "white", "green"],
        "material": ["cotton", "leather", "denim", "wool"],
        "style": ["casual", "formal", "sporty", "vintage"],
        "season": ["summer", "winter", "spring", "autumn"],
        "occasion": ["party", "work", "casual", "outdoor"],
        "bodyGroup": ["upper body", "lower body", "footwear", "accessories"]
    }

def normalize_vector(vec):
    if vec.ndim == 1:
        return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1)[:, None]

def predict_attributes(img, attributes, fclip):
    results = {}
    image_embedding = normalize_vector(fclip.encode_images([img], batch_size=1)[0])
    for attr, prompts in attributes.items():
        text_embeds = normalize_vector(fclip.encode_text(prompts, batch_size=1))
        scores = image_embedding @ text_embeds.T
        best_idx = int(np.argmax(scores))
        results[attr] = prompts[best_idx]
    results["embedding"] = image_embedding.tolist()
    return results

def send_to_prisma(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        print("Inserted into Prisma:", data["fileName"])
    else:
        print("Failed to insert:", response.text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tag_image.py <image_path> [attributes.json]")
        sys.exit(1)

    img_path = sys.argv[1]
    attr_file = sys.argv[2] if len(sys.argv) > 2 else None
    attributes = load_attributes(attr_file)

    fclip = FashionCLIP('fashion-clip')
    img = Image.open(img_path).convert("RGB")
    results = predict_attributes(img, attributes, fclip)
    results["fileName"] = img_path

    send_to_prisma(results)

if __name__ == "__main__":
    main()
