import sys
import numpy as np
import json
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python main.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")

fclip = FashionCLIP('fashion-clip')

# Encode image
image_embedding = fclip.encode_images([img], batch_size=1)[0]
image_embedding /= np.linalg.norm(image_embedding)

# Define label sets
attributes = {
    "category": ["shirt", "pants", "shoes", "jacket", "dress"],
    "color": ["red", "blue", "black", "white", "green"],
    "material": ["cotton", "leather", "denim", "wool"],
    "style": ["casual", "formal", "sporty", "vintage"],
    "season": ["summer", "winter", "spring", "autumn"],
    "occasion": ["party", "work", "casual", "outdoor"],
    "body group": ["upper body", "lower body", "footwear", "accessories"]
}

# Predict best label for each attribute
results = {}
for attr, prompts in attributes.items():
    text_embeds = fclip.encode_text(prompts, batch_size=1)
    text_embeds /= np.linalg.norm(text_embeds, axis=1)[:, None]
    
    scores = image_embedding @ text_embeds.T
    best_idx = int(np.argmax(scores))
    results[attr] = prompts[best_idx]

# Add embedding and image metadata
results["fileName"] = img_path
results["embedding"] = image_embedding.tolist()

# Output full result as JSON
print(json.dumps(results, indent=2))
