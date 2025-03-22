import torch
import clip
import os
import numpy as np
from PIL import Image

# Load the model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the folder containing images
image_folder = "./images"

# Load all images
image_paths = [os.path.join(image_folder, f"{i:05d}.png") for i in range(40)]
images = [Image.open(img_path) for img_path in image_paths]

# Preprocess images
images_preprocessed = torch.stack([preprocess(img) for img in images]).to(device)

# Define your text prompt
text_prompt = "Cubism"

# Tokenize the text
text_tokens = clip.tokenize([text_prompt]).to(device)

with torch.no_grad():
    # Encode images and text
    image_features = model.encode_image(images_preprocessed)
    text_features = model.encode_text(text_tokens)
    
    # Compute similarity scores
    logits_per_image = image_features @ text_features.T
    scores = logits_per_image.softmax(dim=-1).cpu().numpy()
    
# Save results to a text file
output_file = "clip_similarity_scores.txt"
with open(output_file, "w") as f:
    f.write("Similarity scores for each image:\n")
    f.write(np.array2string(scores, precision=4, separator=", ") + "\n")
    
    # Average the scores
    average_score = np.mean(scores)
    f.write(f"\n3D Scene Score: {average_score:.4f}\n")

print(f"Results saved to {output_file}")