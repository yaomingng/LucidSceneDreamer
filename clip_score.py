import torch
import open_clip
from PIL import Image
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B/32")
model.to(device)

# Path to images folder
image_folder = "./images"
image_filenames = [f"{i:05d}.png" for i in range(40)]  

# Define text prompt
text_prompt = ["Low-poly dreamlike valley"]
text_tokens = tokenizer(text_prompt).to(device)

# Compute CLIP scores for each image
clip_scores = []

for filename in image_filenames:
    image_path = os.path.join(image_folder, filename)
    
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Get CLIP features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity (CLIP score)
        score = (image_features @ text_features.T).item()
        clip_scores.append(score)

# Compute average CLIP score
average_clip_score = sum(clip_scores) / len(clip_scores)
print(f"Average CLIP Score for the scene: {average_clip_score:.4f}")
