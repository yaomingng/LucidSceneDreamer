import torch
import clip
from PIL import Image
import os

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the text prompt
text = "Cubism"
text_input = clip.tokenize([text]).to(device)

# Initialize score
total_score = 0

# Path to the images folder
image_folder = "./images"

# Define number of images
num_imgs = 40

# Loop through all images in the folder
for i in range(num_imgs):
    # Construct the image filename
    image_filename = f"{i:05d}.png"
    image_path = os.path.join(image_folder, image_filename)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Calculate CLIP score
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        total_score += similarity

    print(f"Similarity between the {image_filename} and text prompt: {similarity}")

avg_score = total_score / num_imgs

# Save the result to a text file
with open("Cubism.txt", "w") as f:
    f.write(f"Average CLIP Score for '{text}': {avg_score}\n")

print("Results saved")