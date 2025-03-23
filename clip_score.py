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

# Initialize a list to store scores
scores = []

# Path to the images folder
image_folder = "./images"

# Loop through all images in the folder
for i in range(40):
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
        logits_per_image, _ = model(image_input, text_input)
        score = logits_per_image.item()
        scores.append(score)

    print(f"Processed {image_filename}")

# Convert raw logits to probabilities using softmax
probabilities = torch.softmax(torch.tensor(scores), dim=0)

# Calculate the average probability
average_probability = probabilities.mean().item()

# Save the result to a text file
with open("Cubism.txt", "w") as f:
    f.write(f"Average CLIP Score for '{text}': {average_probability}\n")

print("Results saved")