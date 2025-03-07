import torch
import torch.optim as optim
from torchvision.utils import save_image
import os
from tqdm import tqdm

from imaginaire.losses.sds import SDSLoss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SDSLoss
sds = SDSLoss(device, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base", guidance_scale=100)

# Create a random image (batch_size=1, channels=3, height=64, width=64)
random_image = torch.rand((1, 3, 64, 64), device=device) * 2 - 1  # Scale to [-1, 1]
random_image.requires_grad_(True)  # Make the image trainable

# Define a text prompt
text_prompt = ["Bob the Builder"]
negative_text_prompt = ["Low quality"]

# Get text embeddings
cond_embeddings = sds.get_text_embeds(text_prompt)
uncond_embeddings = sds.get_text_embeds(negative_text_prompt)
text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

# Set up optimizer 
optimizer = optim.Adam([random_image], lr=0.1)  

# Create an output directory to save images
output_dir = "sds_test"
os.makedirs(output_dir, exist_ok=True)

# Define the path to the log file
log_file_path = os.path.join(output_dir, "sds_loss.txt")

# Training loop
num_iterations = 100001
for iteration in tqdm(range(1, num_iterations), desc="Training"):
    # Zero the gradients
    optimizer.zero_grad()

    # Compute SDS loss
    loss = sds(random_image, text_embeddings)

    # Backpropagate the loss
    loss.backward()

    # Update the image
    optimizer.step()

    # Open the log file in append mode
    with open(log_file_path, 'a') as f:
        # Print loss every 100 iterations
        if iteration % 100 == 0:
            f.write(f"Iteration {iteration}: SDS Loss = {loss.item()}\n")

    # Save the image every 200 iterations
    if iteration % 200 == 0:
        with torch.no_grad():
            # Clamp the image to [-1, 1] and convert to [0, 1] for saving
            image_to_save = (random_image.clamp(-1, 1) + 1) / 2
            save_path = os.path.join(output_dir, f"image_iter_{iteration}.png")
            save_image(image_to_save, save_path)

# Final image
with torch.no_grad():
    final_image = (random_image.clamp(-1, 1) + 1) / 2
    final_save_path = os.path.join(output_dir, "final_image.png")
    save_image(final_image, final_save_path)  