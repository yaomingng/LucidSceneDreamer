import torch
import torch.optim as optim
from torchvision.utils import save_image
import os
from tqdm import tqdm

from imaginaire.losses.sds import SDSLoss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SDSLoss
sds = SDSLoss(device, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base", guidance_scale=200)

# Create a random image (batch_size=1, channels=3, height=64, width=64)
random_image = torch.rand((1, 4, 64, 64), device=device) 
random_image.requires_grad_(True)  # Make the image trainable

# Define a text prompt
text_prompt = ["Bob the builder"]
negative_text_prompt = ["Low quality"]

# Get text embeddings
cond_embeddings = sds.get_text_embeds(text_prompt)
uncond_embeddings = sds.get_text_embeds(negative_text_prompt)
text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

# Set up optimizer 
optimizer = optim.Adam([random_image], lr=0.01)  

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# Create an output directory to save images
output_dir = "sds_test"
os.makedirs(output_dir, exist_ok=True)

# Define the path to the log file
log_file_path = os.path.join(output_dir, "sds_loss.txt")

# Training loop
num_iterations = 501
for iteration in tqdm(range(1, num_iterations), desc="Training"):
    # Zero the gradients
    optimizer.zero_grad()

    # Compute SDS loss
    loss = sds.get_sds_loss(random_image, text_embeddings)

    # Backpropagate the loss
    loss.backward()

    # Update the image
    optimizer.step()
    scheduler.step()

    with open(log_file_path, 'a') as f:
        # Write loss every 50 iterations
        if iteration % 50 == 0:
            f.write(f"Iteration {iteration}: SDS Loss = {loss.item()}\n")

    # Save the image every 20 iterations
    if iteration % 20 == 0:
        with torch.no_grad():
            image_to_save = sds.decode_latents(random_image)
            save_path = os.path.join(output_dir, f"image_iter_{iteration}.png")
            save_image(image_to_save, save_path)
