import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
import os
from tqdm import tqdm
import torch.nn as nn

from imaginaire.losses.sds import SDSLoss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SDSLoss
sds = SDSLoss(device, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base", guidance_scale=100)

# Create a random image (batch_size=1, channels=3, height=64, width=64)
random_image = nn.Parameter(
            torch.randn(
                1, 3, 100, 100, 
                device=device, 
                dtype=torch.float32,
            )
        ) * 2 - 1 

random_image = random_image.clone().detach().requires_grad_(True)

# Define a text prompt
text_prompt = ["Cat lying on a table"]
negative_text_prompt = ["Low quality"]

# Get text embeddings
cond_embeddings = sds.get_text_embeds(text_prompt)
uncond_embeddings = sds.get_text_embeds(negative_text_prompt)
text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

# Set up optimizer 
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)

# Initialize the optimizer.
optimizer = torch.optim.AdamW([random_image], lr=0.1, weight_decay=0)

# Learning Rate Scheduler
num_iterations = 10001
scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int((num_iterations-1)*1.5))

# Create an output directory to save images
output_dir = "sds_test"
os.makedirs(output_dir, exist_ok=True)

# Define the path to the log file
log_file_path = os.path.join(output_dir, "sds_loss.txt")

# Training loop
for iteration in tqdm(range(1, num_iterations), desc="Training"):
    # Zero the gradients
    optimizer.zero_grad()

    # Compute SDS loss
    loss = sds(random_image, text_embeddings)

    # Backpropagate the loss
    (2000 * loss).backward()

    # Update the image
    optimizer.step()

    # Open the log file in append mode
    with open(log_file_path, 'a') as f:
        # Print loss every 100 iterations
        if iteration % 500 == 0:
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