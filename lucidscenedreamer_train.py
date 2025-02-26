import torch
import torch.optim as optim
import torchvision
from tqdm import tqdm
import os
import time
import importlib
import random

# Import necessary components 
from imaginaire.config import Config
from imaginaire.generators.lucidscenedreamer import Generator as LucidSceneDreamerGenerator
from imaginaire.losses.sds import SDSLoss
from imaginaire.utils.trainer import set_random_seed

# --- Configuration ---
config_file = 'configs/lucidscenedreamer.yaml'  # Path to config file
cfg = Config(config_file)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if cfg.randomized_seed:
    seed = random.randint(0, 10000)
    set_random_seed(seed)
else:
    set_random_seed(cfg.seed)

# --- Model Initialization ---
# Load Pretrained SceneDreamer
lib_G = importlib.import_module(cfg.gen.type)
net_G = lib_G.Generator(cfg.gen).to(device)
# Put the generator in training mode.
net_G.train()

checkpoint = torch.load(cfg.pretrained_weight, map_location='cpu')
net_G.load_state_dict(checkpoint['net_G'])

# Initialize SDS Loss and Text Encoder
sds_loss_fn = SDSLoss(pretrained_model_name_or_path=cfg.trainer.sds.pretrained_model_name_or_path,
                      guidance_scale=cfg.trainer.sds.guidance_scale)

# Optimizer Setup
# --- Parameter Freezing ---
for name, param in net_G.named_parameters():
    if 'hash_encoder' in name or 'world_encoder' in name:
        param.requires_grad = True  # These we want to optimize
        print(f"Parameters of {name} will be optimized.")
    else:
        param.requires_grad = False  # Freeze all other parameters
        print(f"Parameters of {name} will be frozen.")

params_to_optimize = [
    {'params': net_G.world_encoder.parameters(), 'lr': cfg.gen_opt.param_groups['world_encoder']['lr']},  
    {'params': net_G.hash_encoder.parameters(), 'lr': cfg.gen_opt.param_groups['hash_encoder']['lr']},   
]

# Initialize the optimizer.
optimizer = optim.Adam(params_to_optimize) 

# --- Training Loop ---
num_iterations = cfg.max_iter             # Number of iterations to train for
save_interval = cfg.snapshot_save_iter    # Save every 'save_interval' iterations
log_interval = cfg.logging_iter           # Print loss every 'log_interval' iterations
image_save_interval = cfg.image_save_iter # save image every 'image_save_interval' iterations
output_dir = cfg.outputdir
os.makedirs(output_dir, exist_ok=True)    # create output directory
starting_iter = 0                         # for resuming

# --- Checkpoint Loading (for resuming) ---
if cfg.resume:
    checkpoint_path = os.path.join(cfg.outputdir, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net_G.load_state_dict(checkpoint['net_G'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_iter = checkpoint['iteration'] + 1 # Start from the next iteration
        print(f"Resuming training from iteration {starting_iter}")
    else:
        print('No checkpoint found, training from beginning')

# Get text prompt and encode it
text_prompt = cfg.prompt
text_embeddings = sds_loss_fn.get_text_embeddings([text_prompt], "")

# --- Training Loop ---
# --- Voxel Grid Initialization ---
if cfg.gen.pcg_cache:
    net_G.voxel.sample_world(device)  # For precomputed worlds
else:
    world_dir = os.path.join(output_dir, 'world')
    os.makedirs(world_dir, exist_ok=True)
    net_G.voxel.next_world(device, world_dir, checkpoint)  # For procedural generation

# --- Helper function to save images ---
def save_image(image, output_dir, iteration):
    """Saves a generated image."""
    image = (image + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    image = image.clamp(0, 1)  # Ensure values are within valid range

    # Create the 'generated_images' subdirectory if it doesn't exist
    images_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)  

    filepath = os.path.join(images_dir, f"image_{iteration:06d}.png")
    torchvision.utils.save_image(image, filepath)
    print(f"Saved image to {filepath}")

for iteration in tqdm(range(starting_iter, num_iterations), desc="Training"):

    start_time = time.time()

    # 1. Sample Camera and Render
    with torch.no_grad():
        image = net_G()['fake_images'] 

    # 2. Compute SDS Loss
    loss = sds_loss_fn(image, text_embeddings, 1)

    # 3. Backpropagation and Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = time.time()

    # --- Logging and Checkpointing ---
    if iteration % log_interval == 0:
        log_message = f"Iteration {iteration}: SDS Loss = {loss.item()}, Time = {end_time - start_time:.4f}s"
        tqdm.write(log_message)

        # Save log to a text file
        log_file_path = os.path.join(output_dir, "training_log.txt")
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message + "\n")  # Add a newline for each entry

    # Save a checkpoint every iteration (overwriting the previous one)
    latest_checkpoint_path = os.path.join(output_dir, "latest_checkpoint.pt")
    torch.save({
        'net_G': net_G.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration 
    }, latest_checkpoint_path)

    if iteration % save_interval == 0 and iteration > 0:
        output_path = os.path.join(output_dir, f"checkpoint_{iteration}.pt")

        # Create the 'checkpoints' subdirectory if it doesn't exist
        checkpoint_path = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)  
        
        torch.save({
            'net_G': net_G.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration 
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # --- Image Saving ---
    if iteration % image_save_interval == 0:
        save_image(image, output_dir, iteration)

print("Training Done!")