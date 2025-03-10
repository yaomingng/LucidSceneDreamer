import torch
import torch.optim as optim
import torchvision
from tqdm import tqdm
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time
import importlib
import random

# Import necessary components 
from imaginaire.config import Config
from imaginaire.generators.lucidscenedreamer import Generator as LucidSceneDreamerGenerator
from imaginaire.losses.sds import SDSLoss
from imaginaire.utils.trainer import set_random_seed
from imaginaire.utils.cudnn import init_cudnn

# --- Configuration ---
config_file = './configs/lucidscenedreamer_train.yaml'  # Path to config file
cfg = Config(config_file)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Seed Initialization ---
if cfg.randomized_seed:
    seed = random.randint(0, 10000)
    set_random_seed(seed)
else:
    set_random_seed(cfg.seed)

# --- Initialize cudnn ---
init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

# --- Model Initialization ---
# Load Pretrained SceneDreamer
lib_G = importlib.import_module(cfg.gen.type)
net_G = lib_G.Generator(cfg.gen).to(device)
# Put the generator in training mode.
net_G.train()

checkpoint = torch.load(cfg.pretrained_model, map_location='cpu')
state_dict = checkpoint['net_G']

# Remove the 'module.' prefix if present
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key[7:] if key.startswith('module.') else key  
    new_state_dict[new_key] = value

net_G.load_state_dict(new_state_dict)

# Initialize SDS Loss
sds = SDSLoss(device, pretrained_model_name_or_path=cfg.trainer.sds.pretrained_model_name_or_path,
                      guidance_scale=cfg.trainer.sds.guidance_scale)

# --- Training Parameters ---
num_iterations = cfg.max_iter             # Number of iterations to train for
save_interval = cfg.snapshot_save_iter    # Save every 'save_interval' iterations
log_interval = cfg.logging_iter           # Print loss every 'log_interval' iterations
image_save_interval = cfg.image_save_iter # save image every 'image_save_interval' iterations
output_dir = cfg.outputdir
os.makedirs(output_dir, exist_ok=True)    # create output directory
starting_iter = 1                         # for resuming
# Create the 'checkpoints' subdirectory if it doesn't exist
checkpoint_path = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_path, exist_ok=True)  

# Optimizer Setup
# --- Parameter Freezing ---
for name, param in net_G.named_parameters():
    if (
        'hash_encoder' in name or 
        'render_net' in name or 
        'sky_net' in name or 
        'style_net' in name or
        'denoiser' in name
    ):
        param.requires_grad = True  # These we want to optimize
        print(f"Parameters of {name} will be optimized.")
    else:
        param.requires_grad = False  # Freeze all other parameters
        print(f"Parameters of {name} will be frozen.")

params_to_optimize = [
    {'params': net_G.hash_encoder.parameters(), 'lr': cfg.gen_opt.param_groups['hash_encoder']['lr']},  
    {'params': net_G.render_net.parameters(), 'lr': cfg.gen_opt.param_groups['render_net']['lr']},  
    {'params': net_G.sky_net.parameters(), 'lr': cfg.gen_opt.param_groups['sky_net']['lr']},  
    {'params': net_G.style_net.parameters(), 'lr': cfg.gen_opt.param_groups['style_net']['lr']},
    {'params': net_G.denoiser.parameters(), 'lr': cfg.gen_opt.param_groups['denoiser']['lr']}
]

# Initialize the optimizer.
optimizer = optim.Adam(params_to_optimize) 

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

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

# Get text embeddings
cond_embeddings = sds.get_text_embeds(cfg.prompt)
uncond_embeddings = sds.get_text_embeds(cfg.negative_prompt)
text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

# --- Helper function to save images ---
def save_image(image, output_dir, iteration):
    """Saves a generated image."""
    image = (image + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    image = image.clamp(0, 1)  # Ensure values are within valid range

    # Create the 'generated_images' subdirectory if it doesn't exist
    images_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)  

    filepath = os.path.join(images_dir, f"image_{iteration}.png")
    torchvision.utils.save_image(image, filepath)

# --- Training Loop ---
for iteration in tqdm(range(starting_iter, num_iterations), desc="Training"):
    start_time = time.time()
    optimizer.zero_grad()

    # 1. Sample Camera and Render
    image = net_G()['fake_images'] 

    # 2. Compute SDS Loss
    loss = sds(image, text_embeddings)

    # 3. Backpropagation and Optimization
    loss.backward() 
    optimizer.step()  
    scheduler.step()

    end_time = time.time()

    # --- Logging and Checkpointing ---
    if iteration % log_interval == 0:
        log_message = f"Iteration {iteration}: SDS Loss = {loss.item()}, Time = {end_time - start_time:.4f}s"
        
        # Save log to a text file
        log_file_path = os.path.join(output_dir, "training_log.txt")
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message + "\n")  # Add a newline for each entry

    # Save a checkpoint every 100 iteration to avoid losing progress
    if iteration % 100 == 0:
        latest_checkpoint_path = os.path.join(output_dir, "latest_checkpoint.pt")

        # Delete the old checkpoint file if it exists
        if os.path.exists(latest_checkpoint_path):
            os.remove(latest_checkpoint_path)

        torch.save({
            'net_G': net_G.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration 
        }, latest_checkpoint_path)

    if iteration % save_interval == 0 and iteration > 0:
        # Define the output file path
        output_path = os.path.join(checkpoint_path, f"checkpoint_{iteration}.pt")
        
        torch.save({
            'net_G': net_G.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration 
        }, output_path)

    # --- Image Saving ---
    if iteration % image_save_interval == 0:
        save_image(image, output_dir, iteration)

    torch.cuda.empty_cache()

print("Training Done!")

os.system("python ./scripts/copy_tree_assets.py")