import torch
import os

def copy_tree_assets(checkpoint_path, asset_path, output_path):
    """
    Loads a training checkpoint and the scenedreamer_released.pt file,
    adds the 'assets' from the released file to the checkpoint, and
    saves the modified checkpoint.

    Args:
        checkpoint_path (str): Path to the training checkpoint (e.g., 'latest_checkpoint.pt').
        asset_path (str): Path to the 'scenedreamer_released.pt' file.
        output_path (str): Path to save the modified checkpoint.
    """

    # Load the training checkpoint.
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Load the assets from scenedreamer_released.pt.
    try:
        asset_data = torch.load(asset_path, map_location='cpu')
        if 'assets' not in asset_data:
            print(f"Error: 'assets' key not found in {asset_path}")
            return
        assets = asset_data['assets']
    except FileNotFoundError:
        print(f"Error: Asset file not found at {asset_path}")
        return
    except Exception as e:
        print(f"Error loading assets: {e}")
        return

    # Add the assets to the checkpoint dictionary.
    checkpoint['assets'] = assets

    # Save the modified checkpoint.
    try:
        torch.save(checkpoint, output_path)
        print(f"Successfully saved modified checkpoint to {output_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

if __name__ == '__main__':
    checkpoint_path = "./outputs/lucidscenedreamer_train/latest_checkpoint.pt"  
    asset_path = "./scenedreamer_models/scenedreamer_released.pt" 
    output_path = "./outputs/lucidscenedreamer_train/checkpoints/modified_checkpoint.pt"  

    copy_tree_assets(checkpoint_path, asset_path, output_path)
    os.system("python inference.py --config configs/scenedreamer_inference.yaml --output_dir ./outputs/lucidscenedreamer_train/inference/ --seed 8888 --checkpoint ./outputs/lucidscenedreamer_train/checkpoints/modified_checkpoint.pt")
