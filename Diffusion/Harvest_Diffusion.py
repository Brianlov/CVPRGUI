import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from Diffusion_Architecture import UNet
from Sample_Diffusion import DiffusionSampler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting Mass Diffusion Harvest on: {device}")

    # Set up the folder structure for the Hybrid CNN to read later
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    synthetic_dir = os.path.join(dataset_root, "Diffusion_Synthetic", "Normal_0")
    os.makedirs(synthetic_dir, exist_ok=True)

    # Load the U-Net
    model = UNet().to(device)
    weight_path = os.path.join(dataset_root, "Diffusion_Weights", "unet_epoch_100.pth")
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    sampler = DiffusionSampler(num_timesteps=1000, device=device)

    # Harvesting Parameters
    num_images_needed = 2600
    batch_size = 50 
    batches = num_images_needed // batch_size

    print(f"Commencing deep thermodynamic sampling for {num_images_needed} images...")
    
    img_counter = 0
    # Loop through the batches
    for i in range(batches):
        print(f"\nProcessing Batch {i+1}/{batches}...")
        # This is where the heavy lifting happens
        generated_batch = sampler.sample(model, num_images=batch_size, img_size=224)
        
        # Save them out one by one
        for j in range(generated_batch.size(0)):
            save_path = os.path.join(synthetic_dir, f"diff_normal_{img_counter}.png")
            save_image(generated_batch[j], save_path, normalize=True)
            img_counter += 1

    print(f"\nHarvest Complete! {img_counter} diffusion X-rays saved to:")
    print(synthetic_dir)

if __name__ == '__main__':
    main()