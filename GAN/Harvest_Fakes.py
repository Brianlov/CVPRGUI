import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

# Import the blueprint
from GAN_Architecture import Generator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Harvesting synthetic data on: {device}")

    # Set up the exact folder structure needed for easy PyTorch loading later
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    synthetic_dir = os.path.join(dataset_root, "Synthetic_Data", "Normal_0")
    os.makedirs(synthetic_dir, exist_ok=True)

    # 1. Load the frozen brain
    print("Waking up the trained Generator...")
    latent_dim = 100
    generator = Generator(latent_dim=latent_dim).to(device)
    
    weights_path = os.path.join(dataset_root, 'generator_weights.pth')
    generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    generator.eval() # CRITICAL: Lock the gradients

    # 2. Harvesting Parameters
    num_images_needed = 2600
    batch_size = 100
    batches = num_images_needed // batch_size

    print(f"Generating {num_images_needed} high-resolution synthetic X-rays...")
    
    img_counter = 0
    with torch.no_grad():
        for i in tqdm(range(batches), desc="Harvesting Batches"):
            # Generate pure noise
            z = torch.randn(batch_size, latent_dim, device=device)
            
            # Pass noise through the GAN to hallucinate the images
            fake_imgs = generator(z)
            
            # Save each image individually to the NVMe drive
            for j in range(fake_imgs.size(0)):
                # Un-normalize from [-1, 1] back to standard [0, 1] pixel values
                save_path = os.path.join(synthetic_dir, f"synthetic_normal_{img_counter}.png")
                save_image(fake_imgs[j], save_path, normalize=True)
                img_counter += 1

    print(f"\nHarvest Complete! {img_counter} synthetic healthy lungs saved to:")
    print(synthetic_dir)

if __name__ == '__main__':
    main()