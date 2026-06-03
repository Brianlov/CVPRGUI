import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

# 1. Rebuild the exact VAE Architecture to load the weights
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.enc1 = nn.Conv2d(1, 32, 4, 2, 1)    
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)   
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1)  
        self.enc4 = nn.Conv2d(128, 256, 4, 2, 1) 
        self.enc5 = nn.Conv2d(256, 512, 4, 2, 1) 
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.dec1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec5 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 512, 7, 7)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = torch.sigmoid(self.dec5(x)) 
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device for VAE harvesting: {device}")

    # Set up NVMe paths
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    output_dir = os.path.join(dataset_root, "VAE_Synthetic", "Normal_0")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the trained brain
    print("Loading VAE weights...")
    model = VAE(latent_dim=128).to(device)
    weight_path = os.path.join(dataset_root, "vae_baseline.pth")
    
    if not os.path.exists(weight_path):
        print(f"Error: Weights not found at {weight_path}")
        return

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Harvest Parameters
    total_images_needed = 2600
    batch_size = 64
    latent_dim = 128
    generated_count = 0

    print(f"Sampling {total_images_needed} images from the VAE latent space...")

    with torch.no_grad():
        with tqdm(total=total_images_needed, desc="Decoding Images") as pbar:
            while generated_count < total_images_needed:
                current_batch_size = min(batch_size, total_images_needed - generated_count)
                
                # Sample random coordinates from the standard normal distribution
                z = torch.randn(current_batch_size, latent_dim).to(device)
                
                # Push through the Decoder (no need to un-normalize, Sigmoid handled it)
                fake_images = model.decode(z)
                
                # Save out
                for i in range(current_batch_size):
                    img_path = os.path.join(output_dir, f"vae_normal_{generated_count}.png")
                    save_image(fake_images[i], img_path)
                    generated_count += 1
                    
                pbar.update(current_batch_size)

    print(f"\nVAE Harvest complete. Images stored in: {output_dir}")

if __name__ == "__main__":
    main()