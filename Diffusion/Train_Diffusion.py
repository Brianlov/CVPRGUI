import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
from tqdm import tqdm

# Import the U-Net blueprint
from Diffusion_Architecture import UNet

# 1. The Noise Scheduler (Thermodynamics)
class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Betas dictate how much noise is added at each step (starts tiny, ends large)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        # Cumulative product of alphas determines the total noise level at any given step 't'
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_images, noise, timesteps):
        # Fetch the cumulative alpha for the specific timesteps
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
        
        # Match dimensions for broadcasting (Batch, Channels, Height, Width)
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        # The core thermodynamic equation: blend the original image with pure noise
        noisy_images = sqrt_alpha_cumprod * original_images + sqrt_one_minus_alpha_cumprod * noise
        return noisy_images

def main():
    # 2. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting Diffusion Engine on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    os.makedirs(os.path.join(dataset_root, "Diffusion_Weights"), exist_ok=True)

    # 3. Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 2e-4
    timesteps = 1000

    # 4. Strict Dataset Filtering (Normal Lungs Only)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    print("Isolating pure 'Normal (0)' lungs...")
    full_dataset = DataClass(split='train', transform=transform, download=False, size=224, root=dataset_root)
    normal_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1][0] == 0]
    normal_dataset = Subset(full_dataset, normal_indices)
    
    dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Loaded {len(normal_dataset)} baseline images.")

    # 5. Initialize Model and Scheduler
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LinearNoiseScheduler(num_timesteps=timesteps, device=device)
    loss_fn = nn.MSELoss() # Mean Squared Error (Guessing the exact noise pixels)

    # 6. The Entropy Loop
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(dataloader, leave=True)
        running_loss = 0.0
        
        for i, (images, _) in enumerate(loop):
            images = images.to(device)
            current_batch_size = images.shape[0]
            
            # Step A: Generate pure random noise (same shape as the images)
            noise = torch.randn_like(images).to(device)
            
            # Step B: Pick a random timestep 't' for every image in the batch
            t = torch.randint(0, timesteps, (current_batch_size,), device=device).long()
            
            # Step C: Corrupt the images to that specific timestep
            noisy_images = scheduler.add_noise(images, noise, t)
            
            # Step D: Force the U-Net to guess WHAT noise was just added
            optimizer.zero_grad()
            predicted_noise = model(noisy_images, t)
            
            # Step E: Calculate how wrong the guess was and update weights
            loss = loss_fn(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(MSE_Loss=loss.item())

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(dataset_root, "Diffusion_Weights", f"unet_epoch_{epoch+1}.pth"))

    print("\nDiffusion Training Complete! Final U-Net weights saved.")

if __name__ == '__main__':
    main()