import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
from tqdm import tqdm

# ==========================================
# 1. The Energy Function (A simple CNN)
# ==========================================
# This network's only job is to output a single scalar "Energy" score.
# Low score = Real Lung. High score = Fake/Noise.
class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1) # Output a single number
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Langevin Dynamics (The Thermodynamic Generator)
# ==========================================
def sample_langevin(model, x, steps=60, step_size=10, noise_scale=0.005):
    # This is Markov Chain Monte Carlo (MCMC)
    # Detach x to start a fresh computational graph
    x = x.clone().detach().requires_grad_(True)
    
    for _ in range(steps):
        # Calculate the energy of the current image
        energy = model(x)

        # Sum the energy so autograd can compute gradients for the whole batch at once
        grad = torch.autograd.grad(energy.sum(), x, only_inputs=True)[0]
        
        # Langevin Equation: 
        # Move pixels in the OPPOSITE direction of the gradient (to lower the energy)
        # Add a tiny bit of random thermodynamic heat (noise) to prevent getting stuck
        x.data -= step_size * grad + noise_scale * torch.randn_like(x)
        
        # Clamp pixels to stay within valid grayscale image bounds [-1, 1]
        x.data = torch.clamp(x.data, -1.0, 1.0)
        
    return x.detach() # Strip the autograd receipt so it doesn't cause an OOM!

# ==========================================
# 3. The Training Loop
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting Thermodynamic EBM on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    out_dir = os.path.join(dataset_root, "EBM_Outputs")
    os.makedirs(out_dir, exist_ok=True)

    # We scale to [-1, 1] for stable gradient flows in Langevin dynamics
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("Loading Normal Lungs...")
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    full_dataset = DataClass(split='train', transform=transform, download=False, size=224, root=dataset_root)
    
    # Isolate healthy lungs
    normal_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1][0] == 0]
    normal_dataset = Subset(full_dataset, normal_indices)
    dataloader = DataLoader(normal_dataset, batch_size=32, shuffle=True, num_workers=0)

    model = EnergyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100
    print("Commencing Energy Optimization...")

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(dataloader, leave=True)
        
        for real_images, _ in loop:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # 1. Start with pure random static
            initial_noise = torch.rand_like(real_images) * 2 - 1 
            
            # 2. Cool the static down into fake lungs via Langevin Dynamics
            fake_images = sample_langevin(model, initial_noise, steps=60)
            
            optimizer.zero_grad()
            
            # 3. Calculate Energy for both Real and Fake
            real_energy = model(real_images)
            fake_energy = model(fake_images)
            
            # 4. Contrastive Divergence Loss
            # Real Energy low (negative), Fake Energy high (positive)
            loss = real_energy.mean() - fake_energy.mean()
            
            # Add L2 Regularization (Prevents the energy values from exploding to infinity)
            loss += 0.001 * (real_energy ** 2 + fake_energy ** 2).mean()
            
            loss.backward()
            
            # Gradient clipping to prevent thermodynamic explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loop.set_description(f"EBM Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(Loss=loss.item(), Real_E=real_energy.mean().item(), Fake_E=fake_energy.mean().item())

        # Save visual progression
        if (epoch + 1) % 10 == 0:
            model.eval()
            print(f"\nGenerating checkpoint samples for Epoch {epoch+1}...")
            with torch.no_grad():
                # Use gradients to sample, enable grad temporarily
                with torch.enable_grad():
                    test_noise = (torch.rand(16, 1, 224, 224, device=device) * 2 - 1)
                    test_samples = sample_langevin(model, test_noise, steps=100)
                
                # Denormalize from [-1, 1] back to [0, 1] for saving
                test_samples = (test_samples + 1) / 2.0
                save_image(test_samples, os.path.join(out_dir, f'ebm_sample_{epoch+1}.png'), nrow=4)

    torch.save(model.state_dict(), os.path.join(out_dir, 'ebm_baseline.pth'))
    print("\nEBM Training Complete.")

if __name__ == '__main__':
    main()