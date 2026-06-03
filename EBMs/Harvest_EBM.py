import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

# 1. Rebuild the Exact EBM Architecture
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
            nn.Linear(512 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.net(x)

# 2. The Langevin Sampler (Requires gradients to simulate physics!)
def sample_langevin(model, x, steps=100, step_size=10, noise_scale=0.005):
    x = x.clone().detach().requires_grad_(True)
    
    # Enable gradients here, even if the main loop is evaluating
    with torch.enable_grad():
        for _ in range(steps):
            energy = model(x)
            grad = torch.autograd.grad(energy.sum(), x, only_inputs=True)[0]
            
            # Move down the energy slope, add thermodynamic noise
            x.data -= step_size * grad + noise_scale * torch.randn_like(x)
            x.data = torch.clamp(x.data, -1.0, 1.0)
            
    return x.detach()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device for EBM Harvest: {device}")

    # Windows path
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    output_dir = os.path.join(dataset_root, "EBM_Synthetic", "Normal_0")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load the Thermodynamic Brain
    print("Loading EBM weights...")
    model = EnergyModel().to(device)
    weight_path = os.path.join(dataset_root, "EBM_Outputs", "ebm_baseline.pth")
    
    if not os.path.exists(weight_path):
        print(f"Error: Weights not found at {weight_path}")
        return

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    # 4. Harvest Parameters
    total_images_needed = 2600
    batch_size = 64 # Pushing batch size up to maximize CUDA parallelization
    generated_count = 0

    print(f"Commencing Langevin thermodynamic sampling for {total_images_needed} lungs...")
    print("Warning: This will take significantly longer than GAN/VAE harvesting!")

    # No torch.no_grad() globally because Langevin dynamics require gradients
    with tqdm(total=total_images_needed, desc="Cooling Static") as pbar:
        while generated_count < total_images_needed:
            current_batch_size = min(batch_size, total_images_needed - generated_count)
            
            # Start with pure random static [-1, 1]
            initial_noise = (torch.rand(current_batch_size, 1, 224, 224, device=device) * 2 - 1)
            
            # Run the physics simulation (100 steps for higher fidelity)
            fake_images = sample_langevin(model, initial_noise, steps=100)
            
            # Denormalize from [-1, 1] back to [0, 1]
            fake_images = (fake_images + 1) / 2.0
            
            # Save the synthesized lungs
            for i in range(current_batch_size):
                img_path = os.path.join(output_dir, f"ebm_normal_{generated_count}.png")
                save_image(fake_images[i], img_path)
                generated_count += 1
                
            pbar.update(current_batch_size)

    print(f"\nSuccess! EBM Harvest complete. Images stored in: {output_dir}")

if __name__ == "__main__":
    main()