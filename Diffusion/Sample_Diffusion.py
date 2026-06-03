import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

# Import the U-Net blueprint
from Diffusion_Architecture import UNet

class DiffusionSampler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Need the exact same thermodynamic constants used during training
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample(self, model, num_images=16, img_size=224):
        model.eval() # Lock the brain
        with torch.no_grad():
            # 1. Start with pure mathematical static (Gaussian Noise)
            x = torch.randn((num_images, 1, img_size, img_size)).to(self.device)

            # 2. Step backward through time (from t=999 down to t=0)
            for i in tqdm(reversed(range(self.num_timesteps)), desc="Denoising", total=self.num_timesteps):
                t = torch.full((num_images,), i, device=self.device, dtype=torch.long)
                
                # Ask the U-Net to guess what noise is currently in the image
                predicted_noise = model(x, t)
                
                # Fetch the constants for this exact millisecond of the process
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                # 3. The Reverse Entropy Equation
                # This mathematically prevents the image from collapsing into a blurry mess
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) 
                    
                # Subtract the predicted static, scale it, and step forward
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
                
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Reverse Sampling on: {device}")
    
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    output_dir = os.path.join(dataset_root, "Diffusion_Outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the fully baked U-Net
    print("Loading U-Net weights...")
    model = UNet().to(device)
    
    # This will load the final checkpoint once training is finished
    weight_path = os.path.join(dataset_root, "Diffusion_Weights", "unet_epoch_100.pth")
    
    if not os.path.exists(weight_path):
        print(f"ERROR: {weight_path} not found. Let the training finish first!")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    sampler = DiffusionSampler(num_timesteps=1000, device=device)
    
    print("Generating 16 synthetic X-rays from pure noise...")
    generated_images = sampler.sample(model, num_images=16, img_size=224)
    
    save_path = os.path.join(output_dir, "diffusion_samples.png")
    # Un-normalize back to [0, 1] so it saves correctly
    save_image(generated_images, save_path, nrow=4, normalize=True)
    print(f"Success! Photorealistic samples saved to: {save_path}")

if __name__ == '__main__':
    main()