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
import math

# ==========================================
# 1. Timestep Embedding
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ==========================================
# 2. Adaptive Layer Norm (AdaLN) & DiT Block
# ==========================================
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Maps the time embedding to scale (gamma) and shift (beta) parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c is the timestep embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention with AdaLN
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with AdaLN
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

# ==========================================
# 3. The Full Diffusion Transformer (DiT)
# ==========================================
class DiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, hidden_size=256, depth=6, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 1. Patchify: Maps 16x16 pixel blocks into hidden vectors
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # 2. Time Embedding
        self.t_embedder = SinusoidalPositionEmbeddings(hidden_size)
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 4. Un-patchify: Maps hidden vectors back to 16x16 pixel blocks
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        )

    def forward(self, x, t):
        # Patchify the image
        x = self.x_embedder(x)            # (Batch, Hidden, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (Batch, 196, Hidden)
        x = x + self.pos_embed            # Add positional awareness
        
        # Process Timestep
        t = self.t_embedder(t)
        c = self.t_mlp(t)
        
        # Pass through DiT blocks
        for block in self.blocks:
            x = block(x, c)
            
        # Un-patchify and reshape back to image
        x = self.final_layer(x)           # (Batch, 196, 256)
        
        # Reshape: (Batch, 196, 256) -> (Batch, 1, 224, 224)
        b, n, _ = x.shape
        h = w = int(math.sqrt(n))
        x = x.view(b, h, w, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, 1, self.img_size, self.img_size)
        
        return x

# ==========================================
# 4. Training Engine & Noise Scheduler
# ==========================================
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting DiT (Diffusion Transformer) Engine on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    out_dir = os.path.join(dataset_root, "DiT_Outputs")
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("Loading Normal Cohort for Tokenization...")
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    full_dataset = DataClass(split='train', transform=transform, download=False, size=224, root=dataset_root)
    
    normal_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1][0] == 0]
    normal_dataset = Subset(full_dataset, normal_indices)
    dataloader = DataLoader(normal_dataset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize Physics
    timesteps = 1000
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # Initialize Brain
    model = DiT(img_size=224, patch_size=16, in_channels=1, hidden_size=256, depth=6, num_heads=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 150
    print("Commencing DiT Space-Time Training...")

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(dataloader, leave=True)
        
        for images, _ in loop:
            images = images.to(device)
            batch_size = images.shape[0]

            # 1. Sample random noise
            noise = torch.randn_like(images)
            
            # 2. Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # 3. Add noise to images (Forward Diffusion)
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])[:, None, None, None]
            
            noisy_images = sqrt_alphas_cumprod_t * images + sqrt_one_minus_alphas_cumprod_t * noise
            
            # 4. Predict the noise using the Transformer
            optimizer.zero_grad()
            predicted_noise = model(noisy_images, t)
            
            # 5. Optimize
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"DiT Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(Loss=loss.item())

        # Generate Sample
        if (epoch + 1) % 10 == 0:
            model.eval()
            print(f"\nDecoding latent patches for Epoch {epoch+1}...")
            with torch.no_grad():
                x = torch.randn((8, 1, 224, 224), device=device)
                
                # Reverse Diffusion Loop
                for i in tqdm(reversed(range(0, timesteps)), desc='Denoising'):
                    t = torch.full((8,), i, device=device, dtype=torch.long)
                    predicted_noise = model(x, t)
                    
                    alpha = alphas[i]
                    alpha_cumprod = alphas_cumprod[i]
                    beta = betas[i]
                    
                    if i > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                        
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
                
                x = (x.clamp(-1, 1) + 1) / 2
                save_image(x, os.path.join(out_dir, f'dit_sample_{epoch+1}.png'), nrow=4)

    torch.save(model.state_dict(), os.path.join(out_dir, 'dit_baseline.pth'))
    print("\nDiT Matrix locked and saved.")

if __name__ == '__main__':
    main()