import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import math
from tqdm import tqdm

# ==========================================
# 1. Rebuild the DiT Architecture (To load the weights)
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

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class DiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, hidden_size=256, depth=6, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.t_embedder = SinusoidalPositionEmbeddings(hidden_size)
        self.t_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = nn.Sequential(nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6), nn.Linear(hidden_size, patch_size * patch_size * in_channels))

    def forward(self, x, t):
        x = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_mlp(self.t_embedder(t))
        for block in self.blocks: x = block(x, c)
        x = self.final_layer(x)
        b, n, _ = x.shape
        h = w = int(math.sqrt(n))
        x = x.view(b, h, w, self.patch_size, self.patch_size).permute(0, 1, 3, 2, 4).contiguous().view(b, 1, self.img_size, self.img_size)
        return x

# ==========================================
# 2. Physics & Denoising Loop
# ==========================================
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device for DiT Harvest: {device}")

    # File Routing
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    output_dir = os.path.join(dataset_root, "DiT_Synthetic", "Normal_0")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading DiT weights...")
    model = DiT().to(device)
    weight_path = os.path.join(dataset_root, "DiT_Outputs", "dit_baseline.pth")
    
    if not os.path.exists(weight_path):
        print(f"Error: Weights not found at {weight_path}")
        return

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    # Physics schedule matching training exactly
    timesteps = 1000
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # Harvest Parameters
    total_images_needed = 2600
    batch_size = 100
    generated_count = 0

    print(f"Commencing Transformer denoising for {total_images_needed} lungs...")

    with torch.no_grad():
        while generated_count < total_images_needed:
            current_batch_size = min(batch_size, total_images_needed - generated_count)
            
            # Start with pure random static
            x = torch.randn((current_batch_size, 1, 224, 224), device=device)
            
            # The Reverse Diffusion Loop
            for i in tqdm(reversed(range(0, timesteps)), desc=f'Batch {generated_count//batch_size + 1} Denoising', leave=False):
                t = torch.full((current_batch_size,), i, device=device, dtype=torch.long)
                predicted_noise = model(x, t)
                
                alpha = alphas[i]
                alpha_cumprod = alphas_cumprod[i]
                beta = betas[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
            
            # Clamp and Denormalize [-1, 1] -> [0, 1]
            x = (x.clamp(-1, 1) + 1) / 2.0
            
            # Save the synthesized lungs
            for i in range(current_batch_size):
                img_path = os.path.join(output_dir, f"dit_normal_{generated_count}.png")
                save_image(x[i], img_path)
                generated_count += 1
                
            print(f"Harvested: {generated_count} / {total_images_needed}")

    print(f"\nSuccess! DiT Harvest complete. Images stored in: {output_dir}")

if __name__ == "__main__":
    main()