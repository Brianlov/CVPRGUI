import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Sinusoidal Time Embeddings
class TimeEmbedding(nn.Module):
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

# 2. A Standard Convolutional Block
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # First convolution
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Inject the time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend time_emb to match image dimensions (Batch, Channels, Height, Width)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time embedding and apply second convolution
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Downsample or Upsample
        return self.transform(h)

# 3. The Core U-Net Architecture
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection to 224x224
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsampling path (Compressing the 224x224 X-ray)
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim) \
            for i in range(len(down_channels)-1)
        ])
        
        # Upsampling path (Expanding back to 224x224)
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
            for i in range(len(up_channels)-1)
        ])
        
        # Final output projection
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)
        
        # Downsample and store skip connections
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
            
        # Upsample and concatenate skip connections
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Concatenate along the channel dimension
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
            
        return self.output(x)

if __name__ == "__main__":
    print("Testing U-Net Dimensions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy input: Batch of 2, 1 channel (grayscale), 224x224 X-ray
    x = torch.randn(2, 1, 224, 224).to(device)
    # Dummy time steps: Randomly pick time step 50 and 800
    t = torch.tensor([50, 800], dtype=torch.long).to(device)
    
    model = UNet().to(device)
    output = model(x, t)
    
    print(f"Input Shape: {x.shape}")
    print(f"Predicted Noise Output Shape: {output.shape}")
    print("If Output Shape is [2, 1, 224, 224], the blueprint is perfect!")