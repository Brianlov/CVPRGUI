import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
from tqdm import tqdm

# ==========================================
# 1. The Vector Quantizer (The Dictionary)
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # The actual "Dictionary" of visual words
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # Flatten input from (Batch, Channels, H, W) to (Batch*H*W, Channels)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate mathematical distance between the image patches and our dictionary words
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Find the closest matching "word" in the dictionary for each patch
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Retrieve the visual words
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss: Force the dictionary to update, and force the encoder to commit to the dictionary
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator trick (bypasses the non-differentiable argmin for autograd)
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

# ==========================================
# 2. VQ-VAE Architecture
# ==========================================
class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Codebook: 1024 possible visual words, each represented by a 256-dimension math vector
        self.vq = VectorQuantizer(num_embeddings=1024, embedding_dim=256)
        
        # Encoder: Compresses 224x224 to 14x14
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1), nn.ReLU()
        )
        
        # Decoder: Expands 14x14 back to 224x224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss

# ==========================================
# 3. Training Engine
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting VQ-VAE Tokenizer on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    out_dir = os.path.join(dataset_root, "MaskGIT_Outputs")
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("Loading Normal Cohort for Dictionary Extraction...")
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    full_dataset = DataClass(split='train', transform=transform, download=False, size=224, root=dataset_root)
    
    normal_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1][0] == 0]
    normal_dataset = Subset(full_dataset, normal_indices)
    dataloader = DataLoader(normal_dataset, batch_size=64, shuffle=True, num_workers=0)

    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    num_epochs = 50
    print("Commencing Discrete Latent Training...")

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(dataloader, leave=True)
        
        for images, _ in loop:
            images = images.to(device)
            optimizer.zero_grad()
            
            reconstructed, vq_loss = model(images)
            
            # Reconstruction Loss + Dictionary Commitment Loss
            recon_loss = F.mse_loss(reconstructed, images)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"VQ-VAE Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(Loss=loss.item(), Recon=recon_loss.item(), VQ=vq_loss.item())

        # Generate sample reconstructions
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Compare real images (top) to dictionary-reconstructed images (bottom)
                test_imgs = images[:8]
                recon_imgs, _ = model(test_imgs)
                comparison = torch.cat([test_imgs, recon_imgs], dim=0)
                comparison = (comparison + 1) / 2.0
                save_image(comparison, os.path.join(out_dir, f'vqvae_recon_{epoch+1}.png'), nrow=8)

    torch.save(model.state_dict(), os.path.join(out_dir, 'vqvae_tokenizer.pth'))
    print("\nVisual Dictionary locked and saved.")

if __name__ == '__main__':
    main()