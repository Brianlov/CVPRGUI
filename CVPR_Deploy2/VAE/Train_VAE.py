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
# 1. The VAE Architecture
# ==========================================
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # ENCODER: Compress 224x224 down to 7x7
        self.enc1 = nn.Conv2d(1, 32, 4, 2, 1)    # Output: 112x112
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)   # Output: 56x56
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1)  # Output: 28x28
        self.enc4 = nn.Conv2d(128, 256, 4, 2, 1) # Output: 14x14
        self.enc5 = nn.Conv2d(256, 512, 4, 2, 1) # Output: 7x7

        # The Probabilistic Split (Mean and Log-Variance)
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

        # DECODER: Expand from latent_dim back to 224x224
        self.dec_fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.dec1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec5 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = x.view(x.size(0), -1) # Flatten
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        # The Reparameterization Trick: z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 512, 7, 7) # Unflatten
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        # Sigmoid pushes pixels to exact [0, 1] range
        x = torch.sigmoid(self.dec5(x)) 
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ==========================================
# 2. The Custom VAE Loss Function
# ==========================================
def vae_loss_function(recon_x, x, mu, logvar):
    # Loss 1: Reconstruction Loss (How well did it recreate the image?)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # Loss 2: KL Divergence (Forces latent space to be a standard normal distribution)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# ==========================================
# 3. The Training Engine
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting VAE Engine on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    out_dir = os.path.join(dataset_root, "VAE_Outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 1e-4

    # Strict Normalization to [0, 1] for BCE Loss
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Isolating pure 'Normal (0)' lungs...")
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    full_dataset = DataClass(split='train', transform=transform, download=False, size=224, root=dataset_root)
    
    normal_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1][0] == 0]
    normal_dataset = Subset(full_dataset, normal_indices)
    dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = VAE(latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Commencing VAE Training over {num_epochs} Epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        loop = tqdm(dataloader, leave=True)
        
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(Loss=loss.item() / len(data))

        # Generate a test image every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Sample pure random noise from the normal distribution
                sample = torch.randn(16, 128).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(16, 1, 224, 224), os.path.join(out_dir, f'sample_epoch_{epoch+1}.png'), nrow=4)

    # Save the final brain
    save_path = os.path.join(dataset_root, 'vae_baseline.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nVAE Training Complete! Weights saved to: {save_path}")

if __name__ == "__main__":
    main()