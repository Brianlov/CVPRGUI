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
import math

# ==========================================
# 1. Rebuild VQ-VAE to load the Dictionary
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding.weight.shape[1])
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return None, None, encoding_indices.view(inputs.shape[0], inputs.shape[1], inputs.shape[2])

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vq = VectorQuantizer(1024, 256)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()
        )

# ==========================================
# 2. The Masked Transformer
# ==========================================
class MaskGIT(nn.Module):
    def __init__(self, vocab_size=1024, hidden_dim=512, depth=8, num_heads=8, seq_len=196):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size # The special [MASK] token is index 1024
        
        # Embedding layer for the 1024 dictionary words + 1 mask token
        self.token_emb = nn.Embedding(vocab_size + 1, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        # Standard Bidirectional Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Predicts the dictionary word for every position
        self.head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, indices):
        # indices shape: (Batch, 196)
        x = self.token_emb(indices) + self.pos_emb
        x = self.transformer(x)
        logits = self.head(x) # (Batch, 196, 1024)
        return logits

# ==========================================
# 3. Training Engine
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting MaskGIT Transformer on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    out_dir = os.path.join(dataset_root, "MaskGIT_Outputs")
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("Loading Normal Cohort...")
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    full_dataset = DataClass(split='train', transform=transform, download=False, size=224, root=dataset_root)
    
    normal_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1][0] == 0]
    normal_dataset = Subset(full_dataset, normal_indices)
    dataloader = DataLoader(normal_dataset, batch_size=64, shuffle=True, num_workers=0)

    # Load the Frozen Dictionary (VQ-VAE)
    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load(os.path.join(out_dir, 'vqvae_tokenizer.pth'), map_location=device, weights_only=True))
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False # Freeze it!

    # Initialize the Transformer
    model = MaskGIT(vocab_size=1024, hidden_dim=512, depth=6, num_heads=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 150
    print("Commencing Masked Image Modeling (MIM)...")

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(dataloader, leave=True)
        
        for images, _ in loop:
            images = images.to(device)
            batch_size = images.shape[0]
            
            # 1. Translate image into dictionary words using the frozen VQ-VAE
            with torch.no_grad():
                z = vqvae.encoder(images)
                _, _, indices = vqvae.vq(z)
                indices = indices.view(batch_size, -1) # Flatten to (Batch, 196 sequence)
            
            # 2. Dynamic Masking (Cosine Schedule)
            # Randomly decide how much of the image to erase (from 50% to 100%)
            mask_ratio = torch.rand(1).item() * 0.5 + 0.5 
            
            # Create a boolean mask of the sequence
            rand_matrix = torch.rand(batch_size, 196, device=device)
            mask = rand_matrix < mask_ratio
            
            # Replace the erased words with the special [MASK] token (index 1024)
            masked_indices = indices.clone()
            masked_indices[mask] = model.mask_token_id
            
            # 3. Predict the missing words
            optimizer.zero_grad()
            logits = model(masked_indices) # Shape: (Batch, 196, 1024)
            
            # 4. Calculate Loss (Only on the tokens we actually masked)
            # Reshape logits and indices to match CrossEntropyLoss expectations
            loss = criterion(logits[mask], indices[mask])
            
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"MaskGIT Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(Loss=loss.item(), Masked=f"{mask_ratio*100:.0f}%")

    torch.save(model.state_dict(), os.path.join(out_dir, 'maskgit_transformer.pth'))
    print("\nMaskGIT Brain locked and saved.")

if __name__ == '__main__':
    main()