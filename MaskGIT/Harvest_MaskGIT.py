import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import math
from tqdm import tqdm

# ==========================================
# 1. Rebuild VQ-VAE & MaskGIT (To load weights)
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    def forward(self, inputs):
        pass # Not needed for decoding

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vq = VectorQuantizer(1024, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()
        )

class MaskGIT(nn.Module):
    def __init__(self, vocab_size=1024, hidden_dim=512, depth=6, num_heads=8, seq_len=196):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, indices):
        x = self.token_emb(indices) + self.pos_emb
        x = self.transformer(x)
        return self.head(x)

# ==========================================
# 2. Iterative Decoding (The 10-Step Generation)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device for MaskGIT Harvest: {device}")

    # File Routing
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    output_dir = os.path.join(dataset_root, "MaskGIT_Synthetic", "Normal_0")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading VQ-VAE Dictionary and MaskGIT Brain...")
    
    # Load VQ-VAE Decoder
    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load(os.path.join(dataset_root, "MaskGIT_Outputs", "vqvae_tokenizer.pth"), map_location=device, weights_only=True), strict=False)
    vqvae.eval()
    
    # Load MaskGIT Transformer
    model = MaskGIT(vocab_size=1024, hidden_dim=512, depth=6, num_heads=8).to(device)
    model.load_state_dict(torch.load(os.path.join(dataset_root, "MaskGIT_Outputs", "maskgit_transformer.pth"), map_location=device, weights_only=True))
    model.eval()

    # Harvest Parameters
    total_images_needed = 2600
    batch_size = 130 # 20 batches exactly
    generated_count = 0
    
    # MaskGIT specific settings
    seq_len = 196
    inference_steps = 10 # Only 10 steps compared to DiT's 1000!

    print(f"Commencing lightning-fast token decoding for {total_images_needed} lungs...")

    with torch.no_grad():
        while generated_count < total_images_needed:
            current_batch_size = min(batch_size, total_images_needed - generated_count)
            
            # Start with 100% [MASK] tokens
            indices = torch.full((current_batch_size, seq_len), model.mask_token_id, device=device, dtype=torch.long)
            
            for step in range(inference_steps):
                # Cosine schedule: Calculate how many tokens we still need to mask this step
                mask_ratio = math.cos(math.pi / 2.0 * (step / inference_steps))
                num_mask = max(1, int(mask_ratio * seq_len))
                
                # 1. Predict all tokens
                logits = model(indices)
                probs = F.softmax(logits, dim=-1)
                
                # 2. Pick the most likely words
                sampled_tokens = torch.argmax(probs, dim=-1)
                
                # 3. Get the confidence (probability) of those picked words
                confidence = torch.gather(probs, 2, sampled_tokens.unsqueeze(-1)).squeeze(-1)
                
                # If a token was ALREADY locked in from a previous step, give it infinite confidence
                confidence[indices != model.mask_token_id] = float('inf')
                
                # 4. Mask the tokens with the lowest confidence
                if step < inference_steps - 1:
                    mask_threshold = torch.kthvalue(confidence, num_mask, dim=1)[0].unsqueeze(-1)
                    mask = confidence <= mask_threshold
                    indices = torch.where(mask, model.mask_token_id, sampled_tokens)
                else:
                    indices = sampled_tokens

            # Translate 196 valid dictionary words back into an image!
            quantized_vectors = vqvae.vq.embedding(indices) # Shape: (Batch, 196, 256)
            
            # Reshape back to spatial grid (Batch, 256, 14, 14)
            quantized_vectors = quantized_vectors.transpose(1, 2).view(current_batch_size, 256, 14, 14)
            
            # Pass through the Decoder
            generated_images = vqvae.decoder(quantized_vectors)
            
            # Denormalize [-1, 1] -> [0, 1]
            generated_images = (generated_images.clamp(-1, 1) + 1) / 2.0
            
            # Save the synthesized lungs
            for i in range(current_batch_size):
                img_path = os.path.join(output_dir, f"maskgit_normal_{generated_count}.png")
                save_image(generated_images[i], img_path)
                generated_count += 1
                
            print(f"Harvested: {generated_count} / {total_images_needed}")

    print(f"\nSuccess! MaskGIT Harvest complete. Images stored in: {output_dir}")

if __name__ == "__main__":
    main()