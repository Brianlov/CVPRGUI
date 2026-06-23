import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
import medmnist
from medmnist import INFO
from tqdm import tqdm

# Import your blueprints
from GAN_Architecture import Generator, Discriminator

def main():
    # 1. Hardware & Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GAN on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    os.makedirs(os.path.join(dataset_root, "GAN_Outputs"), exist_ok=True)

    # 2. Hyperparameters (The standard for stable GANs)
    latent_dim = 100
    lr = 0.0002
    b1 = 0.5     
    b2 = 0.999   
    num_epochs = 300   # Pushed from 50 to 300
    batch_size = 128   # Taking advantage of the VRAM

    # 3. Load & Filter the Dataset (CRITICAL STEP)
    # Only grayscale images mathematically bounded to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    print("Loading dataset and isolating 'Normal (0)' lungs...")
    full_dataset = DataClass(split='train', transform=transform, download=False, size=224, root=dataset_root)
    
    # Filter out all Pneumonia (1) images.
    normal_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1][0] == 0]
    normal_dataset = Subset(full_dataset, normal_indices)
    
    # num_workers=0 to prevent Windows multiprocessing crashes
    # Pass the new batch_size variable here
    dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Isolated {len(normal_dataset)} healthy lung images for training.")

    # 4. Initialize Networks
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # 5. Loss & Optimizers
    adversarial_loss = nn.BCELoss() # Binary Cross Entropy
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Create a static noise vector to visually track how the Generator improves over time
    fixed_noise = torch.randn(16, latent_dim, device=device)

    # 6. The Arena (Training Loop)
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True)
        for i, (imgs, _) in enumerate(loop):
            
            # Ground truth labels (Real = 1.0, Fake = 0.0)
            # Change from 1.0 to 0.9 for real images
            valid = torch.full((imgs.size(0), 1), 0.9, device=device, requires_grad=False)
            # Change from 0.0 to 0.1 for fake images
            fake = torch.full((imgs.size(0), 1), 0.1, device=device, requires_grad=False)
            
            real_imgs = imgs.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate a batch of images from random noise
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            
            # The Generator's goal is to trick the Discriminator into guessing 'valid' (1.0)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # Save a visual sample of the hallucinated lungs every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_imgs = generator(fixed_noise)
                # Un-normalize from [-1, 1] back to [0, 1] for saving as a standard image file
                vutils.save_image(sample_imgs, os.path.join(dataset_root, "GAN_Outputs", f"epoch_{epoch+1}.png"), nrow=4, normalize=True)

    # 7. Save the final Generator Brain
    torch.save(generator.state_dict(), os.path.join(dataset_root, 'generator_weights.pth'))
    print("GAN Training Complete! Generator saved.")

if __name__ == '__main__':
    main()