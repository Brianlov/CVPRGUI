import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        # Mapping the 100-dimension noise vector to a 7x7 spatial foundation
        self.init_size = 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size ** 2))

        # Using kernel=4, stride=2, padding=1 perfectly doubles the resolution at each step
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 224x224
            # Output is 1 channel (Grayscale) and uses Tanh to map pixels to [-1, 1]
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
                    block = [
                        # Wrap the convolution in spectral normalization
                        nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)
                    ]
                    if bn:
                        block.append(nn.BatchNorm2d(out_filters, 0.8))
                    return block

        self.model = nn.Sequential(
            # Input: 1 x 224 x 224
            *discriminator_block(1, 16, bn=False), # 112x112
            *discriminator_block(16, 32),          # 56x56
            *discriminator_block(32, 64),          # 28x28
            *discriminator_block(64, 128),         # 14x14
            *discriminator_block(128, 256),        # 7x7
        )

        # The downsampled image is flattened and fed into a single neuron to guess: Real or Fake?
        ds_size = 7
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
    
if __name__ == "__main__":
    print("Testing GAN Dimensions...")
    
    # 1. Create a dummy noise vector (Batch Size of 2, 100 random numbers each)
    latent_dim = 100
    z = torch.randn(2, latent_dim)
    
    # 2. Test Generator
    gen = Generator(latent_dim)
    fake_imgs = gen(z)
    print(f"Generator Output Shape: {fake_imgs.shape}") 
    # EXPECTED: [2, 1, 224, 224] (2 images, 1 channel, 224x224 pixels)
    
    # 3. Test Discriminator
    disc = Discriminator()
    validity = disc(fake_imgs)
    print(f"Discriminator Output Shape: {validity.shape}") 
    # EXPECTED: [2, 1] (2 guesses between 0.0 and 1.0)
    
    print("If you see [2, 1, 224, 224] and [2, 1], the architecture is perfectly locked in!")