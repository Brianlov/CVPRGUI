import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import medmnist
from medmnist import INFO
from tqdm import tqdm
import numpy as np

# 1. Custom Dataset Loader for GAN Images
class SyntheticDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        # Force load as grayscale to match the real X-rays perfectly
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        # Explicitly assign the "Normal (0)" label in the exact format MedMNIST uses
        label = np.array([0])
        return image, label

def main():
    # 2. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Hybrid CNN on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    synthetic_folder = os.path.join(dataset_root, "MaskGIT_Synthetic", "Normal_0")

    # 3. Strict Preprocessing Pipeline
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # Convert the 1-channel X-ray to 3-channel RGB for the ResNet
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 4. Load the Real Dataset
    print("Loading Real MedMNIST Data...")
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    real_train_dataset = DataClass(split='train', transform=train_transform, download=False, size=224, root=dataset_root)

    # 5. Load the Synthetic Dataset
    print("Loading Synthetic GAN Data...")
    synthetic_train_dataset = SyntheticDataset(folder_path=synthetic_folder, transform=train_transform)

    # 6. Data Fusion (Stitching them together)
    print("Fusing Datasets into Hybrid Structure...")
    hybrid_dataset = ConcatDataset([real_train_dataset, synthetic_train_dataset])
    
    # Num workers = 0 to prevent Windows I/O crashes
    train_loader = DataLoader(dataset=hybrid_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print(f"Total Training Images: {len(hybrid_dataset)} (Balanced Class Distribution)")

    # 7. Initialize a Fresh ResNet50
    print("Initializing fresh ResNet50 architecture...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # 8. Training Parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10

    # 9. The Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    # 10. Save the Final Hybrid Brain
    save_path = os.path.join(dataset_root, 'hybrid_maskgit_resnet50.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nHybrid Training Complete! Weights saved to: {save_path}")

if __name__ == '__main__':
    main()