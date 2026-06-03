import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    # 1. Setup and Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Set this to point to your secondary NVMe drive to prevent OS drive I/O bottlenecks
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data" 
    os.makedirs(dataset_root, exist_ok=True)

    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # 2. The Golden Preprocessing & Dynamic Augmentation
    # We normalize to [-1, 1] using mean=0.5, std=0.5 so it matches your team's generator math
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # ResNet expects 3 RGB channels
        transforms.RandomHorizontalFlip(),           # Dynamic spatial augmentation
        transforms.RandomRotation(10),               # Dynamic spatial augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),                       # NO spatial augmentation for validation
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. Load Datasets
    print("Fetching 224x224 dataset...")
    train_dataset = DataClass(split='train', transform=train_transform, download=True, size=224, root=dataset_root)
    val_dataset = DataClass(split='val', transform=val_transform, download=True, size=224, root=dataset_root)

    # 4. DataLoaders
    # Using batch size 32. num_workers=0 is the safest default for Windows to prevent multiprocessing crashes.
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 5. Initialize ResNet50
    print("Loading ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the final layer for Binary Classification (Pneumonia vs Normal)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 1e-4 is a very stable learning rate for fine-tuning
    num_epochs = 10

    history_loss = []
    history_acc = []
    
    # 7. The Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # tqdm creates a nice progress bar in the terminal
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device).squeeze().long()
            
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

        # Calculate the average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)

    # 8. Save the Frozen Weights for your team
    save_path = os.path.join(dataset_root, 'baseline_resnet50.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Baseline weights saved to: {save_path}")

    # Create the Learning Curve Graph
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss (Red Line)
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(range(1, num_epochs+1), history_loss, color=color, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Accuracy (Blue Line) on the same graph
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Training Accuracy (%)', color=color, fontweight='bold')
    ax2.plot(range(1, num_epochs+1), history_acc, color=color, marker='s', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('ResNet50 Training Curve', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Save the image
    graph_path = os.path.join(dataset_root, 'learning_curve.png')
    plt.savefig(graph_path, dpi=300)
    print(f"Learning Curve saved to: {graph_path}")

if __name__ == '__main__':
    main()