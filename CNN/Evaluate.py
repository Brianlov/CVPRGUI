import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Hardware Setup (Hardware Agnostic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # Streaming from the secondary NVMe
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data" 
    
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # 2. Strict Validation Preprocessing
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. Load the Validation Dataset
    print("Loading Validation Data...")
    val_dataset = DataClass(split='val', transform=val_transform, download=False, size=224, root=dataset_root)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 4. Reconstruct and Load the Model
    print("Rebuilding ResNet50 Architecture...")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    
    #Put Model Path here
    weights_path = os.path.join(dataset_root, 'hybrid_maskgit_resnet50.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() 

    all_predictions = []
    all_true_labels = []
    all_probabilities = [] # Raw probabilities for the ROC curve

    print("Running Inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            outputs = model(images)
            
            # Apply softmax to get percentages (0.0 to 1.0) instead of raw logits
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            # Save the probability specifically for the "Pneumonia (1)" class
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    # 5. The Clinical Metrics (Sensitivity & Specificity)
    cm = confusion_matrix(all_true_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n" + "="*50)
    print("CLINICAL PERFORMANCE METRICS")
    print("="*50)
    print(f"Sensitivity (Recall for Pneumonia): {sensitivity:.4f}")
    print(f"Specificity (Recall for Normal):    {specificity:.4f}")
    print("="*50)

    # 6. Generate the ROC Curve
    print("Generating ROC Curve Window...")
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) - Hybrid MaskGIT ResNet50', fontweight='bold')
    ax.legend(loc="lower right")
    
    # Save the ROC curve to your NVMe
    roc_path = os.path.join(dataset_root, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    # 7. Generate and Save the Confusion Matrix Grid
    print("Generating Confusion Matrix Window...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Pneumonia (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Hybrid MaskGIT ResNet50 (GAN Augmented) - PneumoniaMNIST', fontweight='bold')
    plt.savefig(os.path.join(dataset_root, 'hybrid_maskgit_resnet50_confusion_matrix.png'), dpi=300)
    plt.show()

if __name__ == '__main__':
    main()