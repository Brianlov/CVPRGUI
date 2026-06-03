import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# ==========================================
# 1. Rebuild the ResNet50 Architecture
# ==========================================
def get_resnet50():
    model = models.resnet50(weights=None)
    # Adapt for 1-channel grayscale X-rays
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adapt for Binary Classification (Normal vs Pneumonia)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ==========================================
# 2. XAI Engine
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Igniting Grad-CAM Engine on: {device}")

    # File Routing
    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    out_dir = os.path.join(dataset_root, "XAI_Outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load the Model
    model_path = os.path.join(dataset_root, "hybrid_resnet50.pth") 
    
    model = get_resnet50().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Classifier weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Could not find model weights at {model_path}. Please update the path!")
        return
    model.eval()

    # The target layer for Grad-CAM in ResNet50 is usually the final convolutional block
    target_layers = [model.layer4[-1]]

    # 2. Load the Test Data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # Force 3 channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', transform=transform, download=False, size=224, root=dataset_root)
    
    # Grab 4 Normal and 4 Pneumonia cases
    normal_idx = [i for i, label in enumerate(test_dataset.labels) if label[0] == 0][:4]
    pneumonia_idx = [i for i, label in enumerate(test_dataset.labels) if label[0] == 1][:4]
    sample_indices = normal_idx + pneumonia_idx
    
    dataset_subset = Subset(test_dataset, sample_indices)
    dataloader = DataLoader(dataset_subset, batch_size=1, shuffle=False)

    # 3. Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    print("Generating attention heatmaps...")
    
    for i, (input_tensor, label) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        class_idx = label.item()
        class_name = "Normal" if class_idx == 0 else "Pneumonia"
        
        # Define the target we want to explain (explain the true label)
        targets = [ClassifierOutputTarget(class_idx)]

        # Generate the heatmap (returns a 2D numpy array [0, 1])
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # 4. Image Processing for Overlay
        # Denormalize the original image tensor
        img_normalized = (input_tensor[0, 0].cpu().numpy() * 0.5) + 0.5
        # Convert to RGB (Grad-CAM overlay requires 3 channels)
        img_rgb = np.stack((img_normalized,)*3, axis=-1)

        # Create the heatmap overlay
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        # Convert back to PIL to save easily
        vis_image = Image.fromarray(visualization)
        
        # Save output
        save_path = os.path.join(out_dir, f"gradcam_sample_{i}_{class_name}.png")
        vis_image.save(save_path)
        print(f"Saved visualization: {save_path}")

    print(f"\nXAI Verification complete. Check the {out_dir} folder.")

if __name__ == '__main__':
    main()