import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Grad-CAM on: {device}")

    dataset_root = r"C:\Users\USER\Downloads\MedMNIST_Data"
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])

    # 2. Rebuild and Load the Frozen Model
    print("Loading Baseline ResNet50...")
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    weights_path = os.path.join(dataset_root, 'baseline_resnet50.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() # Lock the model

    # 3. Hook into the final layer
    # Target layer4[-1], which is the final convolutional block before the classification head
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Grab a sample image (without mathematical augmentations)
    val_dataset_raw = DataClass(split='val', download=False, size=224, root=dataset_root)
    
    # Mathematical preprocessing just for the model's brain
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    sample_idx = 0
    for i in range(len(val_dataset_raw)):
        img, label = val_dataset_raw[i]
        if label[0] == 1: 
            sample_idx = i
            break

    raw_img, _ = val_dataset_raw[sample_idx]
    
    # Convert the raw grayscale image to RGB and scale to [0, 1] for the visual heatmap overlay
    rgb_img = np.array(raw_img.convert('RGB'), dtype=np.float32) / 255.0

    # Push the math-ready tensor to the GPU
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 5. Generate the Heatmap
    # Show pneumonia
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Overlay the red/yellow heatmap on the black and white X-ray
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 6. Plot and Save for the Report
    print("Generating Figure...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original X-Ray (Pneumonia)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM Heatmap", fontweight='bold')
    axes[1].axis('off')

    save_path = os.path.join(dataset_root, 'gradcam_pneumonia.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Success! Image saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    main()