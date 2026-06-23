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

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = r"C:\Users\Brian ooi\Documents\code\CVPR\CVPRAssignment"
RESULTS_ROOT  = os.path.join(PROJECT_ROOT, "results_run2")
FIRST_RESULTS = os.path.join(PROJECT_ROOT, "results")

MODELS = {
    "Baseline": {
        "weights": os.path.join(PROJECT_ROOT, "baseline_resnet50.pth"),
        "label":   "Baseline ResNet50 (No Augmentation)",
    },
    "GAN": {
        "weights": os.path.join(FIRST_RESULTS, "GAN-20260621T100120Z-3-001", "GAN", "hybrid_resnet50.pth"),
        "label":   "Hybrid GAN ResNet50",
    },
    "EBM": {
        "weights": os.path.join(FIRST_RESULTS, "EBM-20260621T100117Z-3-001", "EBM", "hybrid_ebm_resnet50.pth"),
        "label":   "Hybrid EBM ResNet50",
    },
    "DiT": {
        "weights": os.path.join(FIRST_RESULTS, "DiT-20260621T100114Z-3-001", "DiT", "hybrid_dit_resnet50.pth"),
        "label":   "Hybrid DiT ResNet50",
    },
    "Diffusion": {
        "weights": os.path.join(FIRST_RESULTS, "Diffusion-20260621T100111Z-3-001", "Diffusion", "hybrid_diffusion_resnet50.pth"),
        "label":   "Hybrid Diffusion ResNet50",
    },
    "MaskGIT": {
        "weights": os.path.join(FIRST_RESULTS, "MaskGiT-20260621T100123Z-3-001", "MaskGiT", "hybrid_maskgit_resnet50.pth"),
        "label":   "Hybrid MaskGIT ResNet50",
    },
    "VAE": {
        "weights": os.path.join(FIRST_RESULTS, "VAE-20260621T100129Z-3-001", "VAE", "hybrid_vae_resnet50.pth"),
        "label":   "Hybrid VAE ResNet50",
    },
}
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def get_val_loader():
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    info      = INFO["pneumoniamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    val_ds    = DataClass(split="val", transform=val_transform,
                          download=False, size=224, root=PROJECT_ROOT)
    return DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

def evaluate(model, val_loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device).squeeze().long()
            out  = model(imgs)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
            probs.extend(prob[:, 1].cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probs)

def save_results(name, label, preds, labels, probs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal (0)", "Pneumonia (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{label} - PneumoniaMNIST", fontweight="bold")
    plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guessing")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.set_title(f"ROC - {label}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, f"{name}_roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Score TXT ─────────────────────────────────────────────────────────────
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    report      = classification_report(labels, preds, target_names=["Normal (0)", "Pneumonia (1)"])

    score_path = os.path.join(out_dir, f"{name}_score.txt")
    with open(score_path, "w") as f:
        f.write(f"{'='*50}\n")
        f.write(f"CLASSIFICATION REPORT: {name.upper()}-AUGMENTED MODEL\n")
        f.write(f"{'='*50}\n\n")
        f.write(report)
        f.write(f"\n{'='*50}\n")
        f.write(f"CLINICAL METRICS\n")
        f.write(f"{'='*50}\n")
        f.write(f"Accuracy    : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Sensitivity : {sensitivity:.4f} ({sensitivity*100:.2f}%)\n")
        f.write(f"Specificity : {specificity:.4f} ({specificity*100:.2f}%)\n")
        f.write(f"AUC         : {roc_auc:.4f}\n")
        f.write(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")

    return {
        "accuracy": accuracy, "sensitivity": sensitivity,
        "specificity": specificity, "auc": roc_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

def save_summary(summary, out_dir):
    """Save a master comparison table."""
    path = os.path.join(out_dir, "SUMMARY_TABLE.txt")
    with open(path, "w") as f:
        header = f"{'Model':<12} {'Accuracy':>10} {'Sensitivity':>13} {'Specificity':>13} {'AUC':>8} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}"
        f.write("="*len(header) + "\n")
        f.write("MASTER COMPARISON TABLE - PneumoniaMNIST\n")
        f.write("="*len(header) + "\n")
        f.write(header + "\n")
        f.write("-"*len(header) + "\n")
        for name, m in summary.items():
            f.write(
                f"{name:<12} {m['accuracy']*100:>9.2f}% "
                f"{m['sensitivity']*100:>12.2f}% "
                f"{m['specificity']*100:>12.2f}% "
                f"{m['auc']:>8.4f} "
                f"{m['tn']:>5} {m['fp']:>5} {m['fn']:>5} {m['tp']:>5}\n"
            )
        f.write("="*len(header) + "\n")
    print(f"\n✅ Summary table saved → {path}")

def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}\n")
    val_loader = get_val_loader()
    summary    = {}

    for name, cfg in MODELS.items():
        print(f"── Evaluating: {name} ──────────────────────────")
        if not os.path.exists(cfg["weights"]):
            print(f"   ⚠️  Weights not found: {cfg['weights']}\n")
            continue

        model = build_model(device)
        model.load_state_dict(torch.load(cfg["weights"], map_location=device, weights_only=True))

        preds, labels, probs = evaluate(model, val_loader, device)
        out_dir = os.path.join(RESULTS_ROOT, name)
        metrics = save_results(name, cfg["label"], preds, labels, probs, out_dir)
        summary[name] = metrics

        print(f"   Accuracy   : {metrics['accuracy']*100:.2f}%")
        print(f"   Sensitivity: {metrics['sensitivity']*100:.2f}%")
        print(f"   Specificity: {metrics['specificity']*100:.2f}%")
        print(f"   AUC        : {metrics['auc']:.4f}")
        print(f"   TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")
        print(f"   ✅ Saved → {out_dir}\n")

    save_summary(summary, RESULTS_ROOT)
    print(f"\n🎉 All done! Results saved to: {RESULTS_ROOT}")

if __name__ == "__main__":
    main()
