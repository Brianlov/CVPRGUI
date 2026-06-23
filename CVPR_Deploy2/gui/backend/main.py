import io
import base64
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS = PROJECT_ROOT / "results"

CLASSIFIER_WEIGHTS = {
    "Baseline":  PROJECT_ROOT / "baseline_resnet50.pth",
    "GAN":       RESULTS / "GAN-20260621T100120Z-3-001/GAN/hybrid_resnet50.pth",
    "EBM":       RESULTS / "EBM-20260621T100117Z-3-001/EBM/hybrid_ebm_resnet50.pth",
    "DiT":       RESULTS / "DiT-20260621T100114Z-3-001/DiT/hybrid_dit_resnet50.pth",
    "Diffusion": RESULTS / "Diffusion-20260621T100111Z-3-001/Diffusion/hybrid_diffusion_resnet50.pth",
    "MaskGIT":   RESULTS / "MaskGiT-20260621T100123Z-3-001/MaskGiT/hybrid_maskgit_resnet50.pth",
    "VAE":       RESULTS / "VAE-20260621T100129Z-3-001/VAE/hybrid_vae_resnet50.pth",
}

GAN_GEN_PATH = RESULTS / "GAN-20260621T100120Z-3-001/GAN/generator_weights.pth"
EBM_PATH     = RESULTS / "EBM-20260621T100117Z-3-001/EBM/EBM_Outputs/ebm_baseline.pth"
VAE_PATH     = RESULTS / "VAE-20260621T100129Z-3-001/VAE/vae_baseline.pth"

# Pre-generated epoch sample grid image directories (instant serving, no inference)
SAMPLE_DIRS = {
    "EBM":      RESULTS / "EBM-20260621T100117Z-3-001/EBM/EBM_Outputs",
    "DiT":      RESULTS / "DiT-20260621T100114Z-3-001/DiT/DiT_Outputs",
    "Diffusion":RESULTS / "Diffusion-20260621T100111Z-3-001/Diffusion/Diffusion_Outputs",
    "MaskGIT":  RESULTS / "MaskGiT-20260621T100123Z-3-001/MaskGiT/MaskGIT_Outputs",
}

# ── GAN Architecture ──────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.init_size = 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32,  16,  4, 2, 1), nn.BatchNorm2d(16),  nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16,   1,  4, 2, 1), nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

# ── EBM Architecture ──────────────────────────────────────────────────────────
class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,   32,  4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  64,  4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(), nn.Linear(512 * 7 * 7, 1),
        )
    def forward(self, x):
        return self.net(x)

def sample_langevin(model, x, steps=25, step_size=10, noise_scale=0.005):
    x = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        for _ in range(steps):
            energy = model(x)
            grad = torch.autograd.grad(energy.sum(), x, only_inputs=True)[0]
            x.data -= step_size * grad + noise_scale * torch.randn_like(x)
            x.data = torch.clamp(x.data, -1.0, 1.0)
    return x.detach()

# ── VAE Architecture ──────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc5 = nn.Conv2d(256, 512, 4, 2, 1)
        self.fc_mu     = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.dec1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec5 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 512, 7, 7)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        return torch.sigmoid(self.dec5(x))

    def forward(self, x):
        x = F.relu(self.enc1(x)); x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x)); x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x)); x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar

# ── Model Cache ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cache: dict = {}

def get_gan():
    if "gan" not in _cache:
        gen = Generator(latent_dim=100).to(device)
        gen.load_state_dict(torch.load(GAN_GEN_PATH, map_location=device, weights_only=True))
        gen.eval(); _cache["gan"] = gen
    return _cache["gan"]

def get_ebm():
    if "ebm" not in _cache:
        ebm = EnergyModel().to(device)
        ebm.load_state_dict(torch.load(EBM_PATH, map_location=device, weights_only=True))
        ebm.eval(); _cache["ebm"] = ebm
    return _cache["ebm"]

def get_vae():
    if "vae" not in _cache:
        vae = VAE(latent_dim=128).to(device)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=device, weights_only=True))
        vae.eval(); _cache["vae"] = vae
    return _cache["vae"]

def get_classifier(name: str):
    key = f"clf_{name}"
    if key not in _cache:
        m = models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(CLASSIFIER_WEIGHTS[name], map_location=device, weights_only=True))
        m.eval(); m.to(device); _cache[key] = m
    return _cache[key]

# ── Helpers ───────────────────────────────────────────────────────────────────
def tensor_to_b64(t: torch.Tensor) -> str:
    arr = t.squeeze().cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def pil_to_b64(pil: Image.Image) -> str:
    pil = pil.convert("RGB")
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def grid_png_to_tiles(path: Path, n: int = 3) -> list[str]:
    """Crop n individual tiles from a 4x4 grid image and return as base64 list."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    cols, rows = 4, 4
    tw, th = w // cols, h // rows
    tiles = []
    positions = random.sample(range(cols * rows), min(n, cols * rows))
    for pos in positions:
        r, c = divmod(pos, cols)
        tile = img.crop((c * tw, r * th, (c+1) * tw, (r+1) * th))
        tile = tile.resize((224, 224), Image.LANCZOS)
        buf = io.BytesIO(); tile.save(buf, format="PNG")
        tiles.append(base64.b64encode(buf.getvalue()).decode())
    return tiles

def get_sample_tiles(model_key: str, n: int = 3) -> list[str]:
    """Return n sample images from pre-generated epoch outputs."""
    sample_dir = SAMPLE_DIRS.get(model_key)
    if not sample_dir or not sample_dir.exists():
        return []
    pngs = list(sample_dir.glob("*.png"))
    # Prefer later epoch samples (higher quality)
    pngs.sort()
    # Pick from the last half of available samples
    best = pngs[len(pngs)//2:] if len(pngs) > 1 else pngs
    chosen = random.sample(best, min(n, len(best)))
    results = []
    for p in chosen:
        results.extend(grid_png_to_tiles(p, n=1))
        if len(results) >= n:
            break
    return results[:n]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CVPR Medical Imaging GUI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.get("/metrics_images")
def metrics_images():
    baseline_cm = RESULTS / "CNN-20260621T100109Z-3-001/CNN/Confusion Matrixabseline.png"
    gan_cm = RESULTS / "GAN-20260621T100120Z-3-001/GAN/hybrid_confusion_matrix.png"
    ebm_cm = RESULTS / "EBM-20260621T100117Z-3-001/EBM/hybrid_ebm_confusion_matrix.png"
    
    baseline_roc = RESULTS / "CNN-20260621T100109Z-3-001/CNN/rocbaseline.png"
    gan_roc = RESULTS / "GAN-20260621T100120Z-3-001/GAN/roc_curvehybrid.png"
    ebm_roc = RESULTS / "EBM-20260621T100117Z-3-001/EBM/roc_curveebm.png"

    def to_b64(path):
        if not path.exists(): return None
        return pil_to_b64(Image.open(path))

    return {
        "Baseline": {"cm": to_b64(baseline_cm), "roc": to_b64(baseline_roc)},
        "GAN": {"cm": to_b64(gan_cm), "roc": to_b64(gan_roc)},
        "EBM": {"cm": to_b64(ebm_cm), "roc": to_b64(ebm_roc)},
    }

@app.get("/dataset_samples")
def dataset_samples():
    import medmnist
    from medmnist import PneumoniaMNIST
    dataset = PneumoniaMNIST(split='test', download=True)
    normal_imgs = []
    pneumonia_imgs = []
    
    for img, label in dataset:
        if label[0] == 0 and len(normal_imgs) < 3:
            normal_imgs.append(pil_to_b64(img))
        elif label[0] == 1 and len(pneumonia_imgs) < 3:
            pneumonia_imgs.append(pil_to_b64(img))
        if len(normal_imgs) == 3 and len(pneumonia_imgs) == 3:
            break
            
    return {"normal": normal_imgs, "pneumonia": pneumonia_imgs}

@app.post("/augment")
async def augment(file: UploadFile = File(...)):
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw)).convert("L").resize((224, 224), Image.LANCZOS)
    aug_list = [
        ("Original",      transforms.Compose([transforms.Resize((224, 224))])),
        ("H-Flip",        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0)])),
        ("Rotation ±15°", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(15)])),
        ("Brightness",    transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.5, contrast=0.4)])),
        ("Gaussian Blur", transforms.Compose([transforms.Resize((224, 224)), transforms.GaussianBlur(kernel_size=11, sigma=(2, 4))])),
        ("Random Crop",   transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224)])),
    ]
    results = [{"label": lbl, "image": pil_to_b64(t(pil))} for lbl, t in aug_list]
    return {"augmentations": results}

@app.post("/generate_all")
async def generate_all(n: int = 3):
    """Generate/retrieve n samples from all 6 generative models."""
    n = min(max(n, 1), 4)
    output = {}

    # GAN — live generation (fast)
    gen = get_gan()
    z = torch.randn(n, 100, device=device)
    with torch.no_grad():
        fake = gen(z)
    output["GAN"] = [tensor_to_b64((fake[i] + 1) / 2.0) for i in range(n)]

    # VAE — live generation (just decoder, very fast)
    vae = get_vae()
    z_vae = torch.randn(n, 128, device=device)
    with torch.no_grad():
        vae_out = vae.decode(z_vae)
    output["VAE"] = [tensor_to_b64(vae_out[i]) for i in range(n)]

    # EBM — serve from pre-generated epoch samples (Langevin is slow on CPU)
    output["EBM"] = get_sample_tiles("EBM", n)

    # DiT — serve from pre-generated epoch samples (1000-step denoising is slow)
    output["DiT"] = get_sample_tiles("DiT", n)

    # Diffusion — serve from pre-generated epoch samples
    output["Diffusion"] = get_sample_tiles("Diffusion", n)

    # MaskGIT — serve from pre-generated epoch samples
    output["MaskGIT"] = get_sample_tiles("MaskGIT", n)

    return output

# Keep the old /generate endpoint for backward compatibility
@app.post("/generate")
async def generate(n: int = 3):
    n = min(max(n, 1), 6)
    gen = get_gan()
    z = torch.randn(n, 100, device=device)
    with torch.no_grad():
        fake = gen(z)
    gan_imgs = [tensor_to_b64((fake[i] + 1) / 2.0) for i in range(n)]
    ebm = get_ebm()
    noise = torch.rand(n, 1, 224, 224, device=device) * 2 - 1
    ebm_raw = sample_langevin(ebm, noise, steps=25)
    ebm_imgs = [tensor_to_b64((ebm_raw[i] + 1) / 2.0) for i in range(n)]
    return {"gan": gan_imgs, "ebm": ebm_imgs}

@app.post("/classify")
async def classify(file: UploadFile = File(...), model_name: str = Form("Baseline")):
    if model_name not in CLASSIFIER_WEIGHTS:
        return JSONResponse(status_code=400, content={"error": f"Unknown model: {model_name}"})
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw)).convert("L")
    t = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = t(pil).unsqueeze(0).to(device)
    clf = get_classifier(model_name)
    with torch.no_grad():
        prob = torch.softmax(clf(tensor), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(prob))
    return {
        "model": model_name,
        "prediction": "Normal" if idx == 0 else "Pneumonia",
        "confidence": float(prob[idx]),
        "prob_normal": float(prob[0]),
        "prob_pneumonia": float(prob[1]),
    }
