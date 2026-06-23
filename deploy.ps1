$source = "c:\Users\Brian ooi\Documents\code\CVPR\CVPRAssignment"
$dest = "c:\Users\Brian ooi\Documents\code\CVPR\CVPRAssignment\CVPR_Deploy2"

if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Force -Path $dest | Out-Null

Copy-Item "$source\Dockerfile" "$dest\"
Copy-Item "$source\hf_requirements.txt" "$dest\requirements.txt"
Copy-Item "$source\baseline_resnet50.pth" "$dest\"

Copy-Item "$source\gui" "$dest\gui" -Recurse
Copy-Item "$source\CNN" "$dest\CNN" -Recurse
Copy-Item "$source\EBMs" "$dest\EBMs" -Recurse
Copy-Item "$source\GAN" "$dest\GAN" -Recurse
Copy-Item "$source\VAE" "$dest\VAE" -Recurse

Get-ChildItem -Path $dest -Include "__pycache__" -Recurse | Remove-Item -Recurse -Force

$filesToCopy = @(
    "CNN-20260621T100109Z-3-001\CNN\Confusion Matrixabseline.png",
    "CNN-20260621T100109Z-3-001\CNN\rocbaseline.png",
    "GAN-20260621T100120Z-3-001\GAN\hybrid_confusion_matrix.png",
    "GAN-20260621T100120Z-3-001\GAN\roc_curvehybrid.png",
    "GAN-20260621T100120Z-3-001\GAN\hybrid_resnet50.pth",
    "GAN-20260621T100120Z-3-001\GAN\generator_weights.pth",
    "EBM-20260621T100117Z-3-001\EBM\hybrid_ebm_confusion_matrix.png",
    "EBM-20260621T100117Z-3-001\EBM\roc_curveebm.png",
    "EBM-20260621T100117Z-3-001\EBM\hybrid_ebm_resnet50.pth",
    "EBM-20260621T100117Z-3-001\EBM\EBM_Outputs\ebm_baseline.pth",
    "EBM-20260621T100117Z-3-001\EBM\EBM_Outputs\ebm_sample_10.png",
    "VAE-20260621T100129Z-3-001\VAE\vae_baseline.pth"
)

foreach ($f in $filesToCopy) {
    $srcFile = "$source\results\$f"
    $dstFile = "$dest\results\$f"
    $parent = Split-Path $dstFile
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    if (Test-Path $srcFile) {
        Copy-Item $srcFile $dstFile
    } else {
        Write-Host "Warning: $srcFile not found"
    }
}

cd $dest
git init
git lfs install
git lfs track "*.png"
git lfs track "*.pth"
git add .
git commit -m "Clean deployment push"
git remote add hf https://huggingface.co/spaces/BrianLov/guiBackend
# Need to use the token auth remote url so it pushes seamlessly
git remote set-url hf https://huggingface.co/spaces/BrianLov/guiBackend
git push hf master:main -f
Write-Host "Deployment push complete!"
