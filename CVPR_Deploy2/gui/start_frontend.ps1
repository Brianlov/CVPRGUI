# Run from project root: .\gui\start_frontend.ps1
Write-Host "Starting React frontend on http://localhost:5173 ..." -ForegroundColor Cyan
Set-Location ".\gui\frontend"
npm run dev
