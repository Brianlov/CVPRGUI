# Run from project root: .\gui\start_backend.ps1
Write-Host "Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Cyan
& ".\venv\Scripts\python.exe" -m uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000 --reload
