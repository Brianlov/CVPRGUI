FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (for faster caching)
COPY hf_requirements.txt .
RUN pip install --no-cache-dir -r hf_requirements.txt

# Copy only the necessary code and weights
# We copy gui/, CNN/, EBM/, VAE/ because main.py imports from them
COPY gui/ ./gui/
COPY CNN/ ./CNN/
COPY EBM/ ./EBM/
COPY VAE/ ./VAE/

# Copy the pre-generated images and metrics
COPY results/ ./results/

# Copy the model weights needed for live inference
COPY *.pth ./

# Hugging Face Spaces require web apps to run on port 7860
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "gui.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
