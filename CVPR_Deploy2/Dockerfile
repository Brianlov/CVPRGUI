FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (for faster caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary code and weights
COPY gui/ ./gui/
COPY CNN/ ./CNN/
COPY EBMs/ ./EBMs/
COPY GAN/ ./GAN/
COPY VAE/ ./VAE/

# Copy the pre-generated images and metrics
COPY results/ ./results/

# Copy the model weights needed for live inference
COPY *.pth ./

# Hugging Face Spaces require web apps to run on port 7860
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "gui.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
