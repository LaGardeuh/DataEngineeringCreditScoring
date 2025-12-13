# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories for models and MLflow artifacts
RUN mkdir -p models mlruns reports/figures

# Copy models and MLflow runs
# Note: These directories should exist before building the image
# If they don't exist locally, the image will still build but serving won't work
COPY models/ ./models/
COPY mlruns/ ./mlruns/

# Expose MLflow serving port
EXPOSE 5001

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:///app/mlruns \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5001/health || exit 1

# Copy the serving script
COPY serve_model.py .

# Make serving script executable
RUN chmod +x serve_model.py

# Default command to serve the model using Flask
# Model can be changed via MODEL_NAME environment variable
CMD ["python", "serve_model.py"]
