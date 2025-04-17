# Base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for RDKit, Rust, and Firebase Admin SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libc6-dev \
    curl \
    make \
    build-essential \
    rustc \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable to ensure Rust is available for pip builds
ENV PATH="/root/.cargo/bin:$PATH"

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code last to optimize cache
COPY . /app/

# Expose port 8080
EXPOSE 8080

# Command to run the app
CMD ["python", "app.py"]
