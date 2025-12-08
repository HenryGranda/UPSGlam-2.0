# Use NVIDIA CUDA base image with development tools (needed for pycuda compilation)
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set CUDA architecture for PyCUDA (RTX 5070 Ti Blackwell)
ENV PYCUDA_DEFAULT_NVCC_FLAGS="-arch=sm_89"

# Expose port
EXPOSE 5000

# Run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
