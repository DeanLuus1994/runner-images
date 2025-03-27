FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    docker.io \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit for GPU passthrough
RUN curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/ubuntu20.04/nvidia-container-runtime.list | tee /etc/apt/sources.list.d/nvidia-container-runtime.list && \
    apt-get update && apt-get install -y nvidia-container-runtime

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
