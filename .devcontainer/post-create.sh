#!/bin/bash

# Authenticate with GitHub CLI
if [ -n "$GITHUB_TOKEN" ]; then
    echo "Authenticating with GitHub CLI..."
    echo "$GITHUB_TOKEN" | gh auth login --with-token
else
    echo "GITHUB_TOKEN is not set. Skipping GitHub CLI authentication."
fi

# Authenticate with Hugging Face Hub
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "Authenticating with Hugging Face Hub..."
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
else
    echo "HUGGINGFACE_TOKEN is not set. Skipping Hugging Face authentication."
fi

# Install pre-commit hooks
if [ -f .pre-commit-config.yaml ]; then
    echo "Installing pre-commit hooks..."
    pip install pre-commit
    pre-commit install
else
    echo "No .pre-commit-config.yaml file found. Skipping pre-commit setup."
fi

# Verify NVIDIA GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU is accessible in the container."
    nvidia-smi
else
    echo "NVIDIA GPU is not accessible. Ensure NVIDIA drivers and runtime are installed on the host."
fi

# Start debugpy server
echo "Starting debugpy server..."
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client &

# Verify Docker-in-Docker setup
docker --version || echo "Docker is not installed or accessible in the container."
if ! docker info &> /dev/null; then
    echo "Starting Docker daemon..."
    service docker start
fi

# Verify act installation
if command -v act &> /dev/null; then
    echo "act is installed. You can now test GitHub Actions locally."
else
    echo "act is not installed. Check the installation script."
fi

echo "Post-create script executed successfully."
