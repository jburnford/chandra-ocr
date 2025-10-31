#!/bin/bash
#
# Setup script for Chandra OCR on Nibi cluster
# Run this once to set up the Python environment
#

set -e

echo "=== Setting up Chandra OCR on Nibi cluster ==="

# Load modules
echo "Loading required modules..."
module load python/3.12
module load cuda/12.2

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ~/projects/def-jic823/chandra-ocr/.venv

# Activate virtual environment
source ~/projects/def-jic823/chandra-ocr/.venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.2 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Chandra from source
echo "Installing Chandra OCR and dependencies..."
cd ~/projects/def-jic823/chandra-ocr
pip install -e .

# Install flash-attention for better performance (optional but recommended)
echo "Installing flash-attention (this may take a while)..."
pip install flash-attn --no-build-isolation || echo "Warning: flash-attention installation failed, continuing without it"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To use Chandra on Nibi:"
echo "  1. Submit a job: sbatch ~/projects/def-jic823/chandra-ocr/nibi_run_chandra.slurm"
echo "  2. Or run interactively: salloc --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=1:00:00"
echo ""
