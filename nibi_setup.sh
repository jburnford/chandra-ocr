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
module load gcc arrow/18.1.0

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
# Skip hf-xet which requires Rust compiler (optional dependency)
pip install -e . --no-deps
# Now install dependencies individually, skipping hf-xet
pip install beautifulsoup4>=4.14.2 click>=8.0.0 filetype>=1.2.0 flask>=3.0.0 \
    markdownify==1.1.0 openai>=2.2.0 pillow>=10.2.0 pydantic>=2.12.0 \
    pydantic-settings>=2.11.0 pypdfium2>=4.30.0 python-dotenv>=1.1.1 \
    qwen-vl-utils>=0.0.14 transformers>=4.57.1 streamlit>=1.50.0 accelerate>=1.11.0

# Install flash-attention for better performance (optional but recommended)
echo "Installing flash-attention (this may take a while)..."
pip install flash-attn --no-build-isolation || echo "Warning: flash-attention installation failed, continuing without it"

# Reinstall correct torch/torchvision versions after flash-attn downgrades them
echo "Ensuring correct PyTorch version..."
pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To use Chandra on Nibi:"
echo "  1. Submit a job: sbatch ~/projects/def-jic823/chandra-ocr/nibi_run_chandra.slurm"
echo "  2. Or run interactively: salloc --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=1:00:00"
echo ""
