#!/bin/bash
#SBATCH --job-name=chandra-ocr
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

# Chandra OCR Processing on Nibi Cluster
# Usage: sbatch nibi_run_chandra.slurm <input_pdf_or_directory> <output_directory>
#
# Examples:
#   sbatch nibi_run_chandra.slurm ~/data/document.pdf ~/output
#   sbatch nibi_run_chandra.slurm ~/data/pdfs/ ~/output
#
# Configuration via environment variables:
#   METHOD: hf (HuggingFace local) or vllm (vLLM server) - default: hf
#   PAGE_RANGE: e.g., "1-5,7,9-12" - default: all pages
#   MAX_TOKENS: max output tokens per page - default: 12384
#   INCLUDE_IMAGES: true/false - default: true
#   BATCH_SIZE: pages per batch - default: 1

set -e

echo "=== Chandra OCR Job Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load modules
module load python/3.12
module load cuda/12.2

# Activate virtual environment
source ~/projects/def-jic823/chandra-ocr/.venv/bin/activate

# Verify CUDA is available
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Get input/output from command line or use defaults
INPUT_PATH="${1:-~/projects/def-jic823/test_pdfs/sample.pdf}"
OUTPUT_PATH="${2:-~/projects/def-jic823/chandra_output}"

# Configuration from environment variables
METHOD="${METHOD:-hf}"
PAGE_RANGE="${PAGE_RANGE:-}"
MAX_TOKENS="${MAX_TOKENS:-12384}"
INCLUDE_IMAGES="${INCLUDE_IMAGES:-true}"
BATCH_SIZE="${BATCH_SIZE:-1}"

echo "Configuration:"
echo "  Input: $INPUT_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Method: $METHOD"
echo "  Page Range: ${PAGE_RANGE:-all}"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Include Images: $INCLUDE_IMAGES"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Build command
CMD="chandra \"$INPUT_PATH\" \"$OUTPUT_PATH\" --method $METHOD --max-output-tokens $MAX_TOKENS --batch-size $BATCH_SIZE"

if [ -n "$PAGE_RANGE" ]; then
    CMD="$CMD --page-range \"$PAGE_RANGE\""
fi

if [ "$INCLUDE_IMAGES" = "true" ]; then
    CMD="$CMD --include-images"
else
    CMD="$CMD --no-images"
fi

# Run Chandra
echo "Running command: $CMD"
echo ""
eval $CMD

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
echo "Output saved to: $OUTPUT_PATH"
