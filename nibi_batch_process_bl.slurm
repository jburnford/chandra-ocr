#!/bin/bash
#SBATCH --job-name=chandra-bl-batch
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Batch process all British Library PDFs with Chandra OCR
# 600 PDFs from BLN600 collection
#
# Usage: sbatch nibi_batch_process_bl.slurm

set -e

echo "=== Chandra OCR Batch Processing - British Library Collection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load modules
module load python/3.12
module load cuda/12.2
module load gcc arrow/18.1.0

# Set HuggingFace cache to project space (home directory has limited quota)
export HF_HOME=~/projects/def-jic823/.cache/huggingface
export TRANSFORMERS_CACHE=~/projects/def-jic823/.cache/huggingface
mkdir -p $HF_HOME

# Activate virtual environment
source ~/projects/def-jic823/chandra-ocr/.venv/bin/activate

# Verify CUDA is available
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Configuration
INPUT_DIR="~/ocr_bldata/25439023/BLN600/pdf"
OUTPUT_DIR="~/projects/def-jic823/british_library_output"
METHOD="hf"
MAX_TOKENS="12384"
BATCH_SIZE="1"

echo "Configuration:"
echo "  Input Directory: $INPUT_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Method: $METHOD"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Count PDFs
PDF_COUNT=$(find $INPUT_DIR -name "*.pdf" -type f | wc -l)
echo "Found $PDF_COUNT PDFs to process"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process all PDFs in the directory
echo "Starting batch processing..."
echo ""

chandra "$INPUT_DIR" "$OUTPUT_DIR" \
    --method $METHOD \
    --max-output-tokens $MAX_TOKENS \
    --batch-size $BATCH_SIZE \
    --include-images \
    --no-headers-footers

echo ""
echo "=== Batch Processing Complete ==="
echo "End time: $(date)"
echo "Output saved to: $OUTPUT_DIR"
echo ""

# Generate summary
echo "Processing Summary:"
PROCESSED_COUNT=$(find "$OUTPUT_DIR" -name "*.md" -type f | wc -l)
echo "  Total PDFs processed: $PROCESSED_COUNT / $PDF_COUNT"
echo "  Output directory size: $(du -sh $OUTPUT_DIR | cut -f1)"
