#!/bin/bash
#SBATCH --job-name=chandra-vllm
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

# Chandra vLLM Server on Nibi Cluster
# This starts a vLLM inference server for Chandra OCR
#
# Usage:
#   sbatch nibi_vllm_server.slurm
#
# After the server starts, use the URL shown in the output with:
#   chandra input.pdf ./output --method vllm
#
# Configuration:
#   VLLM_PORT: Server port (default: 8000)
#   VLLM_GPUS: GPU device IDs (default: 0)

set -e

echo "=== Starting Chandra vLLM Server ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load modules
module load python/3.12
module load cuda/12.1
module load apptainer

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPUS="${VLLM_GPUS:-0}"

# Get the node hostname for connection info
NODE_HOST=$(hostname)
echo "=== vLLM Server Configuration ==="
echo "  Node: $NODE_HOST"
echo "  Port: $VLLM_PORT"
echo "  GPUs: $VLLM_GPUS"
echo ""
echo "To connect from other jobs, set:"
echo "  export VLLM_API_BASE=http://$NODE_HOST:$VLLM_PORT/v1"
echo ""

# Activate virtual environment
source ~/projects/def-jic823/chandra-ocr/.venv/bin/activate

# Check if vLLM is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm
fi

# Set environment variables
export VLLM_GPUS="$VLLM_GPUS"
export VLLM_MODEL_NAME="chandra"

# Start vLLM server using chandra_vllm command
echo "Starting vLLM server..."
echo "This will run until the job time limit or you cancel it."
echo ""

chandra_vllm

echo ""
echo "=== Server Stopped ==="
echo "End time: $(date)"
