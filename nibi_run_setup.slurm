#!/bin/bash
#SBATCH --job-name=chandra-setup
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out

# Run Chandra OCR setup on Nibi cluster
# This installs all dependencies in a virtual environment

set -e

echo "=== Chandra OCR Setup Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

cd ~/projects/def-jic823/chandra-ocr
bash nibi_setup.sh

echo ""
echo "=== Setup Job Complete ==="
echo "End time: $(date)"
