# Chandra OCR on Nibi Cluster

This repository contains deployment scripts for running [Chandra OCR](https://github.com/datalab-to/chandra) on the Nibi cluster with H100 GPUs.

## What is Chandra?

Chandra is a highly accurate OCR model that converts images and PDFs into structured HTML/Markdown/JSON while preserving layout information. It excels at:
- Handwriting recognition
- Form reconstruction (including checkboxes)
- Tables, math, and complex layouts
- Image and diagram extraction
- 40+ language support

## Requirements

- Python 3.12
- CUDA 12.1
- H100 GPU (or other GPU with compute capability 7.5+)
- 32GB+ GPU memory recommended for large documents

## Setup

### Initial Setup on Nibi

1. **Clone the repository on Nibi:**
```bash
ssh nibi
cd ~/projects/def-jic823
git clone git@github.com:jburnford/chandra-ocr.git
```

2. **Run the setup script:**
```bash
cd ~/projects/def-jic823/chandra-ocr
bash nibi_setup.sh
```

This will:
- Load required modules (Python 3.12, CUDA 12.1)
- Create a virtual environment
- Install PyTorch with CUDA support
- Install Chandra OCR and dependencies
- Install flash-attention for better performance

Setup takes approximately 15-20 minutes.

## Usage

### Method 1: Batch Processing with HuggingFace (Recommended)

This method runs Chandra locally on the GPU using HuggingFace transformers. Best for batch processing multiple documents.

**Basic usage:**
```bash
sbatch nibi_run_chandra.slurm <input_pdf> <output_directory>
```

**Examples:**
```bash
# Process a single PDF
sbatch nibi_run_chandra.slurm ~/data/document.pdf ~/output

# Process all PDFs in a directory
sbatch nibi_run_chandra.slurm ~/data/pdfs/ ~/output

# Process specific page range
PAGE_RANGE="1-10,15,20-25" sbatch nibi_run_chandra.slurm ~/data/doc.pdf ~/output

# Process without extracting images
INCLUDE_IMAGES=false sbatch nibi_run_chandra.slurm ~/data/doc.pdf ~/output
```

**Configuration via environment variables:**
- `PAGE_RANGE`: Page range (e.g., "1-5,7,9-12")
- `MAX_TOKENS`: Max output tokens per page (default: 12384)
- `INCLUDE_IMAGES`: Extract images (default: true)
- `BATCH_SIZE`: Pages per batch (default: 1)

### Method 2: vLLM Server (For Production/API Use)

This method runs a vLLM inference server that can handle multiple concurrent requests.

**Step 1: Start the vLLM server:**
```bash
sbatch nibi_vllm_server.slurm
```

**Step 2: Get the server URL from the job output:**
```bash
cat chandra-vllm-<JOBID>.out
```

Look for the line showing: `export VLLM_API_BASE=http://<node>:8000/v1`

**Step 3: Submit processing jobs using the server:**
```bash
METHOD=vllm VLLM_API_BASE=http://<node>:8000/v1 sbatch nibi_run_chandra.slurm input.pdf output/
```

### Interactive Testing

For quick tests or development:

```bash
# Request interactive GPU node
salloc --gres=gpu:h100:1 --mem=64G --cpus-per-task=8 --time=1:00:00

# Once allocated, activate environment
module load python/3.12 cuda/12.1
source ~/projects/def-jic823/chandra-ocr/.venv/bin/activate

# Run Chandra directly
chandra test.pdf ./output --method hf
```

## Output Structure

Each processed document creates a directory with:
- `<filename>.md` - Markdown output
- `<filename>.html` - HTML output
- `<filename>_metadata.json` - Processing metadata
- `images/` - Extracted images (if enabled)

## Performance Expectations

Based on testing with H100 GPUs:
- **Model Loading**: ~30-60 seconds (first time per job)
- **Processing Speed**:
  - Simple documents: 1-2 pages/second
  - Complex documents (tables, math): 0.5-1 page/second
  - Handwriting-heavy: 0.3-0.5 page/second

For large batches (>500 pages), expect:
- ~10-15 minutes per 100 pages for typical documents
- More time for complex historical documents

## Cost Considerations

According to benchmarks, Chandra costs approximately:
- **$190 USD per million pages** (estimated)
- Compared to $6,200 for GPT-4o
- ~97% cost reduction vs commercial APIs

## Monitoring Jobs

```bash
# Check job status
squeue -u jic823

# View running job output
tail -f chandra-ocr-<JOBID>.out

# View completed job output
cat chandra-ocr-<JOBID>.out

# Cancel a job
scancel <JOBID>
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce batch size: `BATCH_SIZE=1`
2. Process fewer pages at once: `PAGE_RANGE="1-10"`
3. Request more memory in the SLURM script (edit `--mem=` line)

### CUDA Not Available

Check that modules are loaded:
```bash
module load python/3.12 cuda/12.1
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Model Download Issues

First run downloads the model (~15GB). If it fails:
```bash
# Manual download on a compute node
salloc --gres=gpu:1 --time=1:00:00
source ~/projects/def-jic823/chandra-ocr/.venv/bin/activate
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('datalab-to/chandra')"
```

## Updating

To pull latest changes:

**On laptop:**
```bash
cd /home/jic823/chandra
git pull upstream main
git push origin main
```

**On Nibi:**
```bash
cd ~/projects/def-jic823/chandra-ocr
git pull origin main
source .venv/bin/activate
pip install -e . --upgrade
```

## Repository Structure

```
chandra-ocr/
├── nibi_setup.sh              # Initial environment setup
├── nibi_run_chandra.slurm     # Batch processing job script
├── nibi_vllm_server.slurm     # vLLM server job script
├── NIBI_DEPLOYMENT.md         # This file
├── chandra/                   # Source code
├── pyproject.toml             # Dependencies
└── .venv/                     # Virtual environment (after setup)
```

## Resources

- **Upstream Repository**: https://github.com/datalab-to/chandra
- **Your Fork**: https://github.com/jburnford/chandra-ocr
- **Documentation**: See README.md for full API documentation
- **Support**: Discord at https://discord.gg/KuZwXNGnfH

## License

- Code: Apache 2.0
- Model: Modified OpenRAIL-M (free for research, personal use, and startups under $2M)
- See LICENSE file for full details
