# Chandra OCR Quick Start Guide

## Current Status

✅ Repository set up: `git@github.com:jburnford/chandra-ocr.git`
✅ Code deployed to Nibi: `~/projects/def-jic823/chandra-ocr/`
🔄 Setup job running: Job ID 3756266 (installing dependencies)

## Monitoring Setup Progress

```bash
# Check if setup job is complete
ssh nibi squeue -u jic823

# View setup progress
ssh nibi cat ~/projects/def-jic823/chandra-ocr/chandra-setup-3756266.out

# When job completes (status PD changes to nothing), verify installation
ssh nibi "source ~/projects/def-jic823/chandra-ocr/.venv/bin/activate && chandra --help"
```

Setup typically takes 15-20 minutes.

## Once Setup Completes

### Test with a Single PDF

1. **Upload a test PDF to Nibi:**
```bash
scp test.pdf nibi:~/projects/def-jic823/test_pdfs/
```

2. **Submit a test job:**
```bash
ssh nibi "cd ~/projects/def-jic823/chandra-ocr && sbatch nibi_run_chandra.slurm ~/projects/def-jic823/test_pdfs/test.pdf ~/projects/def-jic823/chandra_output"
```

3. **Monitor the job:**
```bash
ssh nibi squeue -u jic823
ssh nibi tail -f ~/projects/def-jic823/chandra-ocr/chandra-ocr-*.out
```

4. **Download results:**
```bash
scp -r nibi:~/projects/def-jic823/chandra_output .
```

### Process Multiple PDFs

```bash
# Upload PDFs
scp *.pdf nibi:~/projects/def-jic823/pdfs/

# Submit batch job
ssh nibi "cd ~/projects/def-jic823/chandra-ocr && sbatch nibi_run_chandra.slurm ~/projects/def-jic823/pdfs/ ~/projects/def-jic823/batch_output"
```

### Advanced Options

**Process specific pages:**
```bash
ssh nibi "cd ~/projects/def-jic823/chandra-ocr && PAGE_RANGE='1-10' sbatch nibi_run_chandra.slurm input.pdf output/"
```

**Skip image extraction:**
```bash
ssh nibi "cd ~/projects/def-jic823/chandra-ocr && INCLUDE_IMAGES=false sbatch nibi_run_chandra.slurm input.pdf output/"
```

## Development Workflow

### Local Changes → Nibi

1. **Edit code locally** (in `/home/jic823/chandra/`)
2. **Commit and push:**
```bash
cd /home/jic823/chandra
git add .
git commit -m "Description of changes"
git push origin main
```

3. **Pull on Nibi:**
```bash
ssh nibi "cd ~/projects/def-jic823/chandra-ocr && git pull origin main"
```

### Nibi → Local (for logs/results)

```bash
# Download output
scp -r nibi:~/projects/def-jic823/chandra_output .

# Download logs
scp nibi:~/projects/def-jic823/chandra-ocr/chandra-ocr-*.out .
```

## Useful Commands

```bash
# Check all running jobs
ssh nibi squeue -u jic823

# Cancel a job
ssh nibi scancel <JOBID>

# View available disk space
ssh nibi "df -h ~/projects/def-jic823"

# List recent output files
ssh nibi "ls -lht ~/projects/def-jic823/chandra-ocr/*.out | head -5"

# Check GPU availability
ssh nibi "sinfo -p gpu -o '%P %.5a %.10l %.6D %.6t %.8z %.6m %.8d %.6w %.8f %20G'"
```

## File Locations

**Local (WSL):**
- Repository: `/home/jic823/chandra/`
- Scripts: `/home/jic823/chandra/nibi_*.slurm`

**Nibi Cluster:**
- Repository: `~/projects/def-jic823/chandra-ocr/`
- Virtual environment: `~/projects/def-jic823/chandra-ocr/.venv/`
- Job outputs: `~/projects/def-jic823/chandra-ocr/*.out`
- Test PDFs: `~/projects/def-jic823/test_pdfs/`
- Output: `~/projects/def-jic823/chandra_output/`

**GitHub:**
- Repository: https://github.com/jburnford/chandra-ocr
- Upstream: https://github.com/datalab-to/chandra

## Troubleshooting

**Setup job fails:**
```bash
# Check the error in the output file
ssh nibi cat ~/projects/def-jic823/chandra-ocr/chandra-setup-3756266.out

# Re-run setup manually on compute node
ssh nibi
salloc --cpus-per-task=4 --mem=16G --time=1:00:00
cd ~/projects/def-jic823/chandra-ocr
bash nibi_setup.sh
```

**CUDA not available:**
```bash
# Verify GPU allocation
ssh nibi squeue -u jic823

# Check CUDA in running job
ssh nibi "module load python/3.12 cuda/12.1 && python3 -c 'import torch; print(torch.cuda.is_available())'"
```

**Out of memory:**
- Edit `nibi_run_chandra.slurm` and increase `--mem=` value
- Or reduce batch size: `BATCH_SIZE=1 sbatch ...`

## Next Steps

After setup completes:
1. Test with a sample PDF
2. Benchmark performance on your historical documents
3. Compare with OLMoCR results
4. Scale up to larger batches

For detailed documentation, see `NIBI_DEPLOYMENT.md`.
