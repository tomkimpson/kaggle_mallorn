#!/bin/bash

#SBATCH --job-name=mallorn_cnn
#SBATCH --output=slurm/outputs/mallorn_%j.txt
#SBATCH --error=slurm/outputs/mallorn_%j.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create output directory
mkdir -p slurm/outputs

# UPDATE THIS PATH for your cluster
cd /path/to/kaggle_mallorn

# Activate virtual environment
source venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | $(date)"
nvidia-smi

# Train single fold for quick testing
time python3 scripts/train.py \
    --model cnn \
    --epochs 50 \
    --batch_size 64 \
    --fold 0 \
    --device cuda \
    --checkpoint_dir checkpoints/cnn_test

echo "Done: $(date)"
