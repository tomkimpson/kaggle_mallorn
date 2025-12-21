#!/bin/bash

#SBATCH --job-name=mallorn_train
#SBATCH --output=slurm/outputs/mallorn_%j.txt
#SBATCH --error=slurm/outputs/mallorn_%j.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Navigate to project directory
cd /Users/tkimpson/projects/kaggle_mallorn

# Activate virtual environment
source venv/bin/activate

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

# Check GPU
nvidia-smi

# Check Python and PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "============================================"
echo "Starting MALLORN training..."
echo "============================================"

# Run training - CNN model with 5-fold CV
time python3 scripts/train.py \
    --model cnn \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --patience 20 \
    --loss focal \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    --max_seq_len 200 \
    --n_folds 5 \
    --num_workers 8 \
    --device cuda \
    --checkpoint_dir checkpoints/cnn

echo "============================================"
echo "CNN training complete. Starting Transformer..."
echo "============================================"

# Run training - Transformer model with 5-fold CV
time python3 scripts/train.py \
    --model transformer \
    --epochs 100 \
    --batch_size 64 \
    --lr 5e-4 \
    --patience 20 \
    --loss focal \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    --max_seq_len 200 \
    --n_folds 5 \
    --num_workers 8 \
    --device cuda \
    --checkpoint_dir checkpoints/transformer

echo "============================================"
echo "All training complete!"
echo "End time: $(date)"
echo "============================================"
