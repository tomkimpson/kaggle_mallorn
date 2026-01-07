#!/bin/bash

#SBATCH --job-name=no_rocket
#SBATCH --output=slurm/outputs/no_rocket_%j.txt
#SBATCH --error=slurm/outputs/no_rocket_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to project directory
cd /fred/oz022/tkimpson/kaggle_mallorn

# Activate virtual environment
source venv/bin/activate

echo "============================================"
echo "Domain + GP/DRW Only Training (No ROCKET)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "============================================"
echo "Purpose: Stacking diversity model"
echo "Features: Domain (~168) + GP/DRW (~69)"
echo "============================================"

# Run training
time python3 scripts/train_domain_gpdrw_only.py \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --use_domain_features \
    --use_gp_drw_features \
    --gp_drw_bands g,r,i \
    --checkpoint_dir checkpoints/no_rocket \
    --output submission_no_rocket.csv

echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"

# Print results
echo ""
echo "CV Summary:"
cat checkpoints/no_rocket/cv_summary.yaml | head -40 2>/dev/null || echo "No summary found"
