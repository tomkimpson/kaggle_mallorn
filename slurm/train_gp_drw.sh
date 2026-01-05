#!/bin/bash

#SBATCH --job-name=gp_drw
#SBATCH --output=slurm/outputs/gp_drw_%j.txt
#SBATCH --error=slurm/outputs/gp_drw_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to worktree directory
cd /fred/oz022/tkimpson/mallorn_worktrees/exp-gp-drw

# Activate virtual environment from main repo
source /fred/oz022/tkimpson/kaggle_mallorn/venv/bin/activate

echo "============================================"
echo "ROCKET + GP/DRW Features Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "============================================"
echo "This experiment exploits how MALLORN was generated:"
echo "  - GP (Matern kernel) for transients"
echo "  - DRW for AGN variability"
echo "  - Likelihood ratio as discriminative feature"
echo "============================================"

# Run training with GP/DRW features
time python3 scripts/train_rocket_lgbm_gp_drw.py \
    --num_kernels 5000 \
    --grid_points 100 \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --use_domain_features \
    --use_gp_drw_features \
    --gp_drw_bands g,r,i \
    --save_gp_drw_features \
    --load_features /fred/oz022/tkimpson/kaggle_mallorn/checkpoints/rocket_domain_5k/train_features.npz \
    --checkpoint_dir checkpoints/gp_drw \
    --output submission_gp_drw.csv

echo "============================================"
echo "GP/DRW training complete!"
echo "End time: $(date)"
echo "============================================"

# Print results
echo ""
echo "CV Summary:"
cat checkpoints/gp_drw/cv_summary.yaml | head -40 2>/dev/null || echo "No summary found"
