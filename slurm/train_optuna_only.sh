#!/bin/bash

#SBATCH --job-name=optuna_tune
#SBATCH --output=slurm/outputs/optuna_tune_%j.txt
#SBATCH --error=slurm/outputs/optuna_tune_%j.err
#SBATCH --export=ALL
#SBATCH --time=30:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to project directory
cd /fred/oz022/tkimpson/kaggle_mallorn

# Activate virtual environment
source venv/bin/activate

echo "============================================"
echo "Optuna Hyperparameter Tuning (using cached features)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

# Run Optuna with pre-computed features (skips 72-min feature extraction)
time python3 scripts/train_rocket_lgbm.py \
    --num_kernels 5000 \
    --grid_points 100 \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --use_domain_features \
    --load_features checkpoints/rocket_domain_5k/train_features.npz \
    --optuna \
    --optuna_trials 20 \
    --checkpoint_dir checkpoints/rocket_domain_optuna \
    --output submission_rocket_domain_optuna.csv

echo "============================================"
echo "Optuna tuning complete!"
echo "End time: $(date)"
echo "============================================"

# Print results
echo ""
echo "Best parameters:"
cat checkpoints/rocket_domain_optuna/optuna_best_params.yaml 2>/dev/null || echo "No params file found"

echo ""
echo "CV Results:"
cat checkpoints/rocket_domain_optuna/cv_summary.yaml | head -50 2>/dev/null || echo "No summary found"
