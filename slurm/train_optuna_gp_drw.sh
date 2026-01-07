#!/bin/bash

#SBATCH --job-name=optuna_gp_drw
#SBATCH --output=slurm/outputs/optuna_gp_drw_%j.txt
#SBATCH --error=slurm/outputs/optuna_gp_drw_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Navigate to project directory
cd /fred/oz022/tkimpson/kaggle_mallorn

# Activate virtual environment
source venv/bin/activate

echo "============================================"
echo "Optuna Hyperparameter Tuning for GP/DRW Model"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "============================================"
echo "Objective: OOF-thresholded F1 (per next_steps_v3.md)"
echo "Search space:"
echo "  - num_leaves: 15-127"
echo "  - learning_rate: 0.01-0.2 (log)"
echo "  - feature_fraction: 0.5-0.9"
echo "  - bagging_fraction: 0.5-0.9"
echo "  - min_child_samples: 20-150"
echo "  - boosting_type: gbdt or dart"
echo "============================================"

# Run training with Optuna
time python3 scripts/train_rocket_lgbm_gp_drw.py \
    --num_kernels 5000 \
    --grid_points 100 \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --use_domain_features \
    --use_gp_drw_features \
    --gp_drw_bands g,r,i \
    --load_features checkpoints/rocket_domain_5k/train_features.npz \
    --optuna \
    --n_trials 50 \
    --checkpoint_dir checkpoints/gp_drw_v2_optuna \
    --output submission_gp_drw_v2_optuna.csv

echo "============================================"
echo "Optuna tuning complete!"
echo "End time: $(date)"
echo "============================================"

# Print results
echo ""
echo "Best Optuna Parameters:"
cat checkpoints/gp_drw_v2_optuna/optuna_best_params.yaml 2>/dev/null || echo "No params found"

echo ""
echo "CV Summary:"
cat checkpoints/gp_drw_v2_optuna/cv_summary.yaml | head -50 2>/dev/null || echo "No summary found"
