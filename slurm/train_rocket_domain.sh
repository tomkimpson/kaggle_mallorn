#!/bin/bash

#SBATCH --job-name=rocket_domain
#SBATCH --output=slurm/outputs/rocket_domain_%j.txt
#SBATCH --error=slurm/outputs/rocket_domain_%j.err
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Navigate to project directory
cd /fred/oz022/tkimpson/kaggle_mallorn

# Activate virtual environment
source venv/bin/activate

echo "============================================"
echo "ROCKET + Domain Features + LightGBM Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

# Check Python
python3 -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')"

# Step 1: Train with domain features (no Optuna first for baseline)
echo "============================================"
echo "Step 1: ROCKET 5K + Domain Features (baseline)"
echo "============================================"

time python3 scripts/train_rocket_lgbm.py \
    --num_kernels 5000 \
    --grid_points 100 \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --use_domain_features \
    --save_features \
    --checkpoint_dir checkpoints/rocket_domain_5k \
    --output submission_rocket_domain_5k.csv

echo "============================================"
echo "Step 1 complete!"
echo "============================================"

# Step 2: Optuna hyperparameter tuning with domain features
echo ""
echo "============================================"
echo "Step 2: ROCKET 5K + Domain Features + Optuna (100 trials)"
echo "============================================"

time python3 scripts/train_rocket_lgbm.py \
    --num_kernels 5000 \
    --grid_points 100 \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --use_domain_features \
    --optuna \
    --optuna_trials 100 \
    --checkpoint_dir checkpoints/rocket_domain_optuna \
    --output submission_rocket_domain_optuna.csv

echo "============================================"
echo "All training complete!"
echo "End time: $(date)"
echo "============================================"

# Print summary
echo ""
echo "Generated submissions:"
ls -la submission_rocket_domain*.csv

echo ""
echo "Model checkpoints:"
du -sh checkpoints/rocket_domain*/

echo ""
echo "CV Results (baseline with domain features):"
cat checkpoints/rocket_domain_5k/cv_summary.yaml | head -30

echo ""
echo "CV Results (with Optuna tuning):"
cat checkpoints/rocket_domain_optuna/cv_summary.yaml | head -30

echo ""
echo "Best Optuna parameters:"
cat checkpoints/rocket_domain_optuna/optuna_best_params.yaml
