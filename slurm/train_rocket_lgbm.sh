#!/bin/bash

#SBATCH --job-name=rocket_lgbm
#SBATCH --output=slurm/outputs/rocket_lgbm_%j.txt
#SBATCH --error=slurm/outputs/rocket_lgbm_%j.err
#SBATCH --export=ALL
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Navigate to project directory
cd /fred/oz022/tkimpson/kaggle_mallorn

# Activate virtual environment
source venv/bin/activate

echo "============================================"
echo "ROCKET + LightGBM Training (Phase 1)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

# Check Python
python3 -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')"

echo "============================================"
echo "Training ROCKET + LightGBM with 5000 kernels..."
echo "============================================"

# Train ROCKET + LightGBM with 5000 kernels per band
time python3 scripts/train_rocket_lgbm.py \
    --num_kernels 5000 \
    --grid_points 100 \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --learning_rate 0.05 \
    --num_leaves 31 \
    --save_features \
    --checkpoint_dir checkpoints/rocket_lgbm_5k \
    --output submission_rocket_lgbm_5k.csv

echo "============================================"
echo "5000 kernels complete!"
echo "============================================"

# Also try with 2000 kernels for comparison (faster)
echo "Training with 2000 kernels for comparison..."
time python3 scripts/train_rocket_lgbm.py \
    --num_kernels 2000 \
    --grid_points 100 \
    --n_folds 5 \
    --num_boost_round 1000 \
    --early_stopping_rounds 50 \
    --save_features \
    --checkpoint_dir checkpoints/rocket_lgbm_2k \
    --output submission_rocket_lgbm_2k.csv

echo "============================================"
echo "All training complete!"
echo "End time: $(date)"
echo "============================================"

# Print summary
echo ""
echo "Generated submissions:"
ls -la submission_rocket_lgbm*.csv

echo ""
echo "Model checkpoints:"
du -sh checkpoints/rocket_lgbm*/

echo ""
echo "CV Results:"
cat checkpoints/rocket_lgbm_5k/cv_summary.yaml | head -20
