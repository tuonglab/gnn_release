#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict_gnn
#SBATCH --time=0:45:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_kelvin_tuong
#SBATCH -e run.error
#SBATCH -o run.out

# Activate Python environment
source ../gnn_env/bin/activate

# === Set user-defined values ===
MODEL_PATH="/scratch/project/tcr_ml/gnn_release/model_2025_ccdi_only"
DATASET_NAME="sarcoma_zero"
DATASET_PATH="/scratch/project/tcr_ml/gnn_release/test_data_v2/${DATASET_NAME}/processed"
SCORES_DIR="${MODEL_PATH}/scores"
RENAMED_SCORES_DIR="${MODEL_PATH}/${DATASET_NAME}_scores"

# === Run pipeline ===

# Run prediction
python test.py --dataset-path "$DATASET_PATH" --model-path "$MODEL_PATH"

# Run scoring and plotting scripts
python scoring.py --directory "$SCORES_DIR"
python plot.py --directory "$SCORES_DIR"

# Rename the scores directory
mv "$SCORES_DIR" "$RENAMED_SCORES_DIR"
