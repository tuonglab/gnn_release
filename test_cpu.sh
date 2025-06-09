#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=predict_gnn
#SBATCH --time=04:20:00
#SBATCH --partition=general
#SBATCH --account=a_kelvin_tuong
#SBATCH -e test_cpu.error
#SBATCH -o tets_cpu.out

# Activate environment
source ../gnn_env/bin/activate

# Define paths and names
MODEL_PATH="/scratch/project/tcr_ml/gnn_release/model_2025_sc"
DATASET_NAME="val_control"
DATASET_PATH="/scratch/project/tcr_ml/gnn_release/test_data_v2/${DATASET_NAME}/processed"
SCORES_DIR="${MODEL_PATH}/${DATASET_NAME}_scores"

# Run test script (with dataset name so scores folder is named correctly)
python test.py --dataset-path "$DATASET_PATH" --model-path "$MODEL_PATH" --dataset-name "$DATASET_NAME"

# Run scoring and plotting
python scoring.py --directory "$SCORES_DIR"
python plot.py --directory "$SCORES_DIR"
