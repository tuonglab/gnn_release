#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=edge
#SBATCH --time=00:10:00
#SBATCH --partition=general
#SBATCH --account=a_kelvin_tuong
#SBATCH -e pytorch.error
#SBATCH -o pytorch.out

# Define shared dataset ID
DATASET_ID="20241106_WGS_20241106_sc_PICA0033-PICA0069_Pool_6"

# Define base paths
ENV_PATH="/scratch/project/tcr_ml/gnn_env/bin/activate"
SCRIPT_PATH="/scratch/project/tcr_ml/gnn_release/graph_generation/process.py"
BASE_DIR="/scratch/project/tcr_ml/gnn_release/test_data_v2"

# Construct full paths
ROOT_DIR="${BASE_DIR}/${DATASET_ID}"
RAW_DIR="${ROOT_DIR}/raw"

# Set flag for cancer classification (set to "--cancer" or leave empty "")
CANCER_FLAG=""  # Set to "" if not cancer dataset

# Activate the virtual environment
source "$ENV_PATH"

# Run the process script
python "$SCRIPT_PATH" \
    --root-dir "$ROOT_DIR" \
    --directory "$RAW_DIR" \
    $CANCER_FLAG
