#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --job-name=edge_gen
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --account=a_kelvin_tuong
#SBATCH -e edge_gen.error
#SBATCH -o edge_gen.out

# Define shared dataset ID
DATASET_ID="tower_predictions"

# Define base paths
ENV_PATH="/scratch/project/tcr_ml/gnn_env/bin/activate"
SCRIPT_PATH="/scratch/project/tcr_ml/gnn_release/graph_generation/create_edgelist.py"
INPUT_BASE="/QRISdata/Q7753"
OUTPUT_BASE="/scratch/project/tcr_ml/gnn_release/dataset_boltz"

# Construct input/output directories using the dataset ID
INPUT_DIR="${INPUT_BASE}/${DATASET_ID}"
OUTPUT_DIR="${OUTPUT_BASE}/${DATASET_ID}/raw"

# Activate the virtual environment
source "$ENV_PATH"

# Run the edge list generation script
python "$SCRIPT_PATH" \
    --tar-dir "$INPUT_DIR" \
    --output-base-dir "$OUTPUT_DIR"
