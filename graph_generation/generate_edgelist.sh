#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --job-name=edge_gen
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --account=a_kelvin_tuong
#SBATCH -e edge_gen.error
#SBATCH -o edge_gen.out

# Activate the virtual environment
source /scratch/project/tcr_ml/gnn_env/bin/activate

# Run the edge list generation script with actual paths
python /scratch/project/tcr_ml/GNN/graph_generation/create_edgelist.py \
    --root-dir reference_data/control \
    --directory /scratch/project/tcr_ml/GNN/reference_data/control/raw
