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

# Activate the virtual environment
source /scratch/project/tcr_ml/gnn_env/bin/activate

# Run the process.py script with actual arguments
python /scratch/project/tcr_ml/gnn_release/graph_generation/process.py \
    --root-dir  /scratch/project/tcr_ml/gnn_release/test_data_v2/20240918_WGS_20240924_sc_PICA0008-PICA0032_Pool_8 \
    --directory /scratch/project/tcr_ml/gnn_release/test_data_v2/20240918_WGS_20240924_sc_PICA0008-PICA0032_Pool_8/raw \

# add/remove cancer flag depending on the dataset class type
