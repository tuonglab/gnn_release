#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=edge
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --account=a_kelvin_tuong
#SBATCH -e pytorch.error
#SBATCH -o pytorch.out

source /scratch/project/tcr_ml/gnn_env/bin/activate
python process.py
# module load biopython
# python /scratch/project/tcr_ml/GNN/create_edgelist.py
