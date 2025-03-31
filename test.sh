#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict_gnn
#SBATCH --time=1:30:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_kelvin_tuong
#SBATCH -e run.error
#SBATCH -o run.out

source ../gnn_env/bin/activate
python test.py --dataset-path /scratch/project/tcr_ml/gnn_release/test_data_v2/phs002517/processed --model-path /scratch/project/tcr_ml/gnn_release/model_2025_360