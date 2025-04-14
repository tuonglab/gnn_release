#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --job-name=test_uncertainty
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account=a_kelvin_tuong
#SBATCH -e test_uncertainty.error
#SBATCH -o test_uncertainty.out
#SBATCH --qos=gpu

source ../../gnn_env/bin/activate
module load cuda/12.1
export PYTHONPATH='/scratch/project/tcr_ml/gnn_release'
python evaluate_uncertainty.py
