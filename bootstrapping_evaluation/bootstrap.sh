#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --job-name=bootstrap
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account=a_kelvin_tuong
#SBATCH -e bootstrap.error
#SBATCH -o bootstrap_isacs_only_2.out
#SBATCH --qos=gpu

source ../../gnn_env/bin/activate
export PYTHONPATH='/scratch/project/tcr_ml/gnn_release'
module load cuda/12.1
python bootstrap_model.py # reminder to change model path if neccessary