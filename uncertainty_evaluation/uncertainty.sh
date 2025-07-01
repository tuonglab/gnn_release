#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --job-name=train_uncertainty
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account=a_kelvin_tuong
#SBATCH -e train.error
#SBATCH -o train.out
#SBATCH --qos=gpu

source /scratch/project/tcr_ml/gnn_env/bin/activate
module load cuda/12.1
python train_uncertainity.py
# python train_model.py # reminder to change model path if neccessary
# python plot.py # change in accordance with the model path in train.py
# python scoring.py # change in accordance with model parth in train.py