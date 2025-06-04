#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --job-name=gnn_train
#SBATCH --time=6:00:00
#SBATCH --account=a_kelvin_tuong
#SBATCH -e train_cpu.error
#SBATCH -o train_cpu.out
#SBATCH --qos=normal

source ../gnn_env/bin/activate
python train_cpu.py
# python train_umap.py # change in accordance with the model path in train.py
# python plot.py # change in accordance with the model path in train.py
# python scoring.py # change in accordance with model parth in train.py