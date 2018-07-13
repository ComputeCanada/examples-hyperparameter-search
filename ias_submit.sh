#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=0-00:05

source ./venv/bin/activate
srun python ias.py
