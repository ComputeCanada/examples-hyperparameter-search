#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --array=0-3

source ./venv/bin/activate
python array_job.py
