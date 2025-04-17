#!/bin/bash
#SBATCH --partition=gpu-tesla
#SBATCH -n 10
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=MLPhysHW1
#SBATCH --output=MLPhysHW1.out
#SBATCH --error=MLPhysHW1.err

python3 main.py