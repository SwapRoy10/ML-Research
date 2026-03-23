#!/bin/bash
#SBATCH --job-name=gcg-attack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu4v100
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gcg_%j.out
#SBATCH --error=logs/gcg_%j.err

source ~/.bashrc
conda activate llm-security
cd ~/ML-Research
source .env

python attacks/gcg/gcg_attack.py
