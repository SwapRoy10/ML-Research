#!/bin/bash
#SBATCH --job-name=gcg_mmlu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gcg_results_%j.log

# Load environment
source ~/.bashrc
conda activate llm-security

# Run the sabotage script
python attacks/gcg/mmlu_sabotage.py
