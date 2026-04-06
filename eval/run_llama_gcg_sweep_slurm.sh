#!/bin/bash

#SBATCH --job-name=llama-gcg

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --constraint=gpul40s

#SBATCH --mem=48G

#SBATCH --time=04:00:00

#SBATCH --array=0-7

#SBATCH --output=logs/llama_gcg_%A_%a.out

#SBATCH --error=logs/llama_gcg_%A_%a.err



source ~/.bashrc

conda activate llm-security

cd ~/ML-Research

set -a

source .env

set +a



export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

export GCG_STEPS="${GCG_STEPS:-100}"

export GCG_SUFFIX_LEN="${GCG_SUFFIX_LEN:-20}"

export GCG_TOPK="${GCG_TOPK:-8}"



subjects=(

  "college_mathematics"

  "elementary_mathematics"

  "elementary_mathematics"

  "elementary_mathematics"

  "high_school_biology"

  "high_school_mathematics"

  "high_school_physics"

  "moral_scenarios"

)



indices=(0 0 1 2 0 0 0 0)



export MMLU_SUBJECT="${subjects[$SLURM_ARRAY_TASK_ID]}"

export MMLU_INDEX="${indices[$SLURM_ARRAY_TASK_ID]}"



mkdir -p logs "${RESULTS_DIR}/gcg"



python attacks/gcg/mmlu_sabotage.py
