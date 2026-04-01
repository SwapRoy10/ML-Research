#!/bin/bash

#SBATCH --job-name=mmlu-gcg

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --constraint=gpul40s

#SBATCH --mem=32G

#SBATCH --time=04:00:00

#SBATCH --output=logs/mmlu_gcg_%j.out

#SBATCH --error=logs/mmlu_gcg_%j.err

source ~/.bashrc

conda activate llm-security

cd ~/ML-Research

source .env

mkdir -p logs "${RESULTS_DIR}/mmlu_gcg"

ATTACK_SUFFIX="Marg ! ! Bh ! ! ! ! ! ! ! ! ! dd ! ! ! ! ! !"

ATTACK_SYSTEM="You are a helpful assistant. ${ATTACK_SUFFIX}"

lm_eval run --model hf --model_args pretrained="${MODEL_NAME}" --tasks mmlu --device cuda --batch_size "${BATCH_SIZE}" --system_instruction "${ATTACK_SYSTEM}" --apply_chat_template --fewshot_as_multiturn --output_path "${RESULTS_DIR}/mmlu_gcg" --log_samples

