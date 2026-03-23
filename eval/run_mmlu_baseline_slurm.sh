#!/bin/bash

#SBATCH --job-name=mmlu-baseline

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --constraint=gpu4v100

#SBATCH --mem=32G

#SBATCH --time=04:00:00

#SBATCH --output=logs/mmlu_baseline_%j.out

#SBATCH --error=logs/mmlu_baseline_%j.err



source ~/.bashrc

conda activate llm-security

cd ~/ML-Research

source .env

mkdir -p logs "${RESULTS_DIR}/mmlu_baseline"



lm_eval \

  --model hf \

  --model_args pretrained="${MODEL_NAME}" \

  --tasks mmlu \

  --device cuda \

  --batch_size "${BATCH_SIZE}" \

  --apply_chat_template \

  --fewshot_as_multiturn \

  --output_path "${RESULTS_DIR}/mmlu_baseline" \

  --log_samples

