#!/bin/bash
#SBATCH --job-name=llama-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2h100
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/llama_baseline_%j.out
#SBATCH --error=logs/llama_baseline_%j.err

source ~/.bashrc
conda activate llm-security
cd ~/ML-Research
set -a
source .env
set +a
mkdir -p logs "${RESULTS_DIR}/llama_baseline"

lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
  --tasks hellaswag,arc_easy,truthfulqa_mc1 \
  --device cuda \
  --batch_size 4 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path ${RESULTS_DIR}/llama_baseline \
  --log_samples
