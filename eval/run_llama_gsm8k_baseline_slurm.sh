#!/bin/bash
#SBATCH --job-name=llama-gsm8k-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2h100
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/llama_gsm8k_baseline_%j.out
#SBATCH --error=logs/llama_gsm8k_baseline_%j.err

source ~/.bashrc
conda activate llm-security
cd ~/ML-Research
set -a
source .env
set +a
mkdir -p logs "${RESULTS_DIR}/llama_gsm8k_baseline"

lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
  --tasks gsm8k \
  --device cuda \
  --batch_size 4 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs "temperature=0,do_sample=False" \
  --output_path "${RESULTS_DIR}/llama_gsm8k_baseline" \
  --log_samples
