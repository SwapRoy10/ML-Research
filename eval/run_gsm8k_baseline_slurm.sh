#!/bin/bash
#SBATCH --job-name=gsm8k-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpul40s
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gsm8k_baseline_%j.out
#SBATCH --error=logs/gsm8k_baseline_%j.err

source ~/.bashrc
conda activate llm-security
cd ~/ML-Research
set -a
source .env
set +a
mkdir -p logs "${RESULTS_DIR}/gsm8k_baseline"

lm_eval \
  --model hf \
  --model_args pretrained="${MODEL_NAME}" \
  --tasks gsm8k \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs "temperature=0,do_sample=False" \
  --output_path "${RESULTS_DIR}/gsm8k_baseline" \
  --log_samples
