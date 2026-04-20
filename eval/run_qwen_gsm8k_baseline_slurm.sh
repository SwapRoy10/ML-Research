#!/bin/bash
#SBATCH --job-name=qwen-gsm8k-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodelist=dgxt001
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/qwen_gsm8k_baseline_%j.out
#SBATCH --error=logs/qwen_gsm8k_baseline_%j.err

source ~/.bashrc
module load CUDA/12.1.1
module load GCCcore/12.3.0
conda activate llm-security
cd ~/ML-Research
set -a
source .env
set +a
mkdir -p logs "${RESULTS_DIR}/qwen_gsm8k_baseline"

lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen1.5-MoE-A2.7B-Chat,trust_remote_code=True \
  --tasks gsm8k \
  --device cuda \
  --batch_size 4 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs "temperature=0,do_sample=False" \
  --output_path "${RESULTS_DIR}/qwen_gsm8k_baseline" \
  --log_samples
