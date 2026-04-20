#!/bin/bash
#SBATCH --job-name=qwen-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodelist=dgxt001
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/qwen_baseline_%j.out
#SBATCH --error=logs/qwen_baseline_%j.err

source ~/.bashrc
module load CUDA/12.1.1
module load GCCcore/12.3.0
conda activate llm-security
cd ~/ML-Research
set -a
source .env
set +a
mkdir -p logs "${RESULTS_DIR}/qwen_baseline"

lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen1.5-MoE-A2.7B-Chat,trust_remote_code=True \
  --tasks hellaswag,arc_easy,truthfulqa_mc1 \
  --device cuda \
  --batch_size 4 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path "${RESULTS_DIR}/qwen_baseline" \
  --log_samples
