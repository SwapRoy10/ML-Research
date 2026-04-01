#!/bin/bash
#SBATCH --job-name=mmlu-gcg-transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu4v100|gpul40s|gpu2v100"
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mmlu_gcg_transfer_%j.out
#SBATCH --error=logs/mmlu_gcg_transfer_%j.err

source ~/.bashrc
conda activate llm-security
cd ~/ML-Research
source .env

mkdir -p logs "${RESULTS_DIR}/mmlu_gcg_transfer"

GCG_RESULT="results/gcg/mmlu_sabotage_elementary_mathematics_0.json"

if [ ! -f "$GCG_RESULT" ]; then
    echo "ERROR: GCG result file not found: $GCG_RESULT"
    echo "Run attacks/gcg/mmlu_sabotage.py first."
    exit 1
fi

GCG_SUFFIX=$(python3 -c "
import json
with open('${GCG_RESULT}') as f:
    r = json.load(f)
print(r['final_suffix'])
")

echo "=== MMLU GCG Transfer Eval ==="
echo "Model:      ${MODEL_NAME}"
echo "GCG result: ${GCG_RESULT}"
echo "Suffix:     ${GCG_SUFFIX}"
echo ""

lm_eval \
  --model hf \
  --model_args pretrained="${MODEL_NAME}" \
  --tasks mmlu \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --system_instruction "${GCG_SUFFIX}" \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path "${RESULTS_DIR}/mmlu_gcg_transfer" \
  --log_samples

echo ""
echo "Done. Results in ${RESULTS_DIR}/mmlu_gcg_transfer"
