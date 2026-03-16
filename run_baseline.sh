#!/bin/bash
source .env

lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_NAME \
  --tasks hellaswag,arc_easy,mmlu,truthfulqa_mc1 \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --output_path $RESULTS_DIR/baseline \
  --log_samples

echo "Baseline eval complete. Results in $RESULTS_DIR/baseline"
