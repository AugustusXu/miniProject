#!/usr/bin/env bash
set -euo pipefail

# Run OFT finetuning on IMDB with a compact default setup.
python src/train_oft_imdb.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs \
  --train_samples 2000 \
  --val_samples 400 \
  --max_length 384 \
  --learning_rate 2e-4 \
  --num_epochs 1 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --eval_batch_size 16 \
  --oft_r 8 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --bf16
