#!/bin/bash
# in project's root directory: bash scripts/run_training.sh

mkdir -p src data boft_sd_weights report

echo "Downloading official PEFT BOFT script to src/ ..."
wget -q -O src/train_dreambooth.py https://raw.githubusercontent.com/huggingface/peft/main/examples/boft_dreambooth/train_dreambooth.py

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/sks_dog"
export OUTPUT_DIR="./boft_sd_weights"

echo "Starting BOFT Training on NVIDIA DGX Spark..."

# use accelerate to start src/train_dreambooth.py with the specified parameters and log output to train.log
accelerate launch src/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --boft_block_size=4 \
  --boft_n_butterfly_factor=1 \
  --seed=42 \
  --mixed_precision="fp16" > train.log 2>&1

echo "Training completed! Model weights saved to $OUTPUT_DIR"