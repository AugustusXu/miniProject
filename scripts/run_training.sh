#!/bin/bash
# 遇到错误立即退出
set -e
# 【关键修复】确保管道命令(| tee)中如果左边报错，整个脚本也会停止
set -o pipefail 

mkdir -p src data boft_sd_weights report

echo "================================================="
echo "1. 检查训练数据..."
echo "================================================="
if [ ! -d "./data/sks_dog" ] || [ -z "$(ls -A ./data/sks_dog 2>/dev/null)" ]; then
    echo "❌ 错误: 找不到训练数据！"
    echo "👉 请先执行命令: python src/prepare_data.py"
    exit 1
fi

echo "================================================="
echo "2. 下载官方训练脚本并修复 API 兼容性..."
echo "================================================="
#wget -q -O src/train_dreambooth.py https://mirror.ghproxy.com/https://raw.githubusercontent.com/huggingface/peft/main/examples/boft_dreambooth/train_dreambooth.py

if [ ! -s src/train_dreambooth.py ]; then
    echo "❌ 错误: 训练代码下载失败，请检查服务器网络。"
    exit 1
fi

# 🚨 核心修复：自动拦截并清理被 HuggingFace 弃用的 Repository API
sed -i 's/from huggingface_hub import Repository, create_repo/from huggingface_hub import create_repo/g' src/train_dreambooth.py
sed -i 's/from huggingface_hub import Repository//g' src/train_dreambooth.py



echo "✅ 官方脚本兼容性补丁打补丁成功！"

export HF_ENDPOINT="https://hf-mirror.com"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/sks_dog"
export OUTPUT_DIR="./boft_sd_weights"

echo "================================================="
echo "3. 🚀 开始在 DGX Spark 上进行 BOFT 微调 (强制单卡)..."
echo "================================================="
echo "⚠️ 提示：检测到极新的 NVIDIA GB10 (Blackwell) 架构，如果启动时稍微卡顿是在进行 JIT 编译，请耐心等待。"

# 调整了 accelerate 参数位置，消除了烦人的 warning
# 新增了 --validation_prompt 解决 NoneType 报错
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision="fp16" \
  --dynamo_backend="no" \
  src/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --validation_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --boft_block_size=4 \
  --boft_n_butterfly_factor=1 \
  --seed=42 2>&1 | tee train.log
  --report_to="none" \

echo "================================================="
echo "🎉 训练真正完成！模型权重已保存至 $OUTPUT_DIR"
echo "================================================="