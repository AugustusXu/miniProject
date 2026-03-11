# BOFT DreamBooth: Subject-Driven Image Generation

**Mini-Project: Parameter-Efficient Finetuning for Pretrained Foundation Models**

This project demonstrates using **BOFT** (Butterfly Orthogonal Fine-Tuning) to finetune a pretrained Stable Diffusion model for subject-driven image generation via DreamBooth.

## Method

BOFT is a parameter-efficient finetuning method from the orthogonal finetuning family. It inserts trainable orthogonal matrices with **butterfly factorization** structure into the attention layers of the UNet. Only these small adapter matrices are trained while all pretrained weights remain frozen.

Key advantages:
- **Parameter-efficient**: Only ~0.8% of parameters are trainable
- **Orthogonal constraint**: Preserves the hyperspherical energy of pretrained representations
- **Full-rank updates**: Unlike LoRA, BOFT supports full-rank orthogonal transformations

References:
- [BOFT Paper (ICLR 2024)](https://arxiv.org/abs/2311.06243)
- [OFT Paper](https://arxiv.org/abs/2306.07280)
- [HuggingFace PEFT Library](https://huggingface.co/docs/peft)

## Project Structure

```
miniProject/
├── demo.ipynb              # Main notebook: training + visualization
├── train_dreambooth.py     # Standalone training script (accelerate-based)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── utils/
    ├── __init__.py
    ├── args_loader.py      # Argument parser for CLI training
    ├── dataset.py          # DreamBooth dataset class
    └── tracemalloc.py      # Memory tracking utilities
```

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Hardware Requirements

- GPU with at least 8GB VRAM (tested on NVIDIA GPUs with CUDA)
- For CPU-only systems, reduce resolution and training steps

## Usage

### Option 1: Jupyter Notebook (Recommended)

Run `demo.ipynb` for an interactive experience with full visualization:

```bash
jupyter notebook demo.ipynb
```

The notebook includes:
1. Dataset download and visualization
2. Baseline image generation (before finetuning)
3. BOFT finetuning with training loss logging
4. Training loss curve visualization
5. Post-finetuning image generation
6. Before vs. after qualitative comparison
7. Multi-prompt generation gallery

### Option 2: Command-Line Training

For training via command line with `accelerate`:

```bash
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --instance_data_dir="./data/dreambooth/dataset/dog" \
  --class_data_dir="./data/class_data/dog" \
  --output_dir="./data/output/boft" \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --num_dataloader_workers=0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --use_boft \
  --boft_block_num=8 \
  --boft_block_size=0 \
  --boft_n_butterfly_factor=1 \
  --boft_dropout=0.1 \
  --boft_bias="boft_only" \
  --learning_rate=3e-5 \
  --max_train_steps=800 \
  --checkpointing_steps=200 \
  --no_tracemalloc \
  --report_to="wandb"
```

> **Note for Windows**: Set `--num_dataloader_workers=0` and add `--no_tracemalloc`.

## BOFT Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `boft_block_num` | 8 | Number of orthogonal blocks |
| `boft_block_size` | 0 | Auto-determined from block_num |
| `boft_n_butterfly_factor` | 1 | Butterfly factors (1 = vanilla OFT) |
| `boft_dropout` | 0.1 | Multiplicative dropout rate |
| `bias` | boft_only | Only train BOFT bias parameters |
| `target_modules` | to_q, to_v, to_k, to_out.0 | UNet attention modules |

## Dataset

We use the **dog** subject from the [Google DreamBooth dataset](https://github.com/google/dreambooth), which contains 5 images of a specific dog. The dataset is automatically downloaded when running the notebook.

## Results

After training, the model generates images of the specific dog subject in various contexts while maintaining the subject's identity. See `demo.ipynb` for full results including:

- Training loss curves
- Before vs. after finetuning comparisons
- Multi-prompt generation gallery
