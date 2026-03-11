# AIST5030 GAI Mini-Project

## Parameter-efficient Finetuning with OFT 

Author: XU Kai (1155239333@link.cuhk.edu.hk)

This repository contains the impletementation of fine-tuing for Subject-driven generation (DreamBooth):

- **Method:** Block-Diagonal Orthogonal Finetuning (BOFT)
- **Hardware:** NVIDIA DGX Spark
- **Base Model:** Stable Diffusion v1.5
- **Task:** Injecting a custom subject ("sks dog") into the diffusion model via BOFT.

## Quick Start
**Note: Please run all commands from the root directory of this project.**

1. Install dependencies: `pip install diffusers transformers accelerate peft datasets huggingface_hub matplotlib jupyter`
2. Prepare Data: `python src/prepare_data.py`
3. Run Training: `bash scripts/run_training.sh`
4. Plot Loss Curve: `python src/plot_loss.py`
5. Visualization: Open `demo/demo.ipynb` to generate images and compare the qualitative results.


## 1. Environment

- Python: `3.12`
- PyTorch: `2.10`
- CUDA: `13.0`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Project Structure

```text
.
├── README.md                           
├── scripts/
│   └── run_training.sh                 # BOFT script
├── src/
│   ├── prepare_data.py                 # dataset download
│   └── plot_loss.py                    # recrod logs and plot loss
├── demo/
│   └── demo.ipynb                      # demoJupyter Notebook
└── report/
    └── Project_Report.md               # Project Report
```

