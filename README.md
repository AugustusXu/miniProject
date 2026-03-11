# AIST5030 GAI Mini-Project

## Parameter-efficient Finetuning with OFT (IMDB Sentiment)

Author: XU Kai (1155239333@link.cuhk.edu.hk)

This repository implements the mini-project requirement:

- Method: **Orthogonal Finetuning (OFT)** using Hugging Face PEFT
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Downstream task: IMDB sentiment classification (`positive` / `negative`)
- Output: code, report, and demonstration notebook

## 1. Environment

Your environment is already prepared as requested:

- Python: `3.12`
- PyTorch: `2.10`
- CUDA: `12.8`
- Environment name: `py312cuda128`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Project Structure

```text
.
├── src/
│   └── train_oft_imdb.py            # Main training + evaluation pipeline
├── scripts/
│   └── run_experiment.sh            # One-click run script
├── notebooks/
│   └── demo_oft_imdb.ipynb          # Demo notebook
├── report/
│   └── report.md                    # 3-page report draft/template
├── outputs/                         # Generated after running experiments
├── requirements.txt
└── README.md
```

## 3. What the Pipeline Does

`src/train_oft_imdb.py` provides an end-to-end experiment:

1. Load IMDB dataset subset
2. Evaluate base model accuracy before OFT
3. Apply OFT adapters to attention projection layers
4. Finetune on instruction-style sentiment prompts
5. Evaluate adapted model accuracy after OFT
6. Save artifacts:
	 - `outputs/logs/summary.json`
	 - `outputs/logs/predictions_before.json`
	 - `outputs/logs/predictions_after.json`
	 - `outputs/figures/training_loss.png`

## 4. Run Experiment

Option A: one command

```bash
bash scripts/run_experiment.sh
```

Option B: run with custom settings

```bash
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
```

## 5. Notebook Demo

Open and run:

- `notebooks/demo_oft_imdb.ipynb`

Notebook sections include:

- Dependency setup
- Training launch
- Summary metric visualization
- Loss curve display

## 6. Report

Draft report is in:

- `report/report.md`

It follows the required mini-project format and references the generated artifacts.
After running the experiment, fill in the final metrics and export to PDF.

## 7. Notes

- If GPU memory is tight, reduce `--train_samples` and `--batch_size`.
- If model download is slow, run once and keep HF cache for future runs.