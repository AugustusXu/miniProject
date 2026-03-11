# Mini-Project Report (AIST5030 GAI)

## Parameter-efficient Finetuning with OFT on Qwen2.5-0.5B-Instruct

### 1. Task and Setup
This project applies **Orthogonal Finetuning (OFT)** to a pretrained language model for a downstream sentiment classification task.

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: OFT (from Hugging Face PEFT)
- Task: IMDB sentiment classification (binary)
- Prompt format: model outputs one word, `positive` or `negative`
- Environment: `py312cuda128`, PyTorch 2.10 (CUDA 12.8)

### 2. Experimental Configuration
Main hyperparameters:

- Training samples: 2000
- Validation samples: 400
- Epochs: 1
- Learning rate: 2e-4
- Batch size: 4
- Gradient accumulation: 4
- OFT rank `r`: 8
- Target modules: `q_proj,k_proj,v_proj,o_proj`

The script saves all artifacts to `outputs/`.

### 3. Training Curve
The training loss curve is generated at:

- `outputs/figures/training_loss.png`

Please insert the figure in your final PDF export.

![Training Loss](../outputs/figures/training_loss.png)

### 4. Quantitative Results (Before vs After OFT)
Primary metric: classification accuracy on the validation subset.

Read from `outputs/logs/summary.json`:

- Accuracy before OFT: `accuracy_before`
- Accuracy after OFT: `accuracy_after`
- Gain: `accuracy_gain`

Table template (replace with actual values after running):

| Model state | Accuracy |
|---|---:|
| Before OFT | TBD |
| After OFT | TBD |
| Improvement | TBD |

### 5. Qualitative Analysis
Prediction examples are exported to:

- `outputs/logs/predictions_before.json`
- `outputs/logs/predictions_after.json`

You can compare the same review before/after adaptation and discuss:

- Better polarity consistency for emotional words
- Reduced `unknown` outputs after OFT
- Typical remaining error patterns (sarcasm, mixed sentiment)

### 6. Conclusion
This mini-project demonstrates that OFT can improve downstream sentiment classification performance while updating only a small set of low-rank orthogonal parameters rather than full model weights.

For final submission, export this report to PDF (about 3 pages) and include the training curve and final metrics.
