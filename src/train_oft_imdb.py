import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from peft import OFTConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROMPT_TEMPLATE = (
    "Classify the sentiment of this movie review as exactly one word: "
    "positive or negative.\n\n"
    "Review: {text}\n"
    "Sentiment:"
)


@dataclass
class ExperimentConfig:
    model_name: str
    output_dir: str
    train_samples: int
    val_samples: int
    max_length: int
    learning_rate: float
    num_epochs: float
    batch_size: int
    grad_accum_steps: int
    eval_batch_size: int
    oft_r: int
    target_modules: str
    seed: int
    bf16: bool
    max_new_tokens: int


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OFT finetuning on IMDB sentiment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=400)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--oft_r", type=int, default=8)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target linear module names for OFT",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=4)
    return parser.parse_args()


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text.strip())


def normalize_label(text: str) -> str:
    lowered = text.lower()
    if "positive" in lowered:
        return "positive"
    if "negative" in lowered:
        return "negative"
    return "unknown"


def evaluate_accuracy(
    model,
    tokenizer,
    texts: List[str],
    labels: List[int],
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[float, List[Dict[str, str]]]:
    model.eval()
    preds = []
    examples = []

    label_map = {0: "negative", 1: "positive"}

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        prompts = [build_prompt(t) for t in batch_texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, out in enumerate(decoded):
            # Decode only the suffix after the prompt for stable label parsing.
            tail = out[len(prompts[i]) :].strip() if out.startswith(prompts[i]) else out
            preds.append(normalize_label(tail))

    gt = [label_map[x] for x in labels]
    correct = sum(int(p == y) for p, y in zip(preds, gt))
    accuracy = correct / len(gt)

    for i in range(min(8, len(gt))):
        examples.append(
            {
                "review": texts[i][:220].replace("\n", " "),
                "gold": gt[i],
                "pred": preds[i],
            }
        )

    return accuracy, examples


def prepare_train_features(tokenizer, texts: List[str], labels: List[int], max_length: int):
    label_map = {0: " negative", 1: " positive"}
    features = []

    for text, label in zip(texts, labels):
        prompt = build_prompt(text)
        completion = label_map[label]
        full_text = prompt + completion + tokenizer.eos_token

        tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length)
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_length)

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        labels_ids = input_ids.copy()
        prompt_len = min(len(tokenized_prompt["input_ids"]), len(labels_ids))
        for i in range(prompt_len):
            labels_ids[i] = -100

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels_ids,
            }
        )

    return features


def save_loss_curve(log_history: List[Dict], fig_path: str) -> None:
    steps = []
    losses = []
    for item in log_history:
        if "loss" in item and "step" in item:
            steps.append(item["step"])
            losses.append(item["loss"])

    if not steps:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linewidth=1.5)
    plt.title("OFT Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "model")
    logs_dir = os.path.join(args.output_dir, "logs")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    config = ExperimentConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        eval_batch_size=args.eval_batch_size,
        oft_r=args.oft_r,
        target_modules=args.target_modules,
        seed=args.seed,
        bf16=args.bf16,
        max_new_tokens=args.max_new_tokens,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    ds = load_dataset("imdb")
    train_ds = ds["train"].shuffle(seed=args.seed).select(range(args.train_samples))
    val_ds = ds["test"].shuffle(seed=args.seed).select(range(args.val_samples))

    val_texts = val_ds["text"]
    val_labels = val_ds["label"]

    before_acc, before_examples = evaluate_accuracy(
        base_model,
        tokenizer,
        val_texts,
        val_labels,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
    )

    train_features = prepare_train_features(
        tokenizer,
        train_ds["text"],
        train_ds["label"],
        args.max_length,
    )

    oft_cfg = OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.oft_r,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )

    model = get_peft_model(base_model, oft_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="no",
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        data_collator=CausalCollator(tokenizer),
    )

    train_output = trainer.train()
    trainer.model.save_pretrained(model_dir)

    # Evaluate adapted model by loading adapter on top of the same base model.
    tuned_model = PeftModel.from_pretrained(base_model, model_dir)

    after_acc, after_examples = evaluate_accuracy(
        tuned_model,
        tokenizer,
        val_texts,
        val_labels,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
    )

    loss_curve_path = os.path.join(figures_dir, "training_loss.png")
    save_loss_curve(trainer.state.log_history, loss_curve_path)

    summary = {
        "config": asdict(config),
        "train_runtime_sec": train_output.metrics.get("train_runtime", None),
        "train_loss": train_output.metrics.get("train_loss", None),
        "accuracy_before": before_acc,
        "accuracy_after": after_acc,
        "accuracy_gain": after_acc - before_acc,
        "loss_curve": os.path.relpath(loss_curve_path, args.output_dir),
    }

    with open(os.path.join(logs_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(logs_dir, "predictions_before.json"), "w", encoding="utf-8") as f:
        json.dump(before_examples, f, indent=2, ensure_ascii=False)

    with open(os.path.join(logs_dir, "predictions_after.json"), "w", encoding="utf-8") as f:
        json.dump(after_examples, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Experiment finished")
    print(f"Accuracy before OFT: {before_acc:.4f}")
    print(f"Accuracy after OFT : {after_acc:.4f}")
    print(f"Accuracy gain      : {after_acc - before_acc:.4f}")
    print(f"Results saved in   : {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
