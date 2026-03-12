"""
fine_tune.py

Fine-tune a BERT-base-uncased model on the IMDb sentiment dataset
using the Hugging Face Trainer API.

Usage:
    python fine_tune.py [--epochs N] [--batch_size N] [--output_dir PATH]
"""

import argparse
import os

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_CHECKPOINT = "bert-base-uncased"
LABEL_NAMES = ["negative", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on IMDb sentiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./output/bert-imdb")
    return parser.parse_args()


def tokenize_dataset(tokenizer, dataset):
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    return dataset.map(preprocess, batched=True)


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    args = parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("Loading IMDb dataset …")
    raw_datasets = load_dataset("imdb")

    # ── Tokenizer & model ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized = tokenize_dataset(tokenizer, raw_datasets)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_NAMES),
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
    )

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Training for {args.epochs} epochs …")
    trainer.train()

    print("Saving best model …")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
