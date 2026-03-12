"""
inference.py

Load a fine-tuned model (or checkpoint) and run inference on custom text.

Usage:
    python inference.py --model_dir ./output/bert-imdb --text "Your review here"
    python inference.py --model_dir ./output/bert-imdb --file reviews.txt
"""

import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output/bert-imdb",
        help="Path to a saved model directory OR a Hugging Face model ID",
    )
    parser.add_argument("--text", type=str, default=None, help="Single text to classify")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .txt file (one review per line)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max token length for truncation"
    )
    return parser.parse_args()


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


@torch.no_grad()
def predict(texts: list[str], tokenizer, model, device: str, max_length: int) -> list[dict]:
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for text, prob in zip(texts, probs):
        label_id = int(prob.argmax())
        label = model.config.id2label[label_id]
        results.append({"text": text, "label": label, "confidence": float(prob[label_id])})
    return results


def main():
    args = parse_args()

    if args.text is None and args.file is None:
        print("Error: provide --text or --file", file=sys.stderr)
        sys.exit(1)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())

    tokenizer, model, device = load_model(args.model_dir)
    print(f"Using device: {device}")

    results = predict(texts, tokenizer, model, device, args.max_length)
    print("\n── Predictions ──")
    for r in results:
        snippet = r["text"][:80] + ("…" if len(r["text"]) > 80 else "")
        print(f"  {r['label'].upper():<10} {r['confidence']:.4f}  |  {snippet!r}")


if __name__ == "__main__":
    main()
