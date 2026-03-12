# t1-python-huggingface-transformer

End-to-end implementation of Hugging Face Transformers for NLP:
pipeline demos, BERT fine-tuning on IMDb, and a standalone inference script.

---

## Project Structure

```
t1-python-huggingface-transformer/
├── pipeline_demo.py    # Quick-start demos: sentiment, generation, summarization, QA, translation
├── fine_tune.py        # Fine-tune BERT-base-uncased on IMDb via Trainer API
├── inference.py        # Load a saved model and classify text (--text or --file)
├── custom_dataset.py   # Utility to load CSV/JSON datasets as DatasetDict
├── requirements.txt
└── data/               # Place your own CSV/JSON datasets here
    └── .gitkeep
```

---

## Quick Start

### 1 — Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2 — Run pipeline demos

Downloads small pretrained models automatically on first run.

```bash
python pipeline_demo.py
```

### 3 — Fine-tune BERT on IMDb

Requires ~6 GB VRAM for full batch; reduce `--batch_size` for CPU runs.

```bash
python fine_tune.py --epochs 3 --batch_size 16 --output_dir ./output/bert-imdb
```

### 4 — Run inference on a saved model

```bash
# Single text
python inference.py --model_dir ./output/bert-imdb \
    --text "This movie was absolutely fantastic!"

# File with one review per line
python inference.py --model_dir ./output/bert-imdb --file reviews.txt
```

You can also point `--model_dir` at any Hugging Face Hub model ID (e.g. `distilbert-base-uncased-finetuned-sst-2-english`).

---

## Key Concepts

| File | What it shows |
|------|---------------|
| `pipeline_demo.py` | `pipeline()` API — zero-config access to dozens of task heads |
| `fine_tune.py` | `AutoModelForSequenceClassification` + `Trainer` workflow |
| `inference.py` | `torch.no_grad()` batched inference, CPU/GPU auto-selection |
| `custom_dataset.py` | `load_dataset()` with custom files, `ClassLabel` casting |

---

## References

- [Hugging Face Transformers docs](https://huggingface.co/docs/transformers)
- [Hugging Face Hub](https://huggingface.co/models)
- [Datasets library](https://huggingface.co/docs/datasets)
- [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
