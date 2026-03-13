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
├── image_gen.py        # Stable Diffusion txt2img, img2img, inpainting
├── requirements.txt
├── data/               # Place your own CSV/JSON datasets here
│   └── .gitkeep
└── loras/
    └── retro_pixel/
        └── retro_pixel_flux.py   # FLUX.1-dev + Retro-Pixel-Flux-LoRA text-to-image
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

### 5 — Generate images with Stable Diffusion

```bash
# Text to image
python image_gen.py txt2img --prompt "a photo of an astronaut riding a horse on mars" --steps 30

# Image to image
python image_gen.py img2img --prompt "convert to oil painting" --image input.png

# Inpainting (fill masked region)
python image_gen.py inpaint --prompt "a white cat" --image input.png --mask mask.png
```

Outputs are saved to `output/images/`. Model weights download automatically on first run but are gitignored.

### 6 — Retro-Pixel-Flux-LoRA (FLUX.1-dev)

Generates retro pixel-art style images using [prithivMLmods/Retro-Pixel-Flux-LoRA](https://huggingface.co/prithivMLmods/Retro-Pixel-Flux-LoRA) on top of `black-forest-labs/FLUX.1-dev`.

**Prerequisites:**
- NVIDIA GPU with ≥12 GB VRAM recommended (FLUX.1-dev is large)
- HuggingFace account with access to the gated FLUX.1-dev model:
  ```bash
  huggingface-cli login
  ```

```bash
python loras/retro_pixel/retro_pixel_flux.py \
    --prompt "Retro Pixel, A pixelated german shepherd dog on a light green background." \
    --steps 28 \
    --seed 42
```

The trigger word `Retro Pixel` is prepended automatically if you forget it. Outputs save to `output/loras/retro_pixel/`. Best at `1024×1024` (default).

| Argument | Default | Description |
|---|---|---|
| `--prompt` | required | Text prompt |
| `--steps` | 28 | Inference steps |
| `--guidance_scale` | 3.5 | CFG scale (FLUX uses ~3.5) |
| `--width` / `--height` | 1024 | Output dimensions |
| `--seed` | None | Reproducibility seed |
| `--output_dir` | `output/loras/retro_pixel` | Where to save |

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
| `image_gen.py` | Stable Diffusion txt2img / img2img / inpainting via `diffusers` |

---

## References

- [Hugging Face Transformers docs](https://huggingface.co/docs/transformers)
- [Hugging Face Hub](https://huggingface.co/models)
- [Datasets library](https://huggingface.co/docs/datasets)
- [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
