"""
image_gen.py

Image generation with Hugging Face Diffusers library.
Covers: Stable Diffusion text-to-image, image-to-image, and inpainting pipelines.

Requirements: pip install diffusers accelerate transformers torch Pillow

Usage:
    python image_gen.py txt2img  --prompt "a photo of an astronaut on mars"
    python image_gen.py img2img  --prompt "turn into oil painting" --image input.png
    python image_gen.py inpaint  --prompt "a white cat" --image input.png --mask mask.png
"""

import argparse
import pathlib
import sys
from typing import Optional

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from PIL import Image


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, output_dir: str, stem: str) -> pathlib.Path:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{stem}.png"
    image.save(dest)
    print(f"  Saved → {dest}")
    return dest


# ── Text-to-Image ─────────────────────────────────────────────────────────────

def text_to_image(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    output_dir: str = "output/images",
) -> Image.Image:
    """Generate an image from a text prompt (Stable Diffusion txt2img)."""
    device = get_device()
    print(f"Loading txt2img pipeline on {device}: {model_id}")

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    image = result.images[0]
    save_image(image, output_dir, "txt2img_output")
    return image


# ── Image-to-Image ────────────────────────────────────────────────────────────

def image_to_image(
    prompt: str,
    input_image_path: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    steps: int = 30,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    output_dir: str = "output/images",
) -> Image.Image:
    """Transform an existing image guided by a text prompt."""
    device = get_device()
    print(f"Loading img2img pipeline on {device}: {model_id}")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    init_image = load_image(input_image_path).resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    result = pipe(
        prompt=prompt,
        image=init_image,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    image = result.images[0]
    save_image(image, output_dir, "img2img_output")
    return image


# ── Inpainting ────────────────────────────────────────────────────────────────

def inpaint(
    prompt: str,
    input_image_path: str,
    mask_image_path: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-inpainting",
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    output_dir: str = "output/images",
) -> Image.Image:
    """Fill a masked region of an image using a text prompt."""
    device = get_device()
    print(f"Loading inpainting pipeline on {device}: {model_id}")

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    image = load_image(input_image_path).resize((512, 512))
    mask = Image.open(mask_image_path).convert("L").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    out_image = result.images[0]
    save_image(out_image, output_dir, "inpaint_output")
    return out_image


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion image generation")
    sub = parser.add_subparsers(dest="mode", required=True)

    # txt2img
    t = sub.add_parser("txt2img", help="Text-to-image generation")
    t.add_argument("--prompt", required=True)
    t.add_argument("--negative_prompt", default="")
    t.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    t.add_argument("--steps", type=int, default=30)
    t.add_argument("--guidance_scale", type=float, default=7.5)
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--height", type=int, default=512)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--output_dir", default="output/images")

    # img2img
    i = sub.add_parser("img2img", help="Image-to-image transformation")
    i.add_argument("--prompt", required=True)
    i.add_argument("--image", required=True, help="Path to the input image")
    i.add_argument("--negative_prompt", default="")
    i.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    i.add_argument("--steps", type=int, default=30)
    i.add_argument("--strength", type=float, default=0.75)
    i.add_argument("--guidance_scale", type=float, default=7.5)
    i.add_argument("--seed", type=int, default=None)
    i.add_argument("--output_dir", default="output/images")

    # inpaint
    p = sub.add_parser("inpaint", help="Inpainting using a mask")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", required=True, help="Path to the input image")
    p.add_argument("--mask", required=True, help="Path to the mask image (white = fill area)")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--model", default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", default="output/images")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "txt2img":
        text_to_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            model_id=args.model,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.mode == "img2img":
        image_to_image(
            prompt=args.prompt,
            input_image_path=args.image,
            negative_prompt=args.negative_prompt,
            model_id=args.model,
            steps=args.steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.mode == "inpaint":
        inpaint(
            prompt=args.prompt,
            input_image_path=args.image,
            mask_image_path=args.mask,
            negative_prompt=args.negative_prompt,
            model_id=args.model,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
