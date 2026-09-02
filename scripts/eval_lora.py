"""Evaluate a trained LoRA checkpoint against a fixed prompt set, per README's "Eval" plan item.

Two prompt sets, both fixed (same seed per prompt every run) so results are comparable
checkpoint-over-checkpoint:

  - Faithfulness set: real training-plate captions, evenly sampled across the 100 plates for
    order/genus diversity. Each generated image is placed side-by-side with its real plate so
    it's immediately visible where the model stays faithful to a known composition and where it
    drifts (composition, color, degree of detail).
  - Novel set: hand-written prompts that recombine training vocabulary (order tags, structural
    language) into combinations that don't exist in the dataset -- genus names not attached to
    that order in training, or blends across orders. This is the "produce specimens that don't
    exist but feel like they could" case from README's "Why Haeckel" section. There's no ground
    truth to compare against here, only the plate's own visual plausibility.

Writes generated images plus an index.html contact sheet (open it directly in a browser) to
outputs/eval/<checkpoint-name>/, so results from different checkpoints don't overwrite each
other and can be flipped between.

Usage:
    python scripts/eval_lora.py
    python scripts/eval_lora.py --checkpoint_dir outputs/checkpoints/step-1250
    python scripts/eval_lora.py --num_faithfulness 20 --guidance_scale 8.0
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import torch
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "final"
DEFAULT_PLATES_DIR = REPO_ROOT / "data" / "raw" / "plates"
DEFAULT_INDEX_CSV = REPO_ROOT / "data" / "processed" / "plate_index.csv"
DEFAULT_EVAL_DIR = REPO_ROOT / "outputs" / "eval"

DEFAULT_PRETRAINED_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRIGGER = "haeckel_kunstformen"

# Hand-picked recombinations of training vocabulary that don't exist in the dataset: known order
# tags paired with a genus that doesn't belong to them, or two orders blended into one prompt.
# There's no ground truth for these -- the point is judging plausibility, not accuracy.
NOVEL_PROMPTS = [
    f"{TRIGGER}, discomedusae, aurelia nova, natural history lithograph illustration",
    f"{TRIGGER}, siphonophorae, a deep-sea species newly discovered, natural history lithograph illustration",
    f"{TRIGGER}, radiolaria, a colonial radiolarian larger than any known species, natural history lithograph illustration",
    f"{TRIGGER}, hexacoralla, a hybrid coral with radiolarian symmetry, natural history lithograph illustration",
    f"{TRIGGER}, ophiodea, a brittle star with siphonophore tentacles, natural history lithograph illustration",
    f"{TRIGGER}, calcispongiae, an undiscovered sponge species, natural history lithograph illustration",
]


def build_caption(order: str, latin_name: str) -> str:
    # Must match scripts/build_captions.py's format exactly -- this is what the model trained on.
    return f"{TRIGGER}, {order.lower()}, {latin_name.lower()}, natural history lithograph illustration"


def load_faithfulness_set(index_csv: Path, n: int):
    with open(index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if n >= len(rows):
        sampled = rows
    else:
        # evenly spaced across the sorted plate list, for order/genus diversity rather than
        # n consecutive plates (which tend to cluster by taxonomic group in this index)
        step = len(rows) / n
        sampled = [rows[int(i * step)] for i in range(n)]
    return [
        {
            "plate": row["plate"],
            "caption": build_caption(row["order"], row["latin_name"]),
            "latin_name": row["latin_name"],
            "order": row["order"],
        }
        for row in sampled
    ]


def load_pipeline(pretrained_model: str, checkpoint_dir: Path, device: str):
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(str(checkpoint_dir))
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def side_by_side(real_path: Path, generated: Image.Image, size: int) -> Image.Image:
    real = Image.open(real_path).convert("RGB").resize((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size * 2 + 8, size), (255, 255, 255))
    canvas.paste(real, (0, 0))
    canvas.paste(generated, (size + 8, 0))
    return canvas


def write_index_html(eval_dir: Path, checkpoint_dir: Path, faithfulness_rows, novel_rows):
    def img_row(caption: str, rel_path: str, label: str = "") -> str:
        return (
            f"<figure><img src='{html.escape(rel_path)}' loading='lazy'>"
            f"<figcaption>{html.escape(label)}{html.escape(caption)}</figcaption></figure>"
        )

    faithfulness_html = "\n".join(
        img_row(r["caption"], f"faithfulness/{r['plate']}_compare.png", f"plate {r['plate']} (real | generated) — ")
        for r in faithfulness_rows
    )
    novel_html = "\n".join(
        img_row(prompt, f"novel/{i:02d}.png") for i, prompt in enumerate(novel_rows)
    )

    (eval_dir / "index.html").write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Haeckel LoRA eval — {html.escape(checkpoint_dir.name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; margin: 2rem; }}
h1, h2 {{ font-weight: 400; }}
figure {{ margin: 0 0 2rem 0; }}
figure img {{ max-width: 100%; display: block; border: 1px solid #444; }}
figcaption {{ font-size: 0.85rem; color: #aaa; margin-top: 0.4rem; }}
</style></head>
<body>
<h1>Haeckel LoRA eval — checkpoint: {html.escape(checkpoint_dir.name)}</h1>
<h2>Faithfulness set (real plate | generated, same caption)</h2>
{faithfulness_html}
<h2>Novel set (recombined vocabulary, no ground truth)</h2>
{novel_html}
</body></html>
""",
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--pretrained_model", type=str, default=DEFAULT_PRETRAINED_MODEL)
    p.add_argument("--plates_dir", type=Path, default=DEFAULT_PLATES_DIR)
    p.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX_CSV)
    p.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--num_faithfulness", type=int, default=12)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=35)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint_dir.exists():
        raise SystemExit(
            f"error: {args.checkpoint_dir} not found -- train a LoRA first (scripts/train_lora.py) "
            f"or point --checkpoint_dir at an existing checkpoint"
        )
    if not torch.cuda.is_available():
        raise SystemExit("error: no CUDA GPU visible")
    device = "cuda"

    out_dir = args.eval_dir / args.checkpoint_dir.name
    (out_dir / "faithfulness").mkdir(parents=True, exist_ok=True)
    (out_dir / "novel").mkdir(parents=True, exist_ok=True)

    print(f"loading {args.pretrained_model} + LoRA from {args.checkpoint_dir}")
    pipeline = load_pipeline(args.pretrained_model, args.checkpoint_dir, device)

    faithfulness_rows = load_faithfulness_set(args.index_csv, args.num_faithfulness)
    print(f"faithfulness set: {len(faithfulness_rows)} plates")

    for i, row in enumerate(faithfulness_rows):
        generator = torch.Generator(device=device).manual_seed(args.seed + i)
        image = pipeline(
            row["caption"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
        ).images[0]

        real_path = args.plates_dir / f"Tafel_{row['plate']}_300.jpg"
        if real_path.exists():
            compare = side_by_side(real_path, image, args.resolution)
            compare.save(out_dir / "faithfulness" / f"{row['plate']}_compare.png")
        else:
            print(f"warning: real plate {real_path} not found, saving generated only")
            image.save(out_dir / "faithfulness" / f"{row['plate']}_compare.png")
        print(f"  [{i+1}/{len(faithfulness_rows)}] plate {row['plate']}: {row['caption']}")

    print(f"novel set: {len(NOVEL_PROMPTS)} prompts")
    for i, prompt in enumerate(NOVEL_PROMPTS):
        generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + i)
        image = pipeline(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
        ).images[0]
        image.save(out_dir / "novel" / f"{i:02d}.png")
        print(f"  [{i+1}/{len(NOVEL_PROMPTS)}] {prompt}")

    write_index_html(out_dir, args.checkpoint_dir, faithfulness_rows, NOVEL_PROMPTS)
    print(f"done. open {out_dir / 'index.html'} in a browser to review")


if __name__ == "__main__":
    main()
