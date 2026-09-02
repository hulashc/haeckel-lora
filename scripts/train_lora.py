"""Train a rank-16 LoRA adapter on Stable Diffusion 1.5's UNet cross-attention layers,
using data/processed/dataset/ (image + .txt caption pairs from build_captions.py).

Designed around one hard constraint: a 6GB VRAM GPU (RTX 3060). That shapes every choice
below, not as an afterthought:

  - 512x512 resolution (SD1.5's native res -- no reason to go higher and every reason not to)
  - LoRA only: VAE and text encoder are frozen and cast to fp16 permanently (they're never
    trained, so there's no reason to keep fp32 master weights around). Only the UNet's LoRA
    adapter weights are trainable.
  - The trainable UNet forward pass runs under torch.autocast (bf16 by default -- the 3060 is
    Ampere, which has native bf16 tensor core support, so there's no GradScaler dance needed;
    fp16 is available as a fallback via --mixed_precision for older/other GPUs).
  - Gradient checkpointing on the UNet, trading recompute for activation memory.
  - Batch size 1 with gradient accumulation to reach a reasonable effective batch size, rather
    than a real batch >1 which would multiply activation memory directly.
  - PyTorch SDPA attention (the diffusers default on torch>=2.0 -- no xformers install needed).
  - No HF Accelerate: this is a single-process, single-GPU script, so plain torch.autocast +
    manual grad accumulation is fewer moving parts than standing up an Accelerator for one
    device. (accelerate stays in requirements.txt for anyone who wants to extend this later.)

With those choices, the whole training loop fits in ~4-5GB: SD1.5 UNet is ~860M params
(~1.7GB in fp32, which is what the trainable copy needs since LoRA base weights + adapter
gradients + AdamW state must be fp32-precise; the *frozen* 96% of the UNet doesn't get
gradients at all, so only the tiny LoRA adapter tensors carry optimizer state), plus a fp16
VAE/text encoder (~350MB) and fp16 activations under gradient checkpointing.

Usage:
    python scripts/train_lora.py
    python scripts/train_lora.py --num_train_epochs 200 --validation_prompt "haeckel_kunstformen, discomedusae, a new medusa species, natural history lithograph illustration"
    python scripts/train_lora.py --mixed_precision fp16   # if bf16 misbehaves on your setup
    python scripts/train_lora.py --resume_from outputs/checkpoints/step-1500

See data/processed/README.md and README.md ("Plan" / "Inference") for the rest of the
project's design rationale.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "processed" / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "checkpoints"
DEFAULT_SAMPLE_DIR = REPO_ROOT / "outputs" / "samples"

# The community mirror -- runwayml/stable-diffusion-v1-5 was pulled from the Hub in 2023.
DEFAULT_PRETRAINED_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


class HaeckelPlateDataset(Dataset):
    """Reads data/processed/dataset/*.txt + same-stem image pairs written by build_captions.py."""

    def __init__(self, dataset_dir: Path, tokenizer: CLIPTokenizer, resolution: int, caption_dropout_prob: float):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.caption_dropout_prob = caption_dropout_prob

        self.examples = []
        for txt_path in sorted(dataset_dir.glob("*.txt")):
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = txt_path.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                print(f"warning: no image found for caption {txt_path.name}, skipping")
                continue
            caption = txt_path.read_text(encoding="utf-8").strip()
            self.examples.append((img_path, caption))

        if not self.examples:
            raise RuntimeError(
                f"no image/caption pairs found in {dataset_dir} -- run scripts/build_captions.py first"
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img_path, caption = self.examples[idx]

        # Direct resize to a square, rather than resize-then-crop: these plates are portrait
        # (~2450x3600) with the specimen roughly centered and a printed caption in the margin.
        # A crop-to-square risks cutting off part of the illustration; a plain resize distorts
        # the aspect ratio slightly but keeps the whole plate, which matters more here since
        # every plate is a single unique piece of content (there's no "reshoot the crop" option).
        image = Image.open(img_path).convert("RGB").resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        pixel_values = pixel_values / 127.5 - 1.0  # [0, 255] -> [-1, 1]

        # Caption dropout: train some fraction of steps on an empty prompt so the model learns
        # a real unconditional distribution, which is what classifier-free guidance (the
        # guidance_scale=7.5 from README's inference plan) actually interpolates against.
        text = "" if random.random() < self.caption_dropout_prob else caption

        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {"pixel_values": pixel_values, "input_ids": tokenized.input_ids[0]}


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def save_lora_checkpoint(unet, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(save_dir),
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )


@torch.no_grad()
def run_validation(args, unet, vae, text_encoder, tokenizer, device, weight_dtype, step: int):
    """Generate a couple of sample images with the LoRA weights as currently trained, so
    progress is visually inspectable without a separate eval script. Best-effort: frees the
    training-loop cache first and restores unet.train() afterward regardless of outcome."""
    print(f"\nrunning validation at step {step}: {args.validation_prompt!r}")
    torch.cuda.empty_cache()

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)

    unet.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    autocast_enabled = args.mixed_precision != "no"
    try:
        # Same reason as the training loop: unet's base weights stay fp32 (only the LoRA
        # adapter is trainable), so the forward pass needs autocast to bridge against the
        # permanently-fp16 vae/text_encoder -- without it this is a Half/Float dtype mismatch.
        with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=autocast_enabled):
            images = pipeline(
                args.validation_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
                num_images_per_prompt=args.num_validation_images,
            ).images
    finally:
        unet.train()

    args.sample_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        image.save(args.sample_dir / f"step-{step:06d}_{i}.png")
    print(f"wrote {len(images)} sample(s) to {args.sample_dir}")

    del pipeline
    torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sample_dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    p.add_argument("--pretrained_model", type=str, default=DEFAULT_PRETRAINED_MODEL)
    p.add_argument("--resume_from", type=Path, default=None, help="dir with pytorch_lora_weights.safetensors to resume adapter weights from")

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=None, help="defaults to --rank")
    p.add_argument("--caption_dropout_prob", type=float, default=0.1)

    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--max_train_steps", type=int, default=None, help="overrides --num_train_epochs if set")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"])
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--validation_epochs", type=int, default=20)
    p.add_argument("--num_validation_images", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = args.rank

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible -- this script is not designed/tested for CPU training")
    device = "cuda"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]
    print(f"device={device} mixed_precision={args.mixed_precision} ({weight_dtype})")

    # --- load components ---
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")

    # Frozen, never trained -> cast to fp16 permanently and drop to eval. No reason to keep
    # fp32 master copies of weights that never get an optimizer step.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.eval()
    vae.eval()

    # UNet: base weights frozen too, but stays fp32 on device -- LoRA forward/backward for the
    # trainable adapter runs under autocast (see the training loop), and autocast needs an
    # fp32-or-compatible base to cast from. Only the LoRA adapter params below get requires_grad.
    unet.requires_grad_(False)
    unet.to(device, dtype=torch.float32)

    if args.resume_from is not None:
        # load_lora_adapter injects the adapter (reading rank/alpha off the saved weights)
        # and loads its values in one call -- don't call add_adapter first, it would double-inject.
        print(f"resuming adapter weights from {args.resume_from}")
        unet.load_lora_adapter(str(args.resume_from), adapter_name="default")
    else:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        )
        unet.add_adapter(lora_config)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    lora_layers = [p for p in unet.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in lora_layers)
    print(f"training {len(lora_layers)} LoRA tensors, {n_params:,} trainable params")

    # --- data ---
    dataset = HaeckelPlateDataset(args.dataset_dir, tokenizer, args.resolution, args.caption_dropout_prob)
    print(f"dataset: {len(dataset)} image/caption pairs from {args.dataset_dir}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    optimizer = torch.optim.AdamW(lora_layers, lr=args.learning_rate, weight_decay=1e-2)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    autocast_enabled = args.mixed_precision != "no"
    # fp16 needs loss scaling to avoid gradient underflow; bf16's exponent range matches fp32
    # so GradScaler is a no-op there (and torch warns/refuses to combine it with bf16).
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision == "fp16"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = args.output_dir / "loss_log.csv"
    loss_log_is_new = not loss_log_path.exists()
    loss_log = open(loss_log_path, "a", newline="", encoding="utf-8")
    loss_log_writer = csv.writer(loss_log)
    if loss_log_is_new:
        loss_log_writer.writerow(["step", "epoch", "loss", "lr"])

    print(
        f"total optimizer steps: {args.max_train_steps} over {args.num_train_epochs} epochs "
        f"(effective batch size {args.train_batch_size * args.gradient_accumulation_steps})"
    )

    global_step = 0
    progress = tqdm(total=args.max_train_steps, desc="train")
    unet.train()

    for epoch in range(args.num_train_epochs):
        if global_step >= args.max_train_steps:
            break
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=autocast_enabled):
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)
                ).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            running_loss += loss.item()
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            is_accum_boundary = (micro_step + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = micro_step == len(dataloader) - 1
            if not (is_accum_boundary or is_last_batch):
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_layers, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            step_loss = running_loss / (micro_step + 1)
            progress.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
            loss_log_writer.writerow([global_step, epoch, step_loss, lr_scheduler.get_last_lr()[0]])
            loss_log.flush()

            if global_step % args.checkpointing_steps == 0:
                ckpt_dir = args.output_dir / f"step-{global_step}"
                save_lora_checkpoint(unet, ckpt_dir)
                progress.write(f"saved checkpoint to {ckpt_dir}")

            if global_step >= args.max_train_steps:
                break

        if (
            args.validation_prompt is not None
            and (epoch + 1) % args.validation_epochs == 0
        ):
            run_validation(args, unet, vae, text_encoder, tokenizer, device, weight_dtype, global_step)

    progress.close()
    loss_log.close()

    final_dir = args.output_dir / "final"
    save_lora_checkpoint(unet, final_dir)
    print(f"done. final LoRA weights at {final_dir}")


if __name__ == "__main__":
    main()
