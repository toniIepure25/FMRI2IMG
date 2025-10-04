# src/fmri2image/generation/sd_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class SDConfig:
    model: str = "sd15"  # "sd15" | "sdxl"
    device: str = "cuda"
    dtype: str = "fp16"  # "fp32" | "fp16" | "bf16"
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    negative_prompt: Optional[str] = None
    scheduler: Optional[str] = None  # keep default unless you know what you want

class StableDiffusionGenerator:
    """
    Minimal wrapper around Diffusers text-to-image pipelines (SD 1.5 / SDXL).
    """
    def __init__(self, cfg: SDConfig):
        self.cfg = cfg
        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        except Exception as e:
            raise RuntimeError(
                "diffusers not installed. Please `pip install diffusers transformers accelerate safetensors`"
            ) from e

        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.dtype]

        if cfg.model.lower() in ("sd15", "sd1.5", "stable-diffusion-1.5"):
            repo = "runwayml/stable-diffusion-v1-5"
            from diffusers import StableDiffusionPipeline as Pipe
            self.pipe = Pipe.from_pretrained(repo, torch_dtype=torch_dtype)
        elif cfg.model.lower() in ("sdxl", "stable-diffusion-xl"):
            repo = "stabilityai/stable-diffusion-xl-base-1.0"
            from diffusers import StableDiffusionXLPipeline as Pipe
            self.pipe = Pipe.from_pretrained(repo, torch_dtype=torch_dtype)
        else:
            raise ValueError(f"Unknown SD model: {cfg.model}")

        self.pipe = self.pipe.to(cfg.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ):
        h = height or self.cfg.height
        w = width or self.cfg.width
        gs = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        steps = self.cfg.num_inference_steps if num_inference_steps is None else num_inference_steps
        neg = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        generator = None
        if seeds is not None:
            # If you pass a list equal to batch size, Diffusers will use per-image seeds.
            generator = [torch.Generator(device=self.cfg.device).manual_seed(s) for s in seeds]

        images = self.pipe(
            prompt=prompts,
            negative_prompt=neg if neg else None,
            num_inference_steps=steps,
            guidance_scale=gs,
            height=h,
            width=w,
            generator=generator
        ).images
        return images
