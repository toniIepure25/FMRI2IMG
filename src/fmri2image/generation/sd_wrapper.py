from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Sequence
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
    scheduler: Optional[str] = None  # keep default

class StableDiffusionGenerator:
    """
    Minimal wrapper around Diffusers text-to-image pipelines (SD 1.5 / SDXL),
    with optional img2img refine.
    """
    def __init__(self, cfg: SDConfig):
        self.cfg = cfg
        try:
            from diffusers import (
                StableDiffusionPipeline,
                StableDiffusionImg2ImgPipeline,
                StableDiffusionXLPipeline,
                StableDiffusionXLImg2ImgPipeline,
            )
        except Exception as e:
            raise RuntimeError(
                "diffusers not installed. Please `pip install diffusers transformers accelerate safetensors`"
            ) from e

        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.dtype]

        self.is_sdxl = cfg.model.lower() in ("sdxl", "stable-diffusion-xl")

        if not self.is_sdxl:
            repo = "runwayml/stable-diffusion-v1-5"
            # NOTE: safety_checker=None to avoid version skew issues
            self.pipe = StableDiffusionPipeline.from_pretrained(
                repo, torch_dtype=torch_dtype, safety_checker=None
            )
            self.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
                repo, torch_dtype=torch_dtype, safety_checker=None
            )
        else:
            repo = "stabilityai/stable-diffusion-xl-base-1.0"
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                repo, torch_dtype=torch_dtype, safety_checker=None
            )
            self.pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                repo, torch_dtype=torch_dtype, safety_checker=None
            )

        self.pipe = self.pipe.to(cfg.device)
        self.pipe_i2i = self.pipe_i2i.to(cfg.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe_i2i.set_progress_bar_config(disable=True)

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

        # Diffusers requires negative_prompt to match prompt type/len
        if isinstance(prompts, list):
            if isinstance(neg, str):
                neg = [neg] * len(prompts)
            elif neg is None:
                neg = None
        else:
            # prompts is str
            if isinstance(neg, list):
                neg = neg[0] if neg else None

        generator = None
        if seeds is not None:
            generator = [torch.Generator(device=self.cfg.device).manual_seed(s) for s in seeds]

        out = self.pipe(
            prompt=prompts,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=gs,
            height=h,
            width=w,
            generator=generator
        )
        return out.images

    @torch.inference_mode()
    def refine_img2img(
        self,
        init_images: Sequence,            # list[PIL.Image.Image]
        prompts: List[str],
        strength: float = 0.3,
        seeds: Optional[List[int]] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ):
        gs = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        steps = self.cfg.num_inference_steps if num_inference_steps is None else num_inference_steps
        neg = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        if isinstance(neg, str):
            neg = [neg] * len(prompts)

        generator = None
        if seeds is not None:
            generator = [torch.Generator(device=self.cfg.device).manual_seed(s) for s in seeds]

        out = self.pipe_i2i(
            prompt=prompts,
            image=list(init_images),
            negative_prompt=neg,
            guidance_scale=gs,
            num_inference_steps=steps,
            strength=float(strength),
            generator=generator
        )
        return out.images
