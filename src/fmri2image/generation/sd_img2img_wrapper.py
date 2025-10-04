# src/fmri2image/generation/sd_img2img_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Union
import torch
from PIL import Image

@dataclass
class SDConfig:
    model: str = "sd15"  # "sd15" | "sdxl" (img2img below covers sd15 for simplicity)
    device: str = "cuda"
    dtype: str = "fp16"  # "fp32" | "fp16" | "bf16"
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    negative_prompt: Optional[str] = None

def _to_list(x: Optional[Union[str, List[str]]], batch: int) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, str):
        return [x] * batch
    if isinstance(x, list):
        return x
    return None

class SDTextImg2Img:
    """
    Text2img + (optional) img2img refinement for SD 1.5 using Diffusers.
    """
    def __init__(self, cfg: SDConfig):
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

        self.cfg = cfg
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg.dtype]

        # text2img
        self.pipe_t2i = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None,  # local usage
            requires_safety_checker=False,
            torch_dtype=torch_dtype,
        )
        self.pipe_t2i.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe_t2i.scheduler.config)
        self.pipe_t2i = self.pipe_t2i.to(cfg.device)
        self.pipe_t2i.set_progress_bar_config(disable=True)

        # img2img
        self.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch_dtype,
        )
        self.pipe_i2i.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe_i2i.scheduler.config)
        self.pipe_i2i = self.pipe_i2i.to(cfg.device)
        self.pipe_i2i.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def text2img(self, prompts: List[str], seeds: Optional[List[int]] = None):
        neg = _to_list(self.cfg.negative_prompt, len(prompts))
        gen = None
        if seeds is not None:
            gen = [torch.Generator(device=self.cfg.device).manual_seed(int(s)) for s in seeds]
        out = self.pipe_t2i(
            prompt=prompts,
            negative_prompt=neg,
            height=self.cfg.height,
            width=self.cfg.width,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=gen,
        )
        return out.images

    @torch.inference_mode()
    def img2img(self, prompts: List[str], init_images: List[Image.Image], strength: float = 0.25, seeds: Optional[List[int]] = None):
        neg = _to_list(self.cfg.negative_prompt, len(prompts))
        gen = None
        if seeds is not None:
            gen = [torch.Generator(device=self.cfg.device).manual_seed(int(s)) for s in seeds]
        out = self.pipe_i2i(
            prompt=prompts,
            negative_prompt=neg,
            image=init_images,
            strength=float(strength),
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=gen,
        )
        return out.images
