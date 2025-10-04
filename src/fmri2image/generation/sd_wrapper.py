# src/fmri2image/generation/sd_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class SDConfig:
    """Config for Stable Diffusion generation."""
    model: str = "sd15"            # "sd15" | "sdxl"
    device: str = "cuda"
    dtype: str = "fp16"            # "fp32" | "fp16" | "bf16"
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    negative_prompt: Optional[str] = None
    scheduler: Optional[str] = None  # e.g., "dpmpp", "default"


class StableDiffusionGenerator:
    """
    Minimal wrapper around Diffusers text-to-image pipelines (SD 1.5 / SDXL).
    This version disables the safety checker and the low-CPU-mem loading path
    to avoid version-mismatch issues like 'offload_state_dict'.
    """

    def __init__(self, cfg: SDConfig):
        self.cfg = cfg

        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        except Exception as e:
            raise RuntimeError(
                "diffusers not installed. Please `pip install diffusers transformers accelerate safetensors`"
            ) from e

        # Map string to torch dtype
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(cfg.dtype, torch.float16)

        # Select model repo and pipeline class
        model_key = cfg.model.lower()
        if model_key in ("sd15", "sd1.5", "stable-diffusion-1.5"):
            repo = "runwayml/stable-diffusion-v1-5"
            Pipe = StableDiffusionPipeline
        elif model_key in ("sdxl", "stable-diffusion-xl"):
            repo = "stabilityai/stable-diffusion-xl-base-1.0"
            Pipe = StableDiffusionXLPipeline
        else:
            raise ValueError(f"Unknown SD model: {cfg.model}")

        # --- IMPORTANT: avoid offload_state_dict issues across versions ---
        #  - safety_checker=None, feature_extractor=None -> do not load safety checker
        #  - low_cpu_mem_usage=False -> do not try to pass offload_state_dict to submodules
        self.pipe = Pipe.from_pretrained(
            repo,
            safety_checker=None,
            feature_extractor=None,
            torch_dtype=torch_dtype,       # 'dtype' is the new arg; keep for compatibility
            low_cpu_mem_usage=False,
        )

        # Optional: swap scheduler if requested (safe fallback if missing)
        if cfg.scheduler:
            try:
                if cfg.scheduler.lower() in {"dpmpp", "dpm-solver", "dpm"}:
                    from diffusers import DPMSolverMultistepScheduler
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                # else: keep default scheduler
            except Exception:
                # If scheduler change fails, silently keep default
                pass

        # Move to device and make UI quiet
        self.pipe = self.pipe.to(cfg.device)
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass  # older diffusers may not have this

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
        """Generate images from text prompts."""
        h = height or self.cfg.height
        w = width or self.cfg.width
        gs = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        steps = self.cfg.num_inference_steps if num_inference_steps is None else num_inference_steps
        neg = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        # Per-image generators if seeds are provided
        generator = None
        if seeds is not None:
            generator = [torch.Generator(device=self.cfg.device).manual_seed(s) for s in seeds]

        out = self.pipe(
            prompt=prompts,
            negative_prompt=neg if neg else None,
            num_inference_steps=steps,
            guidance_scale=gs,
            height=h,
            width=w,
            generator=generator,
        )
        return out.images
