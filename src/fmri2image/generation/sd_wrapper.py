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


def _install_transformers_compat_shim():
    """
    Some diffusers versions pass kwargs like 'offload_state_dict' down into
    transformers' PreTrainedModel.from_pretrained / __init__, which older
    transformers versions don't accept. This shim strips unknown kwargs.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return  # transformers not available, nothing to do

    orig_from_pretrained = PreTrainedModel.from_pretrained

    def _from_pretrained_shim(cls, *args, **kwargs):
        # Strip kwargs that can appear from diffusers loaders in older stacks
        for k in (
            "offload_state_dict",
            "keep_in_fp32_modules",
            "low_cpu_mem_usage",
            "variant",
            "use_safetensors",
            "ignore_mismatched_sizes",
        ):
            kwargs.pop(k, None)
        # Call the original classmethod function
        return orig_from_pretrained.__func__(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = classmethod(_from_pretrained_shim)


class StableDiffusionGenerator:
    """
    Minimal wrapper around Diffusers text-to-image pipelines (SD 1.5 / SDXL).
    Compatible with older transformers by:
      - disabling safety checker
      - forcing low_cpu_mem_usage=False
      - stripping unsupported kwargs via a transformers shim.
    """

    def __init__(self, cfg: SDConfig):
        self.cfg = cfg
        _install_transformers_compat_shim()

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

        # Load pipeline with safer defaults for mixed stacks
        self.pipe = Pipe.from_pretrained(
            repo,
            safety_checker=None,     # NOTE: disabled; keep enabled for public apps
            feature_extractor=None,
            low_cpu_mem_usage=False, # avoid offload paths incompatible with older transformers
            torch_dtype=torch_dtype, # keep for older diffusers; newer also accept dtype
        )

        # Optional: swap scheduler if requested
        if cfg.scheduler:
            try:
                if cfg.scheduler.lower() in {"dpmpp", "dpm-solver", "dpm"}:
                    from diffusers import DPMSolverMultistepScheduler
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            except Exception:
                pass  # keep default if swap fails

        # Move to device & quiet progress
        self.pipe = self.pipe.to(cfg.device)
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str] | str,
        seeds: Optional[List[int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[str | List[str]] = None,
    ):
        """Generate images from text prompts."""
        # --- Normalize prompts to a list ---
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        batch_size = len(prompt_list)

        # --- Normalize negative prompts to match type & length ---
        neg_arg = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt
        if neg_arg is None or neg_arg == "":
            neg_list: Optional[List[str]] = None
        elif isinstance(neg_arg, str):
            # broadcast single neg prompt to all batch items
            neg_list = [neg_arg] * batch_size
        else:
            # received a list: ensure correct length
            neg_list = list(neg_arg)
            if len(neg_list) != batch_size:
                # repeat or truncate to match batch_size
                if len(neg_list) < batch_size:
                    reps = (batch_size + len(neg_list) - 1) // len(neg_list)
                    neg_list = (neg_list * reps)[:batch_size]
                else:
                    neg_list = neg_list[:batch_size]

        # --- Seeds: per-image generators if provided ---
        generator = None
        if seeds is not None and len(seeds) > 0:
            if len(seeds) != batch_size:
                # repeat/truncate seeds to match batch_size
                reps = (batch_size + len(seeds) - 1) // len(seeds)
                seeds = (seeds * reps)[:batch_size]
            generator = [torch.Generator(device=self.cfg.device).manual_seed(s) for s in seeds]

        # --- Other params ---
        h = height or self.cfg.height
        w = width or self.cfg.width
        gs = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        steps = self.cfg.num_inference_steps if num_inference_steps is None else num_inference_steps

        out = self.pipe(
            prompt=prompt_list,
            negative_prompt=neg_list,          # list or None; matches prompt_list
            num_inference_steps=steps,
            guidance_scale=gs,
            height=h,
            width=w,
            generator=generator,
        )
        return out.images
