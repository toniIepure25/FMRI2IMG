# src/fmri2image/generation/sd_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Sequence
import contextlib
import torch

@dataclass
class SDConfig:
    model: str = "sd15"   # "sd15" | "sdxl"
    device: str = "cuda"
    dtype: str = "fp16"   # "fp32" | "fp16" | "bf16"
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    negative_prompt: Optional[str] = None
    scheduler: Optional[str] = None  # keep default unless you know what you want


@contextlib.contextmanager
def _patch_transformers_offload_state_dict():
    """
    Some diffusers versions pass `offload_state_dict` to transformers models,
    but older transformers don't accept it. This patch drops that kwarg.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel  # type: ignore
    except Exception:
        # If transformers is not available, nothing to patch.
        yield
        return

    original = PreTrainedModel.from_pretrained

    def patched(cls, *args, **kwargs):
        kwargs.pop("offload_state_dict", None)  # <- critical
        return original.__func__(cls, *args, **kwargs)  # call unbound func

    try:
        PreTrainedModel.from_pretrained = classmethod(patched)  # type: ignore
        yield
    finally:
        PreTrainedModel.from_pretrained = original  # type: ignore


class StableDiffusionGenerator:
    """
    Minimal wrapper around Diffusers text-to-image pipelines (SD 1.5 / SDXL),
    with optional img2img refine. Includes a fallback patch to ignore
    `offload_state_dict` when older transformers are installed.
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

        common_kwargs = dict(
            torch_dtype=torch_dtype,
            safety_checker=None,          # you disable safety checker; keep internal note in logs
            low_cpu_mem_usage=False,      # try to avoid offload path
        )

        def _build_pipes():
            if not self.is_sdxl:
                repo = "runwayml/stable-diffusion-v1-5"
                self.pipe = StableDiffusionPipeline.from_pretrained(repo, **common_kwargs)
                self.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(repo, **common_kwargs)
            else:
                repo = "stabilityai/stable-diffusion-xl-base-1.0"
                self.pipe = StableDiffusionXLPipeline.from_pretrained(repo, **common_kwargs)
                self.pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(repo, **common_kwargs)

        # First try normally…
        try:
            _build_pipes()
        except TypeError as e:
            # …if we still hit offload_state_dict path, patch transformers & retry once.
            if "offload_state_dict" in str(e):
                with _patch_transformers_offload_state_dict():
                    _build_pipes()
            else:
                raise

        self.pipe = self.pipe.to(cfg.device)
        self.pipe_i2i = self.pipe_i2i.to(cfg.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe_i2i.set_progress_bar_config(disable=True)


    def load_lora(self, lora_path: str, alpha: float | None = None):
        """
        Load a LoRA adapter file (.safetensors) into the pipeline.
        If alpha is provided and the diffusers version supports it, set adapter weight.
        """
        try:
            self.pipe.load_lora_weights(lora_path)
        except Exception as e:
            # Fallback: some versions require passing local_pretrained_model_name_or_path
            try:
                from diffusers.loaders import LoraLoaderMixin  # for type context
                self.pipe.load_lora_weights(lora_path, adapter_name="default")
            except Exception as e2:
                raise RuntimeError(f"Failed to load LoRA weights from {lora_path}: {e} / {e2}")

        if alpha is not None:
            try:
                self.pipe.set_adapters(["default"], adapter_weights=[float(alpha)])
            except Exception:
                # Older diffusers might not support set_adapters; safe to ignore.
                pass

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

        # Ensure negative_prompt matches type/len of prompt (diffusers requirement)
        if isinstance(prompts, list):
            if isinstance(neg, str):
                neg = [neg] * len(prompts)
        else:
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
