# src/fmri2image/generation/sd_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Sequence
import contextlib
import torch

import math
from safetensors.torch import load_file as load_safetensors

class LinearWithLoRA(torch.nn.Module):
    """
    Wraps an existing nn.Linear as `base`, and adds LoRA A/B:
        y = base(x) + B(A(x)) * scale
    """
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        assert isinstance(base, torch.nn.Linear)
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)

        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale

def _get_linear(module, proj_name: str):
    """
    Return the nn.Linear for a given projection name inside a Diffusers attention block.
    proj_name in {'to_q','to_k','to_v','to_out.0'}.
    """
    if proj_name == "to_out.0":
        to_out = getattr(module, "to_out", None)
        if to_out is None or not hasattr(to_out, "__getitem__"):
            return None
        lin = to_out[0]
    else:
        lin = getattr(module, proj_name, None)
    return lin if isinstance(lin, torch.nn.Linear) else None

def _replace_with_lora_wrapper(module, proj_name: str, rank: int, alpha: float):
    """
    Replace the Linear at `proj_name` with LinearWithLoRA, preserving parameter/device.
    Idempotent: if already wrapped, returns the existing wrapper.
    """
    # fetch current linear
    if proj_name == "to_out.0":
        to_out = getattr(module, "to_out", None)
        lin = to_out[0] if (to_out is not None and hasattr(to_out, "__getitem__")) else None
    else:
        lin = getattr(module, proj_name, None)

    if lin is None or not isinstance(lin, torch.nn.Linear):
        return None

    # already wrapped?
    if isinstance(lin, LinearWithLoRA):
        return lin

    # build wrapper and swap in place
    wrapper = LinearWithLoRA(lin, rank=rank, alpha=alpha).to(lin.weight.device, dtype=lin.weight.dtype)

    if proj_name == "to_out.0":
        to_out[0] = wrapper
    else:
        setattr(module, proj_name, wrapper)

    return wrapper

def _load_simple_lora_into_unet(unet, state: dict, alpha: float):
    """
    Load simple A/B LoRA weights from safetensors into UNet attention projections.
    Expects keys:
      '{module_path}.to_q.lora_A.weight' and '{module_path}.to_q.lora_B.weight'  (same for to_k, to_v, to_out.0)
    """
    loaded = 0
    for name, block in unet.named_modules():
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            kA = f"{name}.{proj}.lora_A.weight"
            kB = f"{name}.{proj}.lora_B.weight"
            if kA in state and kB in state:
                rank = int(state[kA].shape[0])
                wrapper = _replace_with_lora_wrapper(block, proj, rank=rank, alpha=alpha)
                if wrapper is None:
                    continue
                with torch.no_grad():
                    wrapper.lora_A.weight.copy_(state[kA].to(wrapper.lora_A.weight.device, dtype=wrapper.lora_A.weight.dtype))
                    wrapper.lora_B.weight.copy_(state[kB].to(wrapper.lora_B.weight.device, dtype=wrapper.lora_B.weight.dtype))
                loaded += 1
    if loaded == 0:
        raise RuntimeError("No matching A/B LoRA keys were loaded into UNet.")
    return loaded

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


    def load_lora(self, lora_path: str, alpha: float = 1.0):
        """
        Try native Diffusers LoRA loader first; if it fails (old/simple A/B format),
        fall back to our UNet wrapper injection.
        """
        # Native loader (diffusers: lora.up/down.weight)
        try:
            self.pipe.load_lora_weights(lora_path)
            try:
                self.pipe.fuse_lora(lora_scale=float(alpha))
            except Exception:
                pass
            return
        except Exception as e_native:
            # Fallback: simple A/B safetensors
            try:
                state = load_safetensors(lora_path)
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.pipe.unet.device)
                loaded = _load_simple_lora_into_unet(self.pipe.unet, state, alpha=float(alpha))
                print(f"[lora] loaded simple A/B LoRA into UNet ({loaded} projections).")
                return
            except Exception as e_fallback:
                raise RuntimeError(
                    f"Failed to load LoRA weights from {lora_path}: {e_native} / {e_fallback}"
                )

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
