# src/fmri2image/generation/sd_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Sequence
import contextlib
import torch

# add near the top of the file
# --- add/replace these helpers near the top of sd_wrapper.py ---
import math
from safetensors.torch import load_file as load_safetensors

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

def _attach_simple_lora_linear(module, proj_name: str, rank: int, alpha: float):
    """
    Attach simple LoRA A/B to the Linear at proj_name and monkey-patch its forward:
      y = W x + scale * B(A(x)), with scale = alpha / rank
    Idempotent: if already attached, reuses existing layers.
    """
    lin = _get_linear(module, proj_name)
    if lin is None:
        return None

    # reuse if present
    lora_A = getattr(module, f"{proj_name}_lora_A", None)
    lora_B = getattr(module, f"{proj_name}_lora_B", None)
    if lora_A is None or lora_B is None:
        in_f, out_f = lin.in_features, lin.out_features
        lora_A = torch.nn.Linear(in_f, rank, bias=False)
        lora_B = torch.nn.Linear(rank, out_f, bias=False)
        torch.nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(lora_B.weight)
        setattr(module, f"{proj_name}_lora_A", lora_A)
        setattr(module, f"{proj_name}_lora_B", lora_B)

        scale_val = float(alpha) / float(rank)

        orig_forward = lin.forward
        # Capture with unambiguous names to avoid collisions
        def lora_forward(x, _orig_fwd=orig_forward, _A=lora_A, _B=lora_B, _scale=scale_val):
            return _orig_fwd(x) + _B(_A(x)) * _scale

        lin.forward = lora_forward  # monkey-patch

    return lora_A, lora_B

def _load_simple_lora_into_unet(unet, state: dict, alpha: float):
    """
    Load simple A/B LoRA weights from safetensors into UNet attention projections.
    Expects keys like:
      '{module_path}.to_q.lora_A.weight' and '{module_path}.to_q.lora_B.weight'
    """
    loaded = 0
    for name, module in unet.named_modules():
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            key_A = f"{name}.{proj}.lora_A.weight"
            key_B = f"{name}.{proj}.lora_B.weight"
            if key_A in state and key_B in state:
                rank = int(state[key_A].shape[0])
                pair = _attach_simple_lora_linear(module, proj, rank=rank, alpha=alpha)
                if pair is None:
                    continue
                lora_A, lora_B = pair
                with torch.no_grad():
                    lora_A.weight.copy_(state[key_A].to(lora_A.weight.device, dtype=lora_A.weight.dtype))
                    lora_B.weight.copy_(state[key_B].to(lora_B.weight.device, dtype=lora_B.weight.dtype))
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
        Try native Diffusers LoRA loader; if it fails (old/simple A/B format),
        fall back to custom UNet injection.
        """
        # 1) Native loader (diffusers format: lora.up/down.weight)
        try:
            self.pipe.load_lora_weights(lora_path)
            try:
                # fuse with scale if available (older versions may not have it)
                self.pipe.fuse_lora(lora_scale=float(alpha))
            except Exception:
                pass
            return
        except Exception as e_native:
            # 2) Fallback: our simple A/B safetensors format
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
