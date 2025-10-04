from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import torch

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: int = 4
    target_modules: Optional[Iterable[str]] = None  # kept for future

def _hidden_size_from_name(name: str, block_out_channels) -> int:
    """Map attention-processor name -> hidden_size via UNet config."""
    if name.startswith("mid_block"):
        return block_out_channels[-1]
    if name.startswith("up_blocks."):
        try:
            i = int(name.split(".")[1])
        except Exception:
            i = 0
        return list(reversed(block_out_channels))[i]
    if name.startswith("down_blocks."):
        try:
            i = int(name.split(".")[1])
        except Exception:
            i = 0
        return block_out_channels[i]
    return block_out_channels[0]

def apply_lora_to_sd(pipe, cfg: LoRAConfig) -> Dict[str, torch.nn.Module]:
    """
    Inject LoRA processors în toate straturile de attention ale UNet-ului.
    Suportă atât AttnProcessor (vechi) cât și AttnProcessor2_0 (nou).
    """
    try:
        from diffusers.models.attention_processor import (
            AttnProcessor, AttnProcessor2_0,
            LoRAAttnProcessor, LoRAAttnProcessor2_0,
        )
    except Exception as e:
        raise RuntimeError("Please install a compatible `diffusers` version.") from e

    unet = pipe.unet
    attn_procs: Dict[str, torch.nn.Module] = {}
    rank = int(cfg.rank)

    block_out_channels = list(getattr(unet.config, "block_out_channels"))
    unet_cross_dim = getattr(unet.config, "cross_attention_dim", None)

    for name, cur in unet.attn_processors.items():
        hidden_size = _hidden_size_from_name(name, block_out_channels)
        cross_dim = None if name.endswith("attn1.processor") else unet_cross_dim

        # alege varianta LoRA corectă după tipul curent
        if isinstance(cur, AttnProcessor2_0):
            # API: LoRAAttnProcessor2_0(hidden_size, cross_attention_dim=None, rank=4, network_alpha=None)
            attn_procs[name] = LoRAAttnProcessor2_0(hidden_size, cross_dim, rank)
        else:
            # API: LoRAAttnProcessor(hidden_size, cross_attention_dim=None, rank=4, network_alpha=None)
            attn_procs[name] = LoRAAttnProcessor(hidden_size, cross_dim, rank)

    # înlocuiește și setează parametrii LoRA ca trainable
    unet.set_attn_processor(attn_procs)
    for p in unet.parameters():
        p.requires_grad = False
    for proc in unet.attn_processors.values():
        for p in proc.parameters():
            p.requires_grad = True
    return unet.attn_processors

def lora_trainable_params(pipe) -> Iterable[torch.nn.Parameter]:
    for proc in pipe.unet.attn_processors.values():
        for p in proc.parameters():
            if p.requires_grad:
                yield p

def save_lora_unet(pipe, path: str):
    """Salvează doar LoRA attn_procs din UNet (compatibil diffusers)."""
    pipe.unet.save_attn_procs(path)

def load_lora_unet(pipe, path: str, device: Optional[str] = None):
    """Încarcă LoRA attn_procs în UNet (poți compune mai multe dacă vrei)."""
    pipe.unet.load_attn_procs(path, device=device or pipe.device)
