# src/fmri2image/models/adapters/lora.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import torch

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: int = 4
    # keep for future: target_modules at text encoder level
    target_modules: Optional[Iterable[str]] = None


def _hidden_size_from_name(name: str, block_out_channels) -> int:
    """
    Map attention processor name -> hidden_size using UNet2DConditionModel config.
    Works for down_blocks, up_blocks, mid_block.
    """
    if name.startswith("mid_block"):
        return block_out_channels[-1]

    if name.startswith("up_blocks."):
        # up_blocks.<i>.attentions.<j>...
        try:
            i = int(name.split(".")[1])
        except Exception:
            # fallback conservativ
            i = 0
        # up_blocks merg invers față de down_blocks
        return list(reversed(block_out_channels))[i]

    if name.startswith("down_blocks."):
        try:
            i = int(name.split(".")[1])
        except Exception:
            i = 0
        return block_out_channels[i]

    # fallback (nu ar trebui să ajungem aici la SD 1.5/SDXL)
    return block_out_channels[0]


def apply_lora_to_sd(pipe, cfg: LoRAConfig) -> Dict[str, torch.nn.Module]:
    """
    Inject LoRA processors în toate straturile de cross/self attention ale UNet-ului.
    Folosește API-ul oficial diffusers: set_attn_processor / save_attn_procs / load_attn_procs.
    """
    try:
        from diffusers.models.attention_processor import LoRAAttnProcessor
    except Exception as e:
        raise RuntimeError("diffusers>=0.16 is required for LoRAAttnProcessor") from e

    unet = pipe.unet
    attn_procs = {}
    rank = int(cfg.rank)

    # unet.config are fields: block_out_channels (list[int]) și cross_attention_dim (int/None)
    block_out_channels = list(getattr(unet.config, "block_out_channels"))
    unet_cross_dim = getattr(unet.config, "cross_attention_dim", None)

    for name in unet.attn_processors.keys():
        # attn1 = self-attention (fără cross); attn2 = cross-attention (cu cross dim)
        cross_attention_dim = None if name.endswith("attn1.processor") else unet_cross_dim
        hidden_size = _hidden_size_from_name(name, block_out_channels)

        attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(attn_procs)

    # freeze toate param-urile UNet cu excepția LoRA
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
    """
    Salvează doar LoRA attn_procs din UNet într-un fișier compatibil diffusers
    (.safetensors / .bin, în funcție de extensie).
    """
    pipe.unet.save_attn_procs(path)


def load_lora_unet(pipe, path: str, device: Optional[str] = None):
    """
    Încarcă LoRA attn_procs în UNet (poți compune mai multe dacă vrei).
    """
    pipe.unet.load_attn_procs(path, device=device or pipe.device)
