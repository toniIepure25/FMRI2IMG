from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import torch


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: int = 4
    target_modules: Optional[Iterable[str]] = None  # reserved for filtering


def _hidden_size_from_name(name: str, block_out_channels) -> int:
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
    Diffusers 0.35.x–friendly:
      1) Force UNet attention to classic AttnProcessor (module-based).
      2) Inject LoRAAttnProcessor (nn.Module) per attention site.
      3) Freeze base UNet; expose LoRA params via AttnProcsLayers for training.
    """
    from diffusers.models.attention_processor import (
        AttnProcessor,
        LoRAAttnProcessor,
    )
    from diffusers.loaders import AttnProcsLayers  # nn.Module wrapper

    unet = pipe.unet
    rank = int(cfg.rank)

    # 1) Force classic attention processors (module-based)
    unet.set_attn_processor(AttnProcessor())

    # 2) Build LoRA processors map
    attn_procs: Dict[str, torch.nn.Module] = {}
    block_out_channels = list(getattr(unet.config, "block_out_channels"))
    unet_cross_dim = getattr(unet.config, "cross_attention_dim", None)

    for name, _cur in unet.attn_processors.items():
        # self-attn (attn1) has no cross-attention
        cross_dim = None if name.endswith("attn1.processor") else unet_cross_dim
        hidden_size = _hidden_size_from_name(name, block_out_channels)

        proc = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            rank=rank,
        )
        attn_procs[name] = proc

    # 3) Attach and freeze base UNet
    unet.set_attn_processor(attn_procs)
    for p in unet.parameters():
        p.requires_grad = False

    # Expose trainable LoRA parameters
    lora_layers = AttnProcsLayers(unet.attn_processors)
    for p in lora_layers.parameters():
        p.requires_grad = True

    pipe._lora_layers = lora_layers  # stash for optimizers
    return unet.attn_processors


def lora_trainable_params(pipe) -> Iterable[torch.nn.Parameter]:
    lora_layers = getattr(pipe, "_lora_layers", None)
    if lora_layers is None:
        return []
    return lora_layers.parameters()


def save_lora_unet(pipe, path: str):
    pipe.unet.save_attn_procs(path)


def load_lora_unet(pipe, path: str, device: Optional[str] = None):
    pipe.unet.load_attn_procs(path, device=device or pipe.device)
