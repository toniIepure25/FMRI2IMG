from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import torch

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: int = 4  # kept for future scaling; not all APIs expose it
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

def _make_lora_processor(cur_proc, hidden_size: int, cross_dim: Optional[int], rank: int):
    """
    Compat layer pentru diverse versiuni de diffusers:
    - AttnProcessor2_0 -> LoRAAttnProcessor2_0() **fără argumente** (0.35.x)
    - AttnProcessor    -> LoRAAttnProcessor(hidden_size, cross_dim, rank) (legacy)
    """
    from diffusers.models.attention_processor import (
        AttnProcessor, AttnProcessor2_0,
        LoRAAttnProcessor, LoRAAttnProcessor2_0,
    )

    if isinstance(cur_proc, AttnProcessor2_0):
        # 0.35.x: constructorul nu acceptă args/kwargs -> fără argumente
        proc = LoRAAttnProcessor2_0()
        # Unele versiuni expun scalarea prin 'lora_scale' / 'alpha'; dacă există, o setăm.
        if hasattr(proc, "alpha"):
            try:
                proc.alpha = rank  # mic boost similar network_alpha
            except Exception:
                pass
        return proc
    else:
        # Versiuni vechi: LoRAAttnProcessor cu semnătura clasică
        try:
            return LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                rank=rank,
            )
        except TypeError:
            # fallback ultra-legacy (poziționale)
            return LoRAAttnProcessor(hidden_size, cross_dim, rank)

def apply_lora_to_sd(pipe, cfg: LoRAConfig) -> Dict[str, torch.nn.Module]:
    """
    Inject LoRA în toate atențiile UNet-ului și marchează doar LoRA ca trainabil.
    """
    unet = pipe.unet
    rank = int(cfg.rank)
    attn_procs: Dict[str, torch.nn.Module] = {}

    block_out_channels = list(getattr(unet.config, "block_out_channels"))
    unet_cross_dim = getattr(unet.config, "cross_attention_dim", None)

    for name, cur in unet.attn_processors.items():
        # attn1 = self-attn (fără cross), attn2 = cross-attn (cu text)
        cross_dim = None if name.endswith("attn1.processor") else unet_cross_dim
        hidden_size = _hidden_size_from_name(name, block_out_channels)
        attn_procs[name] = _make_lora_processor(cur, hidden_size, cross_dim, rank)

    unet.set_attn_processor(attn_procs)

    # îngheață UNet, pornește doar LoRA
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
    # salvează doar proc-urile de atenție (format diffusers)
    pipe.unet.save_attn_procs(path)

def load_lora_unet(pipe, path: str, device: Optional[str] = None):
    pipe.unet.load_attn_procs(path, device=device or pipe.device)
