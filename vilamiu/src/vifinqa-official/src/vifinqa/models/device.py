
from __future__ import annotations

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def pick_attn_implementation(device: str) -> str:
    """SDPA is the portable default; FlashAttention remains an explicit opt-in."""
    del device
    return "sdpa"
