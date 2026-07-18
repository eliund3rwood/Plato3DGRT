"""
src/distributed.py — DDP initialization, rank helpers, and AMP dtype selection.

Designed so the single-GPU path is identical to multi-GPU except for
the launch command and `distributed=True` in build_loader. No logic
elsewhere should branch on world_size directly.
"""

import os
import torch
import torch.distributed as dist
from omegaconf import DictConfig


# ── AMP dtype detection ───────────────────────────────────────────────────────

def resolve_amp_dtype(cfg_dtype: str) -> torch.dtype:
    """
    Resolve the AMP dtype from config, with auto-detection.

    "auto"      → bf16 on sm_80+ (Ampere/Hopper/Blackwell), fp16 otherwise
    "bfloat16"  → always bf16
    "float16"   → always fp16
    "float32"   → disable AMP
    """
    if cfg_dtype == "float32":
        return torch.float32
    if cfg_dtype == "bfloat16":
        return torch.bfloat16
    if cfg_dtype == "float16":
        return torch.float16

    # "auto" — detect from compute capability
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def log_amp_dtype(dtype: torch.dtype, rank: int) -> None:
    if rank == 0:
        needs_scaler = dtype == torch.float16
        print(
            f"[distributed] AMP dtype = {dtype}  "
            f"GradScaler = {'yes' if needs_scaler else 'no (bf16 or fp32)'}"
        )


# ── Distributed setup ─────────────────────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize the process group if launched with torchrun/torch.distributed.
    Returns (rank, world_size, local_rank).
    Falls back gracefully to (0, 1, 0) for single-GPU runs.
    """
    if "RANK" not in os.environ:
        # Single-GPU or non-DDP launch
        return 0, 1, 0

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Global barrier; no-op in single-GPU mode."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ── Rank helpers ──────────────────────────────────────────────────────────────

def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


# ── All-reduce utilities ──────────────────────────────────────────────────────

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / get_world_size()


# ── DDP model wrapping ────────────────────────────────────────────────────────

def maybe_wrap_ddp(module: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """Wrap module in DDP if distributed, otherwise return as-is."""
    if dist.is_available() and dist.is_initialized() and get_world_size() > 1:
        return torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[local_rank], output_device=local_rank
        )
    return module


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP wrapper to access the underlying module for checkpointing."""
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        return module.module
    return module


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_everything(seed: int, rank: int = 0) -> None:
    """
    Seed all RNGs. Each rank gets a unique seed derived from the base seed
    so that data augmentation differs across workers while remaining
    deterministic for a given (seed, rank) pair.
    """
    import random
    import numpy as np

    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
