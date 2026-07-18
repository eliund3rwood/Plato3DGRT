"""
src/data/loader.py — Distributed-ready DataLoader factory.

Single-GPU and multi-GPU code paths differ only by the `distributed` flag.
Call sampler.set_epoch(epoch) every epoch when distributed=True.
"""

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def build_loader(
    dataset: Dataset,
    micro_batch: int,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DistributedSampler | None]:
    """
    Build a DataLoader with an optional DistributedSampler.

    Returns:
        loader:  DataLoader ready for training
        sampler: DistributedSampler (call set_epoch each epoch) or None
    """
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        effective_shuffle = False  # sampler handles it
    else:
        sampler = None
        effective_shuffle = shuffle

    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=effective_shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        # "fork" (Linux default) inherits CUDA context → SIGSEGV in DDP workers
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    return loader, sampler
