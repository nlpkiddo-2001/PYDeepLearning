"""
FSDP Module for parallelism

This module is for one layer in one gpu
"""
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, FSDPModule
from torch.distributed.tensor import DTensor

@dataclass
class DistInfo:
    rank : int = 0
    world_size : int = 1
    local_rank : int = 0
    device : str = "cuda"
    enabled : bool = False # False means plain single process when not initiated with torchrun

    @property
    def is_main(self) -> bool:
        return self.rank == 0 # only rank 0 logs and writes files


def init_distributed(device : str = "cuda") -> DistInfo:
    if "RANK" not in os.environ:
        return DistInfo(device=device)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if device == "cpu":
        # cpu backend is only for testing and debuging purpose
        dist.init_process_group("gloo")
        return DistInfo(rank, world_size, local_rank, "cpu", True)

    torch.cuda.set_device(local_rank) # order matters, so set the current gpu always, so every plain cuda means MY GPU
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return DistInfo(rank, world_size, local_rank, "cuda", True)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def shard_model(
        model,
        param_dtype : torch.dtype = torch.bfloat16,
        reduce_dtype: torch.dtype = torch.float32,
        reshard_after_forward: bool = True
):
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp, reshard_after_forward=reshard_after_forward)

    fully_shard(model, mp_policy=mp, reshard_after_forward=reshard_after_forward)

    if model.config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight, "sharding untied embed_tokens/lm_head"


def all_reduce_sum(values: float, info: DistInfo) -> float:
    """
    Sum a Python number over all ranks; every rank gets the total back
    """
    if not info.enabled:
        return values

    t = torch.tensor(values, dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def set_grad_sync(model, enabled: bool):
    """
    With sync off, backward() still computes grads but skips the reduce-scatter and
    accumulates UNSHARDED grads locally.
    """
    if isinstance(model, FSDPModule):
        model.set_requires_gradient_sync(enabled)


def full_tensor(t: torch.Tensor) -> torch.Tensor:
    """
     clip_grad_norm_ on DTensor grads returns a DTensor (the GLOBAL norm, same on every rank).
    .full_tensor() makes it a plain tensor for isfinite() and .item().
    """
    return t.full_tensor() if isinstance(t, DTensor) else t

