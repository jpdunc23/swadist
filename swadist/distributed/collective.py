"""Utilities for distributed communication.
"""

from typing import List

import torch
import torch.distributed as dist

from torch.nn.utils import parameters_to_vector

__all__ = ['all_gather',
           'is_multiproc']


def is_multiproc():
    return dist.is_initialized() and dist.get_world_size() > 1


def all_gather(*send: torch.Tensor,
               rank: int,
               world_size: int) -> List[torch.Tensor]:
    """Send tensors from each rank to all other ranks, and return the received
    tensors in a list.

    """
    send = parameters_to_vector(send)
    recv = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(recv, send)
    return recv
