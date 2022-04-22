"""Utilities for distributed communication.
"""

from typing import List

import torch
import torch.distributed as dist

from torch.nn.utils import parameters_to_vector

__all__ = ['all_gather',
           'all_reduce',
           'is_multiproc']


def is_multiproc():
    return dist.is_initialized() and dist.get_world_size() > 1


class Handler(object):
    """Wrapper class for async work handles returned by torch.distributed
    collective functions.

    """
    def __init__(self, handle):
        self.handle = handle


    def is_completed(self):
        return self.handle is None or self.handle.is_completed()


    def wait(self):
        """Blocks until self.handle is complete.

        """
        if self.handle is not None:
            rank = dist.get_rank()
            self.handle.wait()


def all_gather(*send: torch.Tensor,
               world_size: int,
               async_op=False) -> List[torch.Tensor]:
    """Sends input tensors from this rank to all other ranks and returns an async
    work handle (or None if async_op=False) and the gathered list of all ranks'
    tensors.

    """
    send = parameters_to_vector(send)
    recv = [torch.empty_like(send) for _ in range(world_size)]
    handle = dist.all_gather(recv, send, async_op=async_op)

    return Handler(handle), recv


def all_reduce(*send: torch.Tensor,
               op=dist.ReduceOp.SUM,
               async_op=False) -> List[torch.Tensor]:
    """Sends input tensors from this rank to all other ranks and returns an async
    work handle (or None if async_op=False) and a single tensor from the reduction
    of all ranks' tensors.

    """
    tensor = parameters_to_vector(send)
    handle = dist.all_reduce(tensor, op=op, async_op=async_op)

    return Handler(handle), tensor
