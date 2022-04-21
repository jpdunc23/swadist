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
               rank: int,
               world_size: int,
               async_op=False) -> List[torch.Tensor]:
    """Sends input tensors from this rank to all other ranks and returns the
    gathered list of all ranks' tensors if aync_op is False or the tensor list
    and a async work handle otherwise.

    """
    send = parameters_to_vector(send)
    recv = [torch.empty_like(send) for _ in range(world_size)]
    handle = dist.all_gather(recv, send, async_op=async_op)

    return Handler(handle), recv
