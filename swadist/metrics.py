"""Loss functions and non-grad evaluation metrics.
"""

import warnings

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import vector_to_parameters

from .distributed import all_gather


__all__ = ['accuracy',
           'CodistillationLoss']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Adapted from https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k / batch_size)
        if len(res) == 1:
            return res[0]
        return res


_transform_fns = {
    'argmax': lambda x: x.argmax(dim=1),
    'softmax' : lambda x: x.softmax(dim=1),
}


class CodistillationLoss(object):
    """Manages model replicas and computes codistillation loss.

    Parameters
    __________
    loss_fn: Union[Callable, torch.nn.modules.loss._Loss]
        Differentiable loss function to use when comparing output to average outputs from
        replicas.
    model: torch.nn.Module
        Model to replicate.
    rank: int
        The index of the current worker, from `0` to `world_size - 1`.
    world_size: int
        The number of distributed workers. If greater than 1, then .
    sync_freq: int, optional
        Frequency of replica synchronization across ranks in number of training steps.
    async_op: bool, optional
        If True, gather replica parameters asynchronously and continue with parameters from the
        last time they were gathered.
    transform: str, optional
        Name of a method to transform the mean replica output before calling `loss_fn`.
    debug: bool, optional
        If True, print step when gathering parameters begins and when the parameters are ready.

    """
    def __init__(self,
                 loss_fn,
                 model,
                 rank,
                 world_size,
                 sync_freq=50,
                 transform=None,
                 async_op=True,
                 debug=False):

        self.__name__ = 'CodistillationLoss'
        self.loss_fn = loss_fn
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.async_op = async_op
        self.sync_freq = sync_freq
        self.transform = transform
        self.replicas = []
        self.step = 0
        self._handle = None
        self._flat_params = None
        self._debug = debug

        for i in range(world_size):
            if i == self.rank:
                self.replicas.append(model)
            else:
                self.replicas.append(deepcopy(model))


    def __call__(self, x, output):
        """Calculate codistillation component of loss, i.e. the original loss function
        of model output and average model outputs from other replicas in the collective.

        Parameters
        __________
        x: torch.Tensor
            A batch of input observations.
        output: torch.Tensor
            A batch of model outputs from this rank's model.

        """

        if self.step % self.sync_freq == 0:
            if self._handle is not None:
                warnings.warn(f'It\'s taking longer than sync_freq={self.sync_freq} steps '
                              'to gather replica parameters. Calling wait() on async work handle.')
                self._handle.wait()
            else:
                # initialize synchronization
                self.gather_params()

        mean_output = self.replica_mean_output(x, output)

        if self.transform is not None:
            mean_output = _transform_fns[self.transform](mean_output)

        # update the step counter
        self.step += 1

        return self.loss_fn(output, mean_output)


    def replica_mean_output(self, x, output):
        # get mean predictions from model replicas

        self.update_replicas(force=self.step == 0)

        with torch.no_grad():

            # average over other ranks
            mean_output = torch.empty_like(output)

            for i, model in enumerate(self.replicas):

                if i != self.rank:
                    mean_output += model(x) / (self.world_size - 1)

        return mean_output


    def gather_params(self):

        if self._debug:
            print(f'\nGathering replica params at step {self.step}')

        # construct the Tensor list to all_gather into
        params = self.replicas[self.rank].parameters()

        # flat_params[i] is a single Tensor with all params from rank i
        self._handle, self._flat_params_list = all_gather(*params,
                                                          rank=self.rank,
                                                          world_size=self.world_size,
                                                          async_op=self.async_op)


    def update_replicas(self, force=False):

        if self._handle is not None:

            if force or self._handle.is_completed():

                self._handle.wait()

                if self._debug:
                    print(f'\nUpdating replica params at step {self.step}')

                # update the replicas' parameters
                for i, flat_params in enumerate(self._flat_params_list):
                    if i != self.rank:
                        vector_to_parameters(flat_params, self.replicas[i].parameters())

                self._handle = None
                self._flat_params_list = None
