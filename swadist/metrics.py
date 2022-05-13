"""Loss functions and non-grad evaluation metrics.
"""

import warnings
import itertools

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import vector_to_parameters

from .distributed import all_gather


__all__ = ['accuracy',
           'CodistillationLoss',
           'SWADistLoss']


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


class ReplicaLoss(object):
    """Base class for losses computed via model replicas.

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
        Frequency of replica updates across ranks, in number of training steps. If `sync_replicas`
        is 0, replicas are sync'ed only once (at initialization), but they can later be updated
        manually using `update_replicas()`, with `force=True` to block if `async_op` is True.
    transform: str, optional
        Name of a method to transform the mean replica output before calling `loss_fn`.
    async_op: bool, optional
        If True, update replica parameters asynchronously and continue with parameters from the
        last time they were updated.
    debug: bool, optional
        If True, print step when gathering parameters begins and when the parameters are ready.

    """
    def __init__(self,
                 loss_fn,
                 _model,
                 rank,
                 world_size,
                 sync_freq=50,
                 transform='softmax',
                 async_op=True,
                 debug=False):

        self.loss_fn = loss_fn
        self._model = _model
        self.rank = rank
        self.world_size = world_size
        self.sync_freq = sync_freq
        self.transform = transform
        self.async_op = async_op
        self.step = 0
        self.replicas = []
        self.debug = debug
        self._handle = None
        self._flat_params = None

        if self.sync_freq is None:
            self.sync_freq = 0

        for i in range(self.world_size):
            if i == self.rank:
                self.replicas.append(self._model)
            else:
                self.replicas.append(deepcopy(self._model))
                for param in self.replicas[-1].parameters():
                    param.detach()

        # block on cpu until all ranks are ready
        torch.cuda.synchronize()

        # force update replicas
        self.update_replicas(force=True)


    def replica_mean_output(self, x, output):
        """Get mean predictions from saved model replicas.
        """

        # update the step counter
        self.step += 1

        if min(self.sync_freq, self.step - 1) > 0 and (self.step - 1) % self.sync_freq == 0:

            if self._handle is not None:
                warnings.warn(
                    f'Rank {self.rank} at codist step {self.step}: It\'s taking longer than '
                    f'sync_freq={self.sync_freq} steps to gather replica parameters. '
                    'Calling wait() on the async work handle.'
                )
                self.update_replicas(force=True)
            else:
                self.update_replicas()

        if self._handle is not None:
            # check on status of update
            self.update_replicas()

        with torch.no_grad():

            # average over other ranks
            mean_output = torch.empty_like(output)

            for i, model in enumerate(self.replicas):

                if i != self.rank:
                    model.eval()
                    mean_output += model(x) / (self.world_size - 1)

            if self.transform is not None:
                mean_output = _transform_fns[self.transform](mean_output)

        assert not mean_output.requires_grad

        return mean_output


    def _all_gather_params(self, force=None):
        if force == None:
            force = self.async_op

        if self.debug:
            print(f'\nRank {self.rank} `_all_gather_params` at {self.__name__} step {self.step}')

        # move params to gpu
        params = [param.to(self.rank).detach() for param in self.replicas[self.rank].parameters()]

        # _flat_params_list[i] is a single Tensor with all params from rank i
        self._handle, self._flat_params_list = all_gather(*params,
                                                          world_size=self.world_size,
                                                          async_op=force)


    def update_replicas(self, force=False):
        """Starts comm to update replicas, if need be, and updates replicas if comm has
        finished.

        """
        with torch.no_grad():

            if self._handle is None:
                # begin comm
                self._all_gather_params(force=force)

            # update params if comm has finished
            if self._handle.is_completed():

                if self.debug:
                    print(f'\nRank {self.rank} `update_replicas`: '
                          f'async handle complete at {self.__name__} step {self.step}')

                # update the replicas' parameters
                for i, flat_params in enumerate(self._flat_params_list):
                    if i != self.rank:
                        vector_to_parameters(flat_params, self.replicas[i].parameters())

                self._handle = None
                self._flat_params_list = None



class CodistillationLoss(ReplicaLoss):
    """Manages model replicas and computes codistillation loss.

    See `ReplicaLoss` for params.

    """
    def __init__(self,
                 loss_fn,
                 model,
                 rank,
                 world_size,
                 sync_freq=50,
                 transform='softmax',
                 async_op=True,
                 debug=False):
        self.model = model
        self.__name__ = 'CodistillationLoss'
        super(CodistillationLoss, self).__init__(loss_fn,
                                                 model,
                                                 rank,
                                                 world_size,
                                                 sync_freq,
                                                 transform,
                                                 async_op,
                                                 debug)

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
        mean_output = self.replica_mean_output(x, output)

        return self.loss_fn(output, mean_output)



class SWADistLoss(ReplicaLoss):
    """Manages `torch.optim.swa_utils.AveragedModel` replicas and computes codist loss.

    Parameters
    __________
    max_averaged: int, optional
        The max number of recent epoch-end model states to use in the `swa_model` average. Default
        (None) uses all of the epochs, starting from the epoch that ended prior to the SWADist phase.
    swa_replicas: bool, optional
        If True, replicas are `torch.optim.swa_utils.AveragedModel`. If False, they are whatever
        `model` is.
    swa_inference: bool, optional
        If True, evaluation is done with the SWA model. If False, it's done with `model`.

    See `ReplicaLoss` for other params.

    """
    def __init__(self,
                 loss_fn,
                 model,
                 rank,
                 world_size,
                 sync_freq=50,
                 max_averaged=None,
                 swa_replicas=False,
                 swa_inference=True,
                 transform='softmax',
                 async_op=True,
                 debug=False):
        self.__name__ = 'SWADistLoss'

        # TODO: finish implementing `swa_inference`

        self.rank = rank
        self.model = model
        self.max_avgd = max_averaged
        self.swa_replicas = swa_replicas
        self.swa_inference = swa_inference

        # TODO: check inputs
        # if self.max_avgd is not None and self.max_avgd > 1 not self.swa_inference and :
        #     assert self.swa_replicas or (), \
        #         ('SWADistLoss with max_averaged <=1 and swa_replicas == False'
        #          'is just codistillation. Use CodstillationLoss instead.')


        # create the AveragedModel
        if self.max_avgd is not None and self.max_avgd > 1:
            avg_fn = partial(self._avg_fn,
                             max_averaged=torch.tensor(self.max_avgd, dtype=torch.long))
        else:
            avg_fn = None

        self.swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn)

        if self.swa_replicas:
            # between epochs, we'll update swadist_model_cp with current self.model
            # state immediately before syncing
            _model = self._cp_swa_model()
        else:
            _model = self.model

        if self.swa_inference:
            self.swadist_model = self.swa_model
        else:
            self.swadist_model = self.model

        super(SWADistLoss, self).__init__(loss_fn,
                                          _model,
                                          rank,
                                          world_size,
                                          sync_freq,
                                          transform,
                                          async_op,
                                          debug)

        # stores up to `max_averaged` recent epoch-end models
        self.epoch_models = []


    @staticmethod
    def _avg_fn(averaged_modelparameter,
                _modelparameter,
                num_averaged,
                max_averaged=None):
        """Like the default torch.optim.swa_utils.AveragedModel.avg_fn, but handles
        windowed averaging.

        """

        if max_averaged is not None:
            max_averaged.to(num_averaged.device)

        if max_averaged is None or num_averaged < max_averaged:
            return averaged_modelparameter + \
                (_modelparameter - averaged_modelparameter) / (num_averaged + 1)
        else:
            # _modelparameter is new model - oldest model in current avg
            return averaged_modelparameter + _modelparameter / max_averaged


    def __call__(self, x, output):
        """Calculate SWADist component of loss.

        Parameters
        __________
        x: torch.Tensor
            A batch of input observations.
        output: torch.Tensor
            A batch of model outputs from this rank's model.

        """
        if (self.swa_replicas and
            # anticipate self.step += 1 in self.replica_mean_output
            min(self.sync_freq, self.step) > 0 and
            self.step % self.sync_freq == 0):

            if self.debug:
                print(f'\nRank {self.rank} syncing replica params at SWADist step {self.step}')

            # update self.replicas[self.rank] before next replica sync

            if self.max_avgd is not None and len(self.epoch_models) == self.max_avgd:
                model = self._model_subtract(deepcopy(self.epoch_models[0]), self.model)
            else:
                model = self.model

            self.replicas[self.rank].update_parameters(model)

        mean_output = self.replica_mean_output(x, output)

        return self.loss_fn(output, mean_output)


    def update_swa_parameters(self):
        """Updates this rank's `AveragedModel` with the current parameters of
        `self.model`. If `self.max_avgd` is not None, then only the most recent
        `max_averaged` models are included in the average. Called whenever
        `Trainer.evaluate` is called.

        """
        with torch.no_grad():
            if self.debug:
                print(f'\nRank {self.rank} updating `AveragedModel` params '
                      f'x{self.swa_model.n_averaged + 1}')

            if self.max_avgd is not None and len(self.epoch_models) == self.max_avgd:

                # window averaged SWA

                if self.debug:
                    print(f'\nRank {self.rank} updating `AveragedModel` params (windowed)')

                self.epoch_models.append(deepcopy(self.model))

                # subtract oldest model's params from new model's params
                self._model_subtract(self.epoch_models[0], self.model)

                # update the AveragedModel with the new model's remainder
                self.swa_model.update_parameters(self.epoch_models[0])

                # done with the oldest model, delete
                del self.epoch_models[0]

            else:
                assert self.max_avgd is None or len(self.epoch_models) < self.max_avgd

                # normal SWA
                if self.debug:
                    print(f'\nRank {self.rank} updating `AveragedModel` params (standard)')

                self.swa_model.update_parameters(self.model)

                if self.max_avgd is not None:
                    self.epoch_models.append(deepcopy(self.model))

            if self.swa_replicas:
                self.replicas[self.rank] = self._cp_swa_model()


    def _zip_params(self, *models):
        """Input should never include swa_model.

        """
        params = [(itertools.chain(m.parameters(), m.buffers())
                   if self.swa_model.use_buffers else m.parameters())
                  for m in models if m is not None]
        return zip(*params)


    def _model_subtract(self, subtract_model, from_model):
        """In-place update of subtract_model's parameters with the difference between
        from_model and itself.

        """
        with torch.no_grad():
            # subtract oldest model's params from new model's params
            for p_model0, p_model in self._zip_params(subtract_model, from_model):
                device = p_model.device
                p_model0.to(device)
                p_model0.detach().mul_(-1).add_(p_model.detach())

            return subtract_model


    def _cp_swa_model(self):
        """Returns a modifield copy of `self.swa_model`. Used at initialization and
        whenever `update_swa_parameters` is called.

        """
        with torch.no_grad():
            # this cp will be updated with current self.model params right before sync
            swa_model_cp = deepcopy(self.swa_model)

            # swa_model_cp will average batch normalization buffers as training proceeds
            swa_model_cp.use_buffers = True

        return swa_model_cp
