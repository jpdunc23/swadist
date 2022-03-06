"""Loss functions and evaluation utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_cross_entropy(output, target):
    """Computes the binary cross entropy between output and target
    """
    target_binary = F.one_hot(target, 10).float()
    bce = nn.BCEWithLogitsLoss()
    return bce(output, target_binary)


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
