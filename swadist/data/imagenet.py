"""Load and pre-process ImageNet. Uses pre-processing from Shallue et al., 2019
[arXiv: 1811.03600] and K. Simonyan & A. Zisserman, 2015 [arXiv:1409.1556].

"""

import numpy as np

from torch.utils.data import Subset
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor


__all__ = ['get_imagenet']


def get_imagenet(root_dir='./data',
                 validation=True,
                 test=True,
                 **kwargs):
    datasets = {}

    # see https://gist.github.com/ksimonyan/211839e770f7b538e2d8#description
    mean = [103.939, 116.779, 123.68]

    train_transform = Compose([
        ToTensor(),
        # TODO: add other transformations from Shallue et al.
        # see https://pytorch.org/vision/stable/transforms.html
        Normalize(mean=mean),
    ])

    # get training data
    train = ImageNet(root=root_dir,
                     split='train',
                     transform=train_transform),

    # use same train / val split as from Shallue et al.
    train_idx0 = 50045

    datasets['train'] = Subset(train, indices=np.arange(train_idx0, len(train)))

    eval_transform = Compose([
        ToTensor(),
        # TODO: add other transformations from Shallue et al.
        # see https://pytorch.org/vision/stable/transforms.html
        Normalize(mean=mean),
    ])

    if validation:
        datasets['validation'] = Subset(
            dataset=ImageNet(root=root_dir,
                             split='train',
                             transform=eval_transform),
            indices=np.arange(train_idx0)
        )

    if test:
        datasets['test'] = ImageNet(root=root_dir,
                                    split='train',
                                    transform=eval_transform)

    return datasets
