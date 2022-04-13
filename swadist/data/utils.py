"""Dataset utility functions.
"""

__all__ = ['per_channel_mean_and_std']


def per_channel_mean_and_std(dataset, max_val=255.):
    """Returns the mean and standard deviation of each channel across images in `dataset`.
    """
    data = dataset.data / max_val
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std
