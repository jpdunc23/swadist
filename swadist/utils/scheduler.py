"""Learning rate schedulers.
"""

from torch.optim.lr_scheduler import LambdaLR

class LinearPolyLR(LambdaLR):
    """Simple wrapper for LambdaLR with a linear (polynomial) decay schedule. Subtracts an
    increasing portion of the initial learning rate `lr0` (weighted by `alpha`) until reaching
    `alpha*lr0`.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapped optimizer.
    alpha: float
        The final multiplicative factor of the initial lr after `decay_epochs` has passed.
    decay_epochs: int
        The number of epochs before multiplicative factor reaches `alpha`. Default: 5.
    verbose: bool
        If True, prints every update.

    """
    def __init__(self,
                 optimizer,
                 alpha,
                 decay_epochs=5,
                 verbose=False):
        self.alpha = alpha
        self.decay_epochs = decay_epochs
        def lr_lambda(epoch):
            if epoch <= self.decay_epochs:
                return 1 - (1 - self.alpha)*epoch/self.decay_epochs
            return alpha
        super().__init__(optimizer=optimizer,
                         lr_lambda=lr_lambda,
                         last_epoch=-1,
                         verbose=verbose)
