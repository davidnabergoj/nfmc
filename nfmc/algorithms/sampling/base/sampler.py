from typing import Tuple, Union

import torch

from nfmc.algorithms.mh.base import MarkovKernel
from nfmc.algorithms.util.samples import Samples


class MCMCSampler:
    """
    MCMC sampler class.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 kernel: MarkovKernel,
                 **kwargs):
        self.event_shape = event_shape
        self.kernel = kernel

    @property
    def name(self) -> str:
        return "Generic MH sampler"

    def calls_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel._n_calls / elapsed_time_seconds
        return torch.nan

    def grads_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel._n_grads / elapsed_time_seconds
        return torch.nan

    def warmup(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> Samples:
        """
        Optimizes kernel parameters.
        """
        raise NotImplementedError

    def sample(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> Samples:
        """
        Samples with a fixed kernel.
        """
        raise NotImplementedError
