from typing import Tuple, Union

import torch

from nfmc.algorithms.mh.base import LocalMHKernel
from nfmc.algorithms.sampling.base.sampler_data import MCMCOutput


class MHSampler:
    """
    Sampler class for Metropolis-Hastings algorithms.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 kernel: LocalMHKernel,
                 **kwargs):
        self.event_shape = event_shape
        self.kernel = kernel
    
    @property
    def name(self) -> str:
        return "Generic MH sampler"

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        """
        Optimizes kernel parameters.
        """
        raise NotImplementedError

    def sample(self, 
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        """
        Samples with a fixed kernel.
        """
        raise NotImplementedError
