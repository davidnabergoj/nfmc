from typing import Any, Dict, Tuple, Union
import torch

from nfmc.algorithms.mh.data import MHOutput


class MHKernel:
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable):
        """
        Base MCMC kernel class for Metropolis-Hastings algorithms.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable target: function that computes the negative log probability density of the target distribution.
         It receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape
         `batch_shape`.
        """
        self.event_shape = event_shape
        self.neg_log_prob_target = neg_log_prob_target

        self._n_calls: int = 0  # Target density evaluation counter
        self._n_grads: int = 0  # Target density gradient evaluation counter


    @property
    def event_size(self):
        return int(torch.prod(torch.as_tensor(self.event_shape)))
    
    def step(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one kernel transition.

        Note: the returned state tensor is the proposed state, not the new state after the accept/reject step.

        :param torch.Tensor x: current state tensor with shape `(n_chains, *event_shape)`.
        :return: proposed state tensor with shape `(n_chains, *event_shape)` and acceptance mask tensor with shape 
         `(n_chains)`.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class MHSampler:
    """
    Sampler class for Metropolis-Hastings algorithms.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 kernel: MHKernel,
                 **kwargs):
        self.event_shape = event_shape
        self.kernel = kernel
    
    @property
    def name(self) -> str:
        return "Generic MH sampler"

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MHOutput:
        """
        Optimizes kernel parameters.
        """
        raise NotImplementedError

    def sample(self, 
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MHOutput:
        """
        Samples with a fixed kernel.
        """
        raise NotImplementedError
