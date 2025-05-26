from typing import Tuple, Union
import torch

from nfmc.algorithms.kernel import MarkovKernel


class MHKernel(MarkovKernel):
    """
    Base MCMC kernel class for Metropolis-Hastings algorithms.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable):
        """
        MHKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable target: function that computes the negative log probability density of the target distribution.
         It receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape
         `batch_shape`.
        """
        super().__init__(event_shape, neg_log_prob_target)

        self._n_attempted_transitions: int = 0
        self._n_accepted_transitions: int = 0

    def reset_statistics(self):
        super().reset_statistics()
        self._n_attempted_transitions = 0
        self._n_accepted_transitions = 0

    @property
    def acceptance_rate(self):
        if self._n_attempted_transitions == 0:
            return torch.nan
        return self._n_accepted_transitions / self._n_attempted_transitions

    def increment_n_attempted_transitions(self, n_chains: int):
        self._n_attempted_transitions += n_chains
        self._n_attempted_transitions = int(self._n_attempted_transitions)

    def increment_n_accepted_transitions(self, n_accepted_chains: int):
        self._n_accepted_transitions += n_accepted_chains
        self._n_accepted_transitions = int(self._n_accepted_transitions)
