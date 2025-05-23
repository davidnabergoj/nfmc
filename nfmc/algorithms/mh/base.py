from typing import Tuple, Union
import torch


class MHKernel:
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
        self.event_shape = event_shape
        self.neg_log_prob_target = neg_log_prob_target

        self._n_steps: int = 0
        self._n_attempted_transitions: int = 0
        self._n_accepted_transitions: int = 0
        self._n_calls: int = 0  # Target density evaluation counter
        self._n_grads: int = 0  # Target density gradient evaluation counter
        # Counts the number of chains that diverged across all steps
        self._n_divergences: int = 0

    @property
    def name(self) -> str:
        raise NotImplementedError

    def reset_statistics(self):
        self._n_steps = 0
        self._n_attempted_transitions = 0
        self._n_accepted_transitions = 0
        self._n_calls = 0
        self._n_grads = 0
        self._n_divergences = 0

    @property
    def acceptance_rate(self):
        if self._n_attempted_transitions == 0:
            return torch.nan
        return self._n_accepted_transitions / self._n_attempted_transitions

    def increment_n_steps(self):
        self._n_steps += 1

    def increment_n_attempted_transitions(self, n_chains: int):
        self._n_attempted_transitions += n_chains
        self._n_attempted_transitions = int(self._n_attempted_transitions)

    def increment_n_accepted_transitions(self, n_accepted_chains: int):
        self._n_accepted_transitions += n_accepted_chains
        self._n_accepted_transitions = int(self._n_accepted_transitions)

    def increment_n_calls(self, n_calls: int):
        self._n_calls += n_calls
        self._n_calls = int(self._n_calls)

    def increment_n_grads(self, n_grads: int):
        self._n_grads += n_grads
        self._n_grads = int(self._n_grads)

    def increment_n_divergences(self, n_divergences: int):
        self._n_divergences += n_divergences
        self._n_divergences = int(self._n_divergences)

    @property
    def event_size(self):
        return int(torch.prod(torch.as_tensor(self.event_shape)))

    def step(self,
             x: torch.Tensor,
             *args,
             update: bool = False,
             **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one kernel transition.

        :param torch.Tensor x: current state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
