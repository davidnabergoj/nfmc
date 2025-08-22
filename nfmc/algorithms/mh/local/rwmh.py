from typing import Tuple, Union

import torch
from nfmc.algorithms.mh.local.base import LocalMHKernel
from nfmc.util import compute_divergence_mask, metropolis_acceptance_log_ratio


def propose_state(x: torch.Tensor, step_size: torch.Tensor):
    """

    :param torch.Tensor x: event tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor step_size: step size tensor with shape `(n_chains,)` or `()`.
    """
    if len(step_size.shape) == 1:
        step_size = step_size.view(
            step_size.shape[0], *[1] * (len(x.shape) - 1))

    x_prime = x + step_size * torch.randn_like(x)
    return x_prime


class RWMHKernel(LocalMHKernel):
    """
    Random-walk Metropolis-Hastings kernel with a centered standard Gaussian proposal.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 target_acceptance_rate: float = 0.234,
                 **kwargs):
        """
        RWMH kernel class constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event.
        :param callable neg_log_prob_target: function that computes the negative of the log target probability density. 
         Receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape 
         `batch_shape`.
        :param float step_size: proposal step size.
        """
        if 'step_size' not in kwargs:
            event_size = int(torch.prod(torch.as_tensor(event_shape)))
            kwargs['step_size'] = torch.tensor(2.38 ** 2 / event_size)
        super().__init__(
            event_shape,
            neg_log_prob_target,
            target_acceptance_rate=target_acceptance_rate,
            **kwargs
        )

    @property
    def name(self):
        return 'RWMH'

    def step(self,
             x: torch.Tensor,
             update: bool = False) -> torch.Tensor:
        """
        Perform one RWMH transition.

        :param torch.Tensor x: incoming state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        if self.warmup_active:
            step_size = self._dual_averaging.value
        else:
            step_size = self.step_size

        # Propose new state
        x_prime = propose_state(x, step_size)

        # Compute divergence mask
        divergence_mask = compute_divergence_mask(x_prime, self.event_shape)

        n_valid_proposals = int(torch.sum((~divergence_mask).long()))
        self.increment_n_divergences(torch.sum(divergence_mask.long()))

        # Compute acceptance mask
        acceptance_mask = torch.zeros_like(divergence_mask)
        log_prob_accept = metropolis_acceptance_log_ratio(
            -self.neg_log_prob_target(x[~divergence_mask]),
            -self.neg_log_prob_target(x_prime[~divergence_mask]),
            0,
            0
        )
        self.increment_n_calls(2 * n_valid_proposals)
        log_u = torch.rand_like(log_prob_accept).log()
        acceptance_mask[~divergence_mask] = log_u < log_prob_accept
        x[acceptance_mask] = x_prime[acceptance_mask]
        x = x.detach()

        if update:
            self._update(acceptance_mask)

        self.increment_n_steps()
        self.increment_n_accepted_transitions(
            int(acceptance_mask.long().sum())
        )
        self.increment_n_attempted_transitions(n_chains=x.shape[0])

        return x
