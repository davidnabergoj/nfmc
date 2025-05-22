from typing import Optional, Tuple, Union

import torch
from nfmc.algorithms.mh.base import MHKernel
from nfmc.util import compute_divergence_mask, metropolis_acceptance_log_ratio, sum_except_batch, diag_mult


def propose_state(x: torch.Tensor,
                  event_shape: Union[Tuple[int, ...], torch.Size],
                  step_size: float,
                  inv_mass_diag: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, float], torch.Tensor]:
    noise = diag_mult(torch.randn_like(x), inv_mass_diag, event_shape).to(x)
    x_prime = x + step_size * noise
    return x_prime


class RWMHKernel(MHKernel):
    """
    Random-walk Metropolis-Hastings kernel with a centered diagonal Gaussian proposal.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 step_size: float = 0.01,
                 proposal_scale: Optional[torch.Tensor] = None):
        """
        RWMH kernel class constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event.
        :param callable neg_log_prob_target: function that computes the negative of the log target probability density. 
         Receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape 
         `batch_shape`.
        :param float step_size: proposal step size.
        :param Optional[torch.Tensor] proposal_scale: scale of the proposal distribution. Analogous to the inverse of 
         the mass matrix in HMC and MALA.
        """
        super().__init__(event_shape, neg_log_prob_target)
        self.step_size = step_size
        self.proposal_scale = proposal_scale
        if self.proposal_scale is None:
            self.proposal_scale = torch.ones(
                size=(self.event_size,),
                dtype=torch.double
            )

    @property
    def name(self):
        return 'RWMH'

    def step(self, x: torch.Tensor):
        # Propose new state
        x_prime = propose_state(
            x,
            self.event_shape,
            self.step_size,
            self.proposal_scale,
        )

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

        self.increment_n_steps()
        self.increment_n_accepted_transitions(
            int(acceptance_mask.long().sum()))
        self.increment_n_attempted_transitions(n_chains=x.shape[0])

        return x_prime.detach(), acceptance_mask
