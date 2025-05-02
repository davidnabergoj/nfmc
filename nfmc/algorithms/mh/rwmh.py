from typing import Optional, Tuple, Union

import torch
from nfmc.algorithms.mh.base import MHKernel
from nfmc.util import metropolis_acceptance_log_ratio, sum_except_batch


def propose_state(x: torch.Tensor,
                  event_shape: Union[Tuple[int, ...], torch.Size],
                  step_size: float,
                  inv_mass_diag: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, float], torch.Tensor]:
    batch_shape = x.shape[:-len(event_shape)]
    event_size = int(torch.prod(torch.as_tensor(event_shape)))
    noise = torch.multiply(
        torch.randn(size=(*batch_shape, event_size)),
        inv_mass_diag[[None] * len(batch_shape)]
    ).view_as(x)
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
        step_size = step_size
        self.proposal_scale = proposal_scale
        if self.proposal_scale is None:
            self.proposal_scale = torch.ones(
                size=(self.event_size,),
                dtype=torch.double
            )

    def step(self, x: torch.Tensor):
        # Propose new state
        x_prime = propose_state(
            x,
            self.event_shape,
            self.step_size,
            self.proposal_scale,
            self.neg_log_prob_target
        )

        # Compute divergence mask
        divergence_mask = sum_except_batch(
            (~torch.isfinite(x_prime)).long(), self.event_shape
        ) > 0
        n_valid = int(torch.sum((~divergence_mask).long()))

        # Compute acceptance mask
        acceptance_mask = torch.zeros_like(divergence_mask)
        log_prob_accept = metropolis_acceptance_log_ratio(
            -self.neg_log_prob_target(x[~divergence_mask]),
            -self.neg_log_prob_target(x_prime[~divergence_mask]),
            0,
            0
        )
        self._n_calls += 2 * n_valid
        log_u = torch.rand_like(log_prob_accept).log()
        acceptance_mask[~divergence_mask] = log_u < log_prob_accept

        return x_prime.detach(), acceptance_mask
