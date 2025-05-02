from typing import Tuple, Union
import torch
from nfmc.algorithms.mh.base import MHKernel
from nfmc.util import metropolis_acceptance_log_ratio, sum_except_batch


class IMHKernel(MHKernel):
    """
    Independent Metropolis-Hastings kernel.
    """
    def __init__(self, 
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 proposal_log_prob: callable,
                 proposal_sample_with_log_prob: callable):
        """
        IMH kernel constructor.
        
        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event.
        :param callable neg_log_prob_target: function that computes the negative of the log target probability density. 
         Receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape 
         `batch_shape`.
        :param callable proposal_log_prob: function that computes the log proposal probability density. 
         Receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape 
         `batch_shape`.
        :param callable proposal_sample_with_log_prob: function that draws samples from the proposal distribution. 
         Receives as input a sample shape tuple `batch_shape` and returns a tensor with shape 
         `(*batch_shape, *event_shape)`, and the corresponding log probability density tensor with shape `batch_shape`.
        """
        super().__init__(event_shape, neg_log_prob_target)
        self.proposal_log_prob = proposal_log_prob
        self.proposal_sample_with_log_prob = proposal_sample_with_log_prob

    def step(self, x: torch.Tensor):
        # Propose new state
        batch_shape = x.shape[:-len(self.event_shape)]
        u_x = -self.proposal_log_prob(x)
        x_prime, log_prob_x_prime = self.proposal_sample_with_log_prob(batch_shape)
        u_x_prime = -log_prob_x_prime

        # Compute divergence mask
        divergence_mask = sum_except_batch(
            (~torch.isfinite(x_prime)).long(), self.event_shape
        ) > 0

        # Compute acceptance mask
        acceptance_mask = torch.zeros_like(divergence_mask)
        log_prob_accept = metropolis_acceptance_log_ratio(
            -self.neg_log_prob_target(x[~divergence_mask]),
            -self.neg_log_prob_target(x_prime[~divergence_mask]),
            -u_x[~divergence_mask],
            -u_x_prime[~divergence_mask],
        )
        log_u = torch.rand_like(log_prob_accept).log()
        acceptance_mask[~divergence_mask] = log_u < log_prob_accept

        return x_prime.detach(), acceptance_mask
