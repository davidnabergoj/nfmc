from typing import Tuple, Union
import math

import torch
from nfmc.algorithms.mh.local.base import LocalMHKernel
from nfmc.util import compute_divergence_mask, grad_f, metropolis_acceptance_log_ratio, sum_except_batch


def propose_state(x: torch.Tensor,
                  event_shape: Union[Tuple[int, ...], torch.Size],
                  step_size: torch.Tensor,
                  neg_log_prob_target: callable):
    """

    :param torch.Tensor x: event tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor step_size: step size tensor with shape `(n_chains,)` or `()`.
    """
    if len(step_size.shape) == 1:
        step_size = step_size.view(
            step_size.shape[0], *[1] * (len(x.shape) - 1))

    x = x.detach()
    noise = torch.randn_like(x).to(x)

    # Compute potential and gradient at current state
    u_x, grad_u_x, nc, ng = grad_f(x, neg_log_prob_target, event_shape)

    grad_term = -0.5 * step_size * grad_u_x
    noise_term = noise * torch.sqrt(step_size)
    x_prime = x + grad_term + noise_term
    return x_prime, u_x, grad_u_x, nc, ng


def proposal_neg_log_prob(x_prime: torch.Tensor,
                          event_shape: Union[Tuple[int, ...], torch.Size],
                          x: torch.Tensor,
                          grad_u_x: torch.Tensor,
                          tau: torch.Tensor):
    """
    Compute the negative log probability density of the MALA proposal q(x_prime | x).

    :param torch.Tensor x: event tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor x_prime: proposed event tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor tau: step size tensor with shape `(n_chains,)` or `()`.
    """
    if len(tau.shape) == 1:
        tau = tau.view(tau.shape[0], *[1] * (len(x.shape) - 1))

    term = x_prime - (x - tau * grad_u_x)
    return sum_except_batch(term ** 2 / (4 * tau), event_shape)


class MALAKernel(LocalMHKernel):
    """
    MALA kernel with a diagonal mass matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 target_acceptance_rate: float = 0.571,
                 **kwargs):
        """
        HMCKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable neg_log_prob_target: negative log probability density function. Takes as input a tensor with 
        :param kwargs: keyword arguments for the LocalMHKernel constructor.
        """
        super().__init__(
            event_shape,
            neg_log_prob_target,
            target_acceptance_rate=target_acceptance_rate,
            **kwargs
        )

    @property
    def name(self):
        return 'MALA'

    def step(self,
             x: torch.Tensor,
             update: bool = False) -> torch.Tensor:
        """
        Perform one MALA transition.

        :param torch.Tensor x: incoming state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        if not torch.isfinite(x).all():
            raise ValueError("Input state contains NaN or Inf values.")
        
        if self.warmup_active:
            step_size = self._dual_averaging.value
        else:
            step_size = self.step_size

        # Propose new state
        x_prime, u_x, grad_u_x, nc, ng = propose_state(
            x,
            self.event_shape,
            step_size,
            self.neg_log_prob_target
        )
        self.increment_n_calls(nc)
        self.increment_n_grads(ng)

        # Compute divergence mask
        divergence_mask_x = compute_divergence_mask(x_prime, self.event_shape)
        divergence_mask_u = ~torch.isfinite(u_x)
        divergence_mask_grad_u = compute_divergence_mask(
            grad_u_x, self.event_shape)
        divergence_mask = divergence_mask_x | divergence_mask_u | divergence_mask_grad_u
        self.increment_n_divergences(int(divergence_mask.long().sum()))
        self.increment_n_divergences_per_chain(divergence_mask)

        # Compute acceptance mask
        u_x_prime, grad_u_x_prime, nc, ng = grad_f(
            x_prime[~divergence_mask],
            self.neg_log_prob_target,
            self.event_shape
        )
        self.increment_n_calls(nc)
        self.increment_n_grads(ng)

        acceptance_mask = torch.zeros_like(divergence_mask)
        if (~divergence_mask).any():
            log_prob_accept = metropolis_acceptance_log_ratio(
                log_prob_target_curr=-u_x[~divergence_mask],
                log_prob_target_prime=-u_x_prime,
                log_prob_proposal_curr=-proposal_neg_log_prob(
                    x[~divergence_mask],
                    self.event_shape,
                    x_prime[~divergence_mask],
                    grad_u_x_prime,
                    step_size
                ),
                log_prob_proposal_prime=-proposal_neg_log_prob(
                    x_prime[~divergence_mask],
                    self.event_shape,
                    x[~divergence_mask],
                    grad_u_x[~divergence_mask],
                    step_size
                )
            )
            log_u = torch.rand_like(log_prob_accept.clamp(min=1e-10)).log()
            acceptance_mask[~divergence_mask] = log_u < log_prob_accept

        x_new = x.clone()
        x_new[acceptance_mask] = x_prime[acceptance_mask]
        x_new = x_new.detach()
        x = x_new.detach()

        if update:
            self._update(acceptance_mask)

        self.increment_n_steps()
        self.increment_n_attempted_transitions(n_chains=x.shape[0])
        self.increment_n_accepted_transitions(
            int(acceptance_mask.long().sum()))

        return x
