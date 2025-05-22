from typing import Tuple, Union
import torch
from nfmc.algorithms.mh.local import LocalMHKernel
from nfmc.util import compute_divergence_mask, grad_f, metropolis_acceptance_log_ratio, sum_except_batch, diag_mult


def propose_state(x: torch.Tensor,
                  event_shape: Union[Tuple[int, ...], torch.Size],
                  step_size: float,
                  inv_mass_diag: torch.Tensor,
                  neg_log_prob_target: callable):
    noise = torch.randn_like(x).to(x)

    # Compute potential and gradient at current state
    u_x, grad_u_x, nc, ng = grad_f(x, neg_log_prob_target, event_shape)

    grad_term = -step_size * \
        diag_mult(grad_u_x, inv_mass_diag.to(x), event_shape)
    noise_term = diag_mult(noise, torch.sqrt(
        2 * step_size * inv_mass_diag.to(x)), event_shape)
    x_prime = x + grad_term + noise_term
    return x_prime, u_x, grad_u_x, nc, ng


def proposal_neg_log_prob(x_prime: torch.Tensor,
                          event_shape: Union[Tuple[int, ...], torch.Size],
                          x: torch.Tensor,
                          grad_u_x: torch.Tensor,
                          inv_mass_diag: torch.Tensor,
                          tau: float):
    """
    Compute the negative log probability density of the MALA proposal q(x_prime | x).
    """
    term = x_prime - (x - tau * diag_mult(grad_u_x,
                      inv_mass_diag, event_shape))
    return sum_except_batch(diag_mult(term ** 2, inv_mass_diag, event_shape), event_shape) / (4 * tau)


class MALAKernel(LocalMHKernel):
    """
    MALA kernel with a diagonal mass matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        HMCKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable neg_log_prob_target: negative log probability density function. Takes as input a tensor with 
        :param kwargs: keyword arguments for the LocalMHKernel constructor.
        """
        if 'target_acceptance_rate' not in kwargs:
            kwargs['target_acceptance_rate'] = 0.57
        super().__init__(event_shape, neg_log_prob_target, **kwargs)

    @property
    def name(self):
        return 'MALA'

    def step(self,
             x: torch.Tensor,
             update: bool = False,
             **kwargs) -> torch.Tensor:
        """
        Perform one MALA transition.

        :param torch.Tensor x: incoming state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :param kwargs: keyword arguments for kernel updates.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """

        # Propose new state
        x_prime, u_x, grad_u_x, nc, ng = propose_state(
            x,
            self.event_shape,
            self.step_size,
            self.inv_mass_diag,
            self.neg_log_prob_target
        )
        self.increment_n_calls(nc)
        self.increment_n_grads(ng)

        # Compute divergence mask
        divergence_mask_x = compute_divergence_mask(x_prime, self.event_shape)
        divergence_mask_u = (~torch.isfinite(u_x)).long() > 0
        divergence_mask_grad_u = compute_divergence_mask(
            grad_u_x, self.event_shape)
        divergence_mask = divergence_mask_x | divergence_mask_u | divergence_mask_grad_u
        self.increment_n_divergences(int(divergence_mask.long().sum()))

        # Compute acceptance mask
        u_x_prime, grad_u_x_prime, nc, ng = grad_f(
            x_prime[~divergence_mask],
            self.neg_log_prob_target,
            self.event_shape
        )
        self.increment_n_calls(nc)
        self.increment_n_grads(ng)

        acceptance_mask = torch.zeros_like(divergence_mask)
        log_prob_accept = metropolis_acceptance_log_ratio(
            log_prob_target_curr=-u_x[~divergence_mask],
            log_prob_target_prime=-u_x_prime,
            log_prob_proposal_curr=-proposal_neg_log_prob(
                x[~divergence_mask],
                self.event_shape,
                x_prime[~divergence_mask],
                grad_u_x_prime,
                1 / self.inv_mass_diag,
                self.step_size
            ),
            log_prob_proposal_prime=-proposal_neg_log_prob(
                x_prime[~divergence_mask],
                self.event_shape,
                x[~divergence_mask],
                grad_u_x[~divergence_mask],
                1 / self.inv_mass_diag,
                self.step_size
            )
        )
        log_u = torch.rand_like(log_prob_accept).log()
        acceptance_mask[~divergence_mask] = log_u < log_prob_accept
        x[acceptance_mask] = x_prime[acceptance_mask]
        x = x.detach()

        if update:
            self._update(x, acceptance_mask, **kwargs)

        self.increment_n_steps()
        self.increment_n_attempted_transitions(n_chains=x.shape[0])
        self.increment_n_accepted_transitions(
            int(acceptance_mask.long().sum()))

        return x
