from typing import Optional, Tuple, Union
import torch
from nfmc.algorithms.mh.base import MHKernel
from nfmc.util import grad_f, metropolis_acceptance_log_ratio, sum_except_batch


def propose_state(
        x: torch.Tensor,
        event_shape: Union[Tuple[int, ...], torch.Size],
        step_size: float,
        inv_mass_diag: torch.Tensor,
        neg_log_prob_target: callable
):
    noise = torch.randn_like(x)

    # Compute potential and gradient at current state
    u_x, grad_u_x, nc, ng = grad_f(x, neg_log_prob_target, event_shape)

    grad_term = -step_size * inv_mass_diag[None] * grad_u_x
    noise_term = torch.sqrt(2 * step_size * inv_mass_diag[None]) * noise
    x_prime = x + grad_term + noise_term
    return x_prime, u_x, grad_u_x, nc, ng


def proposal_neg_log_prob(x_prime: torch.Tensor,
                          x: torch.Tensor,
                          grad_u_x: torch.Tensor,
                          inv_mass_diag: torch.Tensor,
                          tau: float):
    """
    Compute the negative log probability density of the MALA proposal q(x_prime | x).
    """
    imd = inv_mass_diag.view(1, -1)
    assert x_prime.shape == x.shape == grad_u_x.shape
    term = x_prime - (x - tau * imd * grad_u_x)
    return (term ** 2 / imd).sum(dim=-1) / (4 * tau)


class MALAKernel(MHKernel):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 step_size: float = 0.01,
                 inv_mass_diag: Optional[torch.Tensor] = None):
        super().__init__(event_shape, neg_log_prob_target)
        step_size = step_size
        self.inv_mass_diag = inv_mass_diag
        if self.inv_mass_diag is None:
            self.inv_mass_diag = torch.ones(
                size=(self.event_size,),
                dtype=torch.double
            )

    def step(self, x: torch.Tensor):
        # Propose new state
        x_prime, u_x, grad_u_x, nc, ng = propose_state(
            x,
            self.step_size,
            self.inv_mass_diag,
            self.neg_log_prob_target
        )
        self._n_calls += nc
        self._n_grads += ng

        # Compute divergence mask
        divergence_mask_x = sum_except_batch(
            (~torch.isfinite(x_prime)).long(), self.event_shape
        ) > 0
        divergence_mask_u = (~torch.isfinite(u_x)).long() > 0
        divergence_mask_grad_u = sum_except_batch(
            (~torch.isfinite(grad_u_x)).long(), self.event_shape
        ) > 0
        divergence_mask = divergence_mask_x | divergence_mask_u | divergence_mask_grad_u

        # Compute acceptance mask
        u_x_prime, grad_u_x_prime, nc, ng = grad_f(
            x_prime[~divergence_mask], self.neg_log_prob_target
        )
        self._n_calls += nc
        self._n_grads += ng

        acceptance_mask = torch.zeros_like(divergence_mask)
        log_prob_accept = metropolis_acceptance_log_ratio(
            log_prob_target_curr=-u_x[~divergence_mask],
            log_prob_target_prime=-u_x_prime,
            log_prob_proposal_curr=-self.kernel.proposal_potential(
                x[~divergence_mask],
                x_prime[~divergence_mask],
                grad_u_x_prime,
                1 / self.kernel.inv_mass_diag,
                self.kernel.step_size
            ),
            log_prob_proposal_prime=-self.kernel.proposal_potential(
                x_prime[~divergence_mask],
                x[~divergence_mask],
                grad_u_x[~divergence_mask],
                1 / self.kernel.inv_mass_diag,
                self.kernel.step_size
            )
        )
        log_u = torch.rand_like(log_prob_accept).log()
        acceptance_mask[~divergence_mask] = log_u < log_prob_accept

        return x_prime.detach(), acceptance_mask
