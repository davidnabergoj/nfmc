import math
from typing import Optional, Tuple, Union
import torch
from dataclasses import dataclass

from nfmc.algorithms.sampling.mcmc.base import MetropolisSampler, MetropolisParameters, MetropolisKernel
from nfmc.util import metropolis_acceptance_log_ratio, sum_except_batch


@dataclass
class LangevinKernel(MetropolisKernel):
    step_size: Optional[float] = None

    def __post_init__(self):
        # Set initial step size
        if self.step_size is None:
            # self.step_size = self.event_size ** (-1 / 3)
            # self.step_size = self.event_size ** (-1 / 3)
            self.step_size = 0.01
        super().__post_init__()

    @staticmethod
    def proposal_potential(x_prime: torch.Tensor,
                           x: torch.Tensor,
                           grad_u_x: torch.Tensor,
                           inv_mass_diag: torch.Tensor,
                           tau: float):
        """
        Compute the Langevin algorithm proposal potential q(x_prime | x).
        """
        imd = inv_mass_diag.view(1, -1)
        assert x_prime.shape == x.shape == grad_u_x.shape
        term = x_prime - (x - tau * imd * grad_u_x)
        return (term ** 2 / imd).sum(dim=-1) / (4 * tau)

    def potential_and_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = x.shape[:-len(self.event_shape)]
        infinite_elements = ~torch.isfinite(x)
        finite_mask = sum_except_batch(infinite_elements.long(), self.event_shape) == 0
        u_value = torch.full(size=batch_shape, fill_value=torch.nan).to(x)
        grad_u_value = torch.full(size=x.shape, fill_value=torch.nan).to(x)

        with torch.enable_grad():
            # Compute potential and gradient at current state
            x_finite = x[finite_mask]
            x_finite.requires_grad_(True)

            u_finite = self.target(x_finite).to(u_value)
            u_value[finite_mask] = u_finite
            grad_u_value[finite_mask] = torch.autograd.grad(u_finite.sum(), x_finite, create_graph=False)[0]

            x_finite = x_finite.detach()
            x_finite.grad = None  # Clear gradients
            u_value = u_value.detach()
            u_value.grad = None  # Clear gradients
            grad_u_value = grad_u_value.detach()
            grad_u_value.grad = None  # Clear gradients

        return u_value, grad_u_value

    def propose(self, x: torch.Tensor):
        noise = torch.randn_like(x)

        # Compute potential and gradient at current state
        u_x, grad_u_x = self.potential_and_grad(x)

        # Compute new state
        grad_term = -self.step_size * self.inv_mass_diag[None] * grad_u_x
        noise_term = torch.sqrt(2 * self.step_size * self.inv_mass_diag[None]) * noise
        x_prime = x + grad_term + noise_term

        divergence_mask_x = sum_except_batch((~torch.isfinite(x_prime)).long(), self.event_shape) > 0
        divergence_mask_u = (~torch.isfinite(u_x)).long() > 0
        divergence_mask_grad_u = sum_except_batch((~torch.isfinite(grad_u_x)).long(), self.event_shape) > 0
        divergence_mask = divergence_mask_x | divergence_mask_u | divergence_mask_grad_u

        return x_prime, divergence_mask, u_x, grad_u_x

    def __repr__(self):
        return (f'log step: {math.log(self.step_size):.2f}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')


@dataclass
class LangevinParameters(MetropolisParameters):
    pass


class Langevin(MetropolisSampler):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: Optional[LangevinKernel] = None,
                 params: Optional[LangevinParameters] = None):
        if kernel is None:
            kernel = LangevinKernel(event_shape, target)
        if params is None:
            params = LangevinParameters()
        super().__init__(event_shape, target, kernel, params)

    @property
    def name(self):
        return 'LMC'

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        n_chains = x.shape[0]
        x_prime, divergence_mask, u_x, grad_u_x = self.kernel.propose(x)
        acceptance_mask = torch.zeros_like(divergence_mask)

        if self.params.adjustment:
            # Compute potential and gradient at proposed state
            u_x_prime, grad_u_x_prime = self.kernel.potential_and_grad(x_prime[~divergence_mask])

            # Perform metropolis adjustment (MALA)
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
        else:
            acceptance_mask[~divergence_mask] = True

        n_divergences = int(divergence_mask.long().sum())
        n_calls = n_chains
        n_grads = n_chains
        if self.params.adjustment:
            n_calls += n_chains
            n_grads += n_chains

        return x_prime.detach(), acceptance_mask, n_calls, n_grads, n_divergences


class MALA(Langevin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = True


class ULA(Langevin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
