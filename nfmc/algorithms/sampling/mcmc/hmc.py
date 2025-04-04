import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch

from nfmc.algorithms.sampling.mcmc.base import MetropolisParameters, MetropolisKernel, MetropolisSampler
from torchflows.utils import sum_except_batch


@dataclass
class HMCKernel(MetropolisKernel):
    event_size: int
    n_leapfrog_steps: int = 20

    def __repr__(self):
        return (f'log step: {math.log(self.step_size):.2f}, '
                f'leapfrogs: {self.n_leapfrog_steps}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')


@dataclass
class HMCParameters(MetropolisParameters):
    pass


def mass_matrix_multiply(x: torch.Tensor, inverse_mass_matrix_diagonal: torch.Tensor, event_shape):
    # x.shape = (*batch_shape, *event_shape)
    # event_size = prod(event_shape)
    # inverse_mass_matrix_diagonal.shape = (event_size,)
    # output_shape = x.shape
    # output = x * inverse_mass_matrix_diagonal
    batch_shape = x.shape[:-len(event_shape)]
    event_size = int(torch.prod(torch.tensor(event_shape)))
    x_reshaped = x.view(*batch_shape, event_size)
    x_reshaped_multiplied = torch.einsum('...i,i->...i', x_reshaped, inverse_mass_matrix_diagonal.to(x_reshaped))
    x_multiplied = x_reshaped_multiplied.view_as(x)
    return x_multiplied


def grad_potential(x: torch.Tensor, potential: callable, event_shape):
    grad_value = torch.full(size=x.shape, fill_value=torch.nan).to(x)
    finite_mask = sum_except_batch((~torch.isfinite(x)).long(), event_shape) == 0

    if torch.any(torch.as_tensor(finite_mask, dtype=torch.bool)):
        with torch.enable_grad():
            x_finite = x[finite_mask]
            x_finite.requires_grad_(True)

            grad_value[finite_mask] = torch.autograd.grad(potential(x_finite).sum(), x_finite)[0]

            x_finite = x_finite.detach()
            x_finite.requires_grad_(False)

            grad_value = grad_value.detach()
            grad_value.requires_grad_(False)

    return grad_value


def hmc_step_b(x: torch.Tensor, momentum: torch.Tensor, step_size: float, potential: callable, event_shape):
    # momentum update
    return momentum - step_size / 2 * grad_potential(x, potential, event_shape)


def hmc_step_a(x: torch.Tensor, momentum: torch.Tensor, inv_mass_diag, step_size: float, event_shape):
    # position update
    return x + step_size * mass_matrix_multiply(momentum, inv_mass_diag.to(momentum), event_shape)


def hmc_trajectory(x: torch.Tensor,
                   momentum: torch.Tensor,
                   event_shape,
                   kernel: HMCKernel,
                   potential: callable,
                   full_output: bool = False):
    xs = []
    for j in range(kernel.n_leapfrog_steps):
        momentum = hmc_step_b(x, momentum, kernel.step_size, potential, event_shape)
        x = hmc_step_a(x, momentum, kernel.inv_mass_diag, kernel.step_size, event_shape)
        momentum = hmc_step_b(x, momentum, kernel.step_size, potential, event_shape)
        if full_output:
            xs.append(x)

    if full_output:
        return torch.stack(xs), x, momentum
    return x, momentum


class HMC(MetropolisSampler):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: Optional[HMCKernel] = None,
                 params: Optional[HMCParameters] = None):
        if kernel is None:
            kernel = HMCKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if params is None:
            params = HMCParameters()
        super().__init__(event_shape, target, kernel, params)

    @property
    def name(self):
        return "HMC"

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        n_chains = x.shape[0]
        p = mass_matrix_multiply(torch.randn_like(x), 1 / self.kernel.inv_mass_diag.sqrt().to(x), self.event_shape)
        x_prime, p_prime = hmc_trajectory(x, p, self.event_shape, self.kernel, potential=self.target)

        # Divergence occurs if an element of x_prime of p_prime is not finite
        divergence_mask_x = sum_except_batch((~torch.isfinite(x_prime)).long(), self.event_shape) > 0
        divergence_mask_p = sum_except_batch((~torch.isfinite(p_prime)).long(), self.event_shape) > 0
        divergence_mask = divergence_mask_x | divergence_mask_p

        acceptance_mask = torch.zeros_like(divergence_mask)

        if self.params.adjustment:
            hamiltonian_start: torch.Tensor = self.target(x[~divergence_mask]) + 0.5 * sum_except_batch(
                mass_matrix_multiply(p[~divergence_mask] ** 2, self.kernel.inv_mass_diag, self.event_shape),
                self.event_shape
            )
            hamiltonian_end: torch.Tensor = self.target(x_prime[~divergence_mask]) + 0.5 * sum_except_batch(
                mass_matrix_multiply(p_prime[~divergence_mask] ** 2, self.kernel.inv_mass_diag, self.event_shape),
                self.event_shape
            )
            log_prob_accept = -hamiltonian_end - (-hamiltonian_start)
            log_u = torch.rand_like(log_prob_accept).log()  # n_chains
            acceptance_mask[~divergence_mask] = (log_u < log_prob_accept)  # n_chains
        else:
            acceptance_mask[~divergence_mask] = True

        n_divergences = int(divergence_mask.long().sum())
        n_calls = 2 * self.kernel.n_leapfrog_steps * n_chains
        n_grads = 2 * self.kernel.n_leapfrog_steps * n_chains
        if self.params.adjustment:
            n_calls += 2 * n_chains
        return x_prime.detach(), acceptance_mask, n_calls, n_grads, n_divergences


class UHMC(HMC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
