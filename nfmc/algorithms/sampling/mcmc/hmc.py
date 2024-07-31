import math
from typing import Sized, Optional
from tqdm import tqdm
from nfmc.algorithms.sampling.base import Sampler, MCMCKernel, MCMCParameters, MCMCOutput, MCMCStatistics
from dataclasses import dataclass
import torch

from nfmc.algorithms.sampling.tuning import DualAveragingParams, DualAveraging
from normalizing_flows.utils import sum_except_batch


@dataclass
class HMCKernel(MCMCKernel):
    event_size: int
    inv_mass_diag: torch.Tensor = None
    step_size: float = 0.01
    n_leapfrog_steps: int = 20

    def __post_init__(self):
        if self.inv_mass_diag is None:
            self.inv_mass_diag = torch.ones(self.event_size)
        else:
            if self.inv_mass_diag.shape != (self.event_size,):
                raise ValueError

    def __repr__(self):
        return (f'log step: {math.log(self.step_size)}, '
                f'leapfrogs: {self.n_leapfrog_steps}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')


@dataclass
class HMCParameters(MCMCParameters):
    tune_inv_mass_diag: bool = False
    tune_step_size: bool = False
    adjustment: bool = True
    imd_adjustment: float = 1e-3
    da_params = DualAveragingParams()


def mass_matrix_multiply(x: torch.Tensor, inverse_mass_matrix_diagonal: torch.Tensor, event_shape):
    # x.shape = (*batch_shape, *event_shape)
    # event_size = prod(event_shape)
    # inverse_mass_matrix_diagonal.shape = (event_size,)
    # output_shape = x.shape
    # output = x * inverse_mass_matrix_diagonal
    batch_shape = x.shape[:-len(event_shape)]
    event_size = int(torch.prod(torch.tensor(event_shape)))
    x_reshaped = x.view(*batch_shape, event_size)
    x_reshaped_multiplied = torch.einsum('...i,i->...i', x_reshaped, inverse_mass_matrix_diagonal)
    x_multiplied = x_reshaped_multiplied.view_as(x)
    return x_multiplied


def grad_potential(x: torch.Tensor, potential: callable):
    with torch.enable_grad():
        x.requires_grad_(True)
        grad = torch.autograd.grad(potential(x).sum(), x)[0]
        x = x.detach()
        x.requires_grad_(False)
        grad = grad.detach()
        grad.requires_grad_(False)
    return grad


def hmc_step_b(x: torch.Tensor, momentum: torch.Tensor, step_size: float, potential: callable):
    # momentum update
    return momentum - step_size / 2 * grad_potential(x, potential)


def hmc_step_a(x: torch.Tensor, momentum: torch.Tensor, inv_mass_diag, step_size: float, event_shape):
    # position update
    return x + step_size * mass_matrix_multiply(momentum, inv_mass_diag, event_shape)


def hmc_trajectory(x: torch.Tensor,
                   momentum: torch.Tensor,
                   event_shape,
                   kernel: HMCKernel,
                   potential: callable,
                   full_output: bool = False):
    xs = []
    for j in range(kernel.n_leapfrog_steps):
        momentum = hmc_step_b(x, momentum, kernel.step_size, potential)
        x = hmc_step_a(x, momentum, kernel.inv_mass_diag, kernel.step_size, event_shape)
        momentum = hmc_step_b(x, momentum, kernel.step_size, potential)
        if full_output:
            xs.append(x)

    if full_output:
        return torch.stack(xs), x, momentum
    return x, momentum


class HMC(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: Optional[HMCKernel] = None,
                 params: Optional[HMCParameters] = None):
        if kernel is None:
            kernel = HMCKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if params is None:
            params = HMCParameters()
        super().__init__(event_shape, target, kernel, params)

    def sample(self, x0: torch.Tensor, show_progress: bool = False) -> MCMCOutput:
        self.kernel: HMCKernel
        self.params: HMCParameters

        # Initialize
        n_chains, *event_shape = x0.shape
        xs = torch.zeros(size=(self.params.n_iterations, n_chains, *event_shape), dtype=x0.dtype, device=x0.device)
        da = DualAveraging(initial_step_size=self.kernel.step_size, params=self.params.da_params)
        x = torch.clone(x0).detach()
        statistics = MCMCStatistics(n_accepted_trajectories=0, n_divergences=0)

        for i in (pbar := tqdm(range(self.params.n_iterations), desc='HMC', disable=not show_progress)):
            p = mass_matrix_multiply(torch.randn_like(x), 1 / self.kernel.inv_mass_diag.sqrt(), event_shape)
            try:
                x_prime, p_prime = hmc_trajectory(x, p, event_shape, self.kernel, potential=self.target)
                with torch.no_grad():
                    if self.params.adjustment:
                        hamiltonian_start = self.target(x) + 0.5 * sum_except_batch(
                            mass_matrix_multiply(p ** 2, self.kernel.inv_mass_diag, event_shape),
                            event_shape
                        )
                        hamiltonian_end = self.target(x_prime) + 0.5 * sum_except_batch(
                            mass_matrix_multiply(p_prime ** 2, self.kernel.inv_mass_diag, event_shape),
                            event_shape
                        )
                        log_prob_accept = -hamiltonian_end - (-hamiltonian_start)
                        log_u = torch.rand_like(log_prob_accept).log()  # n_chains
                        accepted_mask = (log_u < log_prob_accept)  # n_chains
                    else:
                        accepted_mask = torch.ones(size=(n_chains,), dtype=torch.bool)
                    x[accepted_mask] = x_prime[accepted_mask]
            except ValueError:
                accepted_mask = torch.zeros(size=(n_chains,), dtype=torch.bool)
                statistics.n_divergences += 1

            with torch.no_grad():
                statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
                statistics.n_attempted_trajectories += n_chains
                x = x.detach()
                xs[i] = x

                if n_chains > 1 and self.params.tune_inv_mass_diag:
                    # inv_mass_diag = torch.var(x.flatten(1, -1), dim=0)  # Mass matrix adaptation
                    self.kernel.inv_mass_diag = (
                            self.params.imd_adjustment * torch.var(x.flatten(1, -1), dim=0) +
                            (1 - self.params.imd_adjustment) * self.kernel.inv_mass_diag
                    )
                if self.params.tune_step_size and self.params.adjustment:
                    # Step size tuning is only possible with adjustment right now
                    acc_rate = torch.mean(accepted_mask.float())
                    error = self.params.da_params.target_acceptance_rate - acc_rate
                    da.step(error)
                    self.kernel.step_size = da.value  # Step size adaptation
                pbar.set_postfix_str(f'{statistics} | {self.kernel} | {da}')

        return MCMCOutput(samples=xs, statistics=statistics)


class UHMC(HMC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
