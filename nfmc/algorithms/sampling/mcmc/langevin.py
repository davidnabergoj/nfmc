import math
import time
from copy import deepcopy
from typing import Sized, Optional

import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCOutput, MCMCKernel, MCMCParameters, MCMCStatistics
from dataclasses import dataclass
from tqdm import tqdm
from nfmc.algorithms.sampling.tuning import DualAveragingParams, DualAveraging
from nfmc.util import metropolis_acceptance_log_ratio


@dataclass
class LangevinKernel(MCMCKernel):
    event_size: int
    step_size: Optional[float] = None
    inv_mass_diag: torch.Tensor = None

    def __post_init__(self):
        # Set initial step size
        if self.step_size is None:
            self.step_size = self.event_size ** (-1 / 3)

        # Set initial mass matrix
        if self.inv_mass_diag is None:
            self.inv_mass_diag = torch.ones(self.event_size)
        else:
            if self.inv_mass_diag.shape != (self.event_size,):
                raise ValueError

    def __repr__(self):
        return (f'log step: {math.log(self.step_size):.2f}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')


@dataclass
class LangevinParameters(MCMCParameters):
    tune_inv_mass_diag: bool = False
    tune_step_size: bool = False
    adjustment: bool = True
    imd_adjustment: float = 1e-3
    da_params = DualAveragingParams()


@torch.no_grad()
def proposal_potential(x_prime: torch.Tensor,
                       x: torch.Tensor,
                       grad_u_x: torch.Tensor,
                       a_diag: torch.Tensor,
                       tau: float):
    """
    Compute the Langevin algorithm proposal potential q(x_prime | x).
    """
    assert x_prime.shape == x.shape == grad_u_x.shape
    term = x_prime - x + tau * a_diag.view(1, -1) * grad_u_x
    return (term * (1 / a_diag.view(1, -1)) * term).sum(dim=-1) / (4 * tau)


class Langevin(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: Optional[LangevinKernel] = None,
                 params: Optional[LangevinParameters] = None):
        if kernel is None:
            kernel = LangevinKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if params is None:
            params = LangevinParameters()
        super().__init__(event_shape, target, kernel, params)

    def warmup(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: LangevinKernel
        self.params: LangevinParameters

        warmup_copy = deepcopy(self)
        warmup_copy.params.tune_inv_mass_diag = True
        warmup_copy.params.tune_step_size = True
        warmup_copy.params.n_iterations = self.params.n_warmup_iterations
        warmup_output = warmup_copy.sample(x0, show_progress=show_progress)

        self.kernel = warmup_copy.kernel
        new_params = warmup_copy.params
        new_params.n_iterations = self.params.n_iterations
        new_params.tune_step_size = self.params.tune_step_size
        new_params.tune_inv_mass_diag = self.params.tune_inv_mass_diag
        self.params = new_params

        return warmup_output

    def sample(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.params: LangevinParameters
        self.kernel: LangevinKernel

        # Initialize
        statistics = MCMCStatistics(n_accepted_trajectories=0, n_divergences=0)

        t0 = time.time()
        n_chains, *event_shape = x0.shape
        xs = torch.zeros(size=(self.params.n_iterations, n_chains, *event_shape), dtype=x0.dtype, device=x0.device)
        da = DualAveraging(initial_step_size=self.kernel.step_size, params=self.params.da_params)
        x = torch.clone(x0).detach()
        statistics.elapsed_time_seconds += time.time() - t0

        for i in (pbar := tqdm(range(self.params.n_iterations), desc='LMC', disable=not show_progress)):
            t0 = time.time()
            noise = torch.randn_like(x)
            try:
                # Compute potential and gradient at current state
                x.requires_grad_(True)
                u_x = self.target(x)
                statistics.n_target_calls += n_chains
                grad_u_x, = torch.autograd.grad(u_x.sum(), x)
                statistics.n_target_gradient_calls += n_chains
                x.requires_grad_(False)  # Clear gradients
                x = x.detach()
                x.grad = None  # Clear gradients

                # Compute new state
                grad_term = -self.kernel.step_size / self.kernel.inv_mass_diag[None].square() * grad_u_x
                noise_term = math.sqrt(2 * self.kernel.step_size) / self.kernel.inv_mass_diag[None] * noise
                x_prime = x + grad_term + noise_term

                if self.params.adjustment:
                    # Compute potential and gradient at proposed state
                    x_prime.requires_grad_(True)
                    u_x_prime = self.target(x_prime)
                    statistics.n_target_calls += n_chains
                    grad_u_x_prime, = torch.autograd.grad(u_x_prime.sum(), x_prime)
                    statistics.n_target_gradient_calls += n_chains
                    x_prime.requires_grad_(False)  # Clear gradients
                    x_prime = x_prime.detach()
                    x_prime.grad = None  # Clear gradients

                    # Perform metropolis adjustment (MALA)
                    log_ratio = metropolis_acceptance_log_ratio(
                        log_prob_curr=-u_x,
                        log_prob_prime=-u_x_prime,
                        log_proposal_curr=-proposal_potential(
                            x,
                            x_prime,
                            grad_u_x_prime,
                            1 / self.kernel.inv_mass_diag ** 2,
                            self.kernel.step_size
                        ),
                        log_proposal_prime=-proposal_potential(
                            x_prime,
                            x,
                            grad_u_x,
                            1 / self.kernel.inv_mass_diag ** 2,
                            self.kernel.step_size
                        )
                    )
                    accepted_mask = torch.as_tensor(torch.log(torch.rand(n_chains)) < log_ratio)
                else:
                    # No adjustment (ULA)
                    accepted_mask = torch.ones(n_chains, dtype=torch.bool)
                x[accepted_mask] = x_prime[accepted_mask]
            except ValueError:
                accepted_mask = torch.zeros(n_chains, dtype=torch.bool)
                statistics.n_divergences += 1

            statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            statistics.n_attempted_trajectories += n_chains

            with torch.no_grad():
                x = x.detach()
                xs[i] = x

                # Update the inverse mass diagonal
                if n_chains > 1 and self.params.tune_inv_mass_diag:
                    # self.kernel.inv_mass_diag = torch.std(x, dim=0)  # root of the preconditioning matrix diagonal
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

            statistics.elapsed_time_seconds += time.time() - t0
            pbar.set_postfix_str(f'{statistics} | {self.kernel} | {da}')

        return MCMCOutput(samples=xs, statistics=statistics)


class MALA(Langevin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = True


class ULA(Langevin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
