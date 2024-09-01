import math
import time
from copy import deepcopy
from typing import Sized, Optional, Tuple, Union, Dict, Any

import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCOutput, MCMCKernel, MCMCParameters, MCMCStatistics
from dataclasses import dataclass
from tqdm import tqdm

from nfmc.algorithms.sampling.mcmc.base import MCMCSampler, MetropolisSampler, MetropolisParameters, MetropolisKernel
from nfmc.algorithms.sampling.tuning import DualAveragingParams, DualAveraging
from nfmc.util import metropolis_acceptance_log_ratio


@dataclass
class LangevinKernel(MetropolisKernel):
    event_size: int
    step_size: Optional[float] = None

    def __post_init__(self):
        # Set initial step size
        if self.step_size is None:
            self.step_size = self.event_size ** (-1 / 3)
        super().__post_init__()

    def __repr__(self):
        return (f'log step: {math.log(self.step_size):.2f}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')


@dataclass
class LangevinParameters(MetropolisParameters):
    pass


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


class Langevin(MetropolisSampler):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: Optional[LangevinKernel] = None,
                 params: Optional[LangevinParameters] = None):
        if kernel is None:
            kernel = LangevinKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if params is None:
            params = LangevinParameters()
        super().__init__(event_shape, target, kernel, params)

    @property
    def name(self):
        return 'LMC'

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        n_chains = x.shape[0]
        noise = torch.randn_like(x)
        try:
            # Compute potential and gradient at current state
            x.requires_grad_(True)
            u_x = self.target(x)
            grad_u_x, = torch.autograd.grad(u_x.sum(), x)
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
                grad_u_x_prime, = torch.autograd.grad(u_x_prime.sum(), x_prime)
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
                mask = torch.as_tensor(torch.log(torch.rand(n_chains)) < log_ratio)
            else:
                # No adjustment (ULA)
                mask = torch.ones(n_chains, dtype=torch.bool)
            n_divergences = 0
        except ValueError:
            x_prime = x
            mask = torch.zeros(n_chains, dtype=torch.bool)
            n_divergences = 1

        n_calls = n_chains
        n_grads = n_chains
        if self.params.adjustment:
            n_calls += n_chains
            n_grads += n_chains

        return x_prime.detach(), mask, n_calls, n_grads, n_divergences


class MALA(Langevin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = True


class ULA(Langevin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
