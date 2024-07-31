import math
from dataclasses import dataclass
from typing import Sized, Optional
import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCKernel, MCMCParameters, MCMCOutput, MCMCStatistics
from nfmc.algorithms.sampling.tuning import DualAveragingParams, DualAveraging
from nfmc.util import metropolis_acceptance_log_ratio
from tqdm import tqdm


@dataclass
class MHKernel(MCMCKernel):
    event_size: int
    step_size: Optional[float] = 0.01
    inv_mass_diag: torch.Tensor = None

    def __post_init__(self):
        # Set initial mass matrix
        if self.inv_mass_diag is None:
            self.inv_mass_diag = torch.ones(self.event_size)
        else:
            if self.inv_mass_diag.shape != (self.event_size,):
                raise ValueError

    def __repr__(self):
        return (f'log step: {math.log(self.step_size)}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')


@dataclass
class MHParameters(MCMCParameters):
    tune_inv_mass_diag: bool = False
    tune_step_size: bool = False
    adjustment: bool = True
    imd_adjustment: float = 1e-3
    da_params = DualAveragingParams()


class MH(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: Optional[MHKernel] = None,
                 params: Optional[MHParameters] = None):
        if kernel is None:
            kernel = MHKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if params is None:
            params = MHParameters()
        super().__init__(event_shape, target, kernel, params)

    def sample(self, x0: torch.Tensor, show_progress: bool = False) -> MCMCOutput:
        self.params: MHParameters
        self.kernel: MHKernel

        # Initialize
        n_chains, *event_shape = x0.shape
        xs = torch.zeros(size=(self.params.n_iterations, n_chains, *event_shape), dtype=x0.dtype, device=x0.device)
        da = DualAveraging(initial_step_size=self.kernel.step_size, params=self.params.da_params)
        statistics = MCMCStatistics(n_accepted_trajectories=0, n_divergences=0)
        x = torch.clone(x0).detach()

        for i in (pbar := tqdm(range(self.params.n_iterations), desc='LMC', disable=not show_progress)):
            try:
                noise = torch.randn_like(x) * self.kernel.inv_mass_diag
                x_prime = x + noise

                if self.params.adjustment:
                    log_ratio = metropolis_acceptance_log_ratio(-self.target(x), -self.target(x_prime), 0, 0)
                    accepted_mask = torch.as_tensor(torch.log(torch.rand(n_chains)) < log_ratio)
                else:
                    accepted_mask = torch.ones(n_chains, dtype=torch.bool)
                x[accepted_mask] = x_prime[accepted_mask]
            except ValueError:
                accepted_mask = torch.zeros(n_chains, dtype=torch.bool)
                statistics.n_divergences += 1

            with torch.no_grad():
                statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
                statistics.n_attempted_trajectories += n_chains
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
                pbar.set_postfix_str(f'{statistics} | {self.kernel} | {da}')

        return MCMCOutput(samples=xs, statistics=statistics)


class RandomWalk(MH):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
