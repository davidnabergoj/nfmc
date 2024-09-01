import math
import time
from copy import deepcopy
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
        return (f'log step: {math.log(self.step_size):.2f}, '
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

    def warmup(self, x0: torch.Tensor, show_progress: bool = True, thinning: int = 1,
               time_limit_seconds: int = 3600 * 24) -> MCMCOutput:
        self.kernel: MHKernel
        self.params: MHParameters

        warmup_copy = deepcopy(self)
        warmup_copy.params.tune_inv_mass_diag = True
        warmup_copy.params.tune_step_size = True
        warmup_copy.params.n_iterations = self.params.n_warmup_iterations
        warmup_output = warmup_copy.sample(x0, show_progress=show_progress, time_limit_seconds=time_limit_seconds)

        self.kernel = warmup_copy.kernel
        new_params = warmup_copy.params
        new_params.n_iterations = self.params.n_iterations
        new_params.tune_step_size = self.params.tune_step_size
        new_params.tune_inv_mass_diag = self.params.tune_inv_mass_diag
        self.params = new_params

        return warmup_output

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               thinning: int = 1,
               time_limit_seconds: int = 3600 * 24) -> MCMCOutput:
        self.params: MHParameters
        self.kernel: MHKernel

        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=self.params.store_samples)
        out.running_samples.thinning = thinning

        # Initialize
        t0 = time.time()
        n_chains = x0.shape[0]
        da = DualAveraging(initial_step_size=self.kernel.step_size, params=self.params.da_params)
        x = torch.clone(x0).detach()
        out.statistics.elapsed_time_seconds += time.time() - t0

        for i in (pbar := tqdm(range(self.params.n_iterations), desc='MH', disable=not show_progress)):
            if out.statistics.elapsed_time_seconds >= time_limit_seconds:
                break
            t0 = time.time()
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
                out.statistics.n_divergences += 1

            if self.params.adjustment:
                out.statistics.n_target_calls += 2 * n_chains

            out.statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            out.statistics.n_attempted_trajectories += n_chains

            with torch.no_grad():
                x = x.detach()
                out.running_samples.add(x)

                # Update the inverse mass diagonal
                if n_chains > 1 and self.params.tune_inv_mass_diag:
                    # self.kernel.inv_mass_diag = torch.std(x, dim=0)  # root of the preconditioning matrix diagonal
                    self.kernel.inv_mass_diag = (
                            self.params.imd_adjustment * torch.var(x.flatten(1, -1), dim=0) +
                            (1 - self.params.imd_adjustment) * self.kernel.inv_mass_diag
                    )
                # if self.params.tune_step_size and self.params.adjustment:
                #     # Step size tuning is only possible with adjustment right now
                #     acc_rate = torch.mean(accepted_mask.float())
                #     error = self.params.da_params.target_acceptance_rate - acc_rate
                #     da.step(error)
                #     self.kernel.step_size = da.value  # Step size adaptation

            out.statistics.elapsed_time_seconds += time.time() - t0
            pbar.set_postfix_str(f'{out.statistics} | {self.kernel} | {da}')

        out.kernel = self.kernel
        return out


class RandomWalk(MH):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
