import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

import torch
from tqdm import tqdm

from nfmc.algorithms.sampling.base import Sampler, MCMCKernel, MCMCParameters, MCMCOutput
from nfmc.algorithms.sampling.tuning import DualAveraging, DualAveragingParams


class MCMCSampler(Sampler):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: MCMCKernel,
                 params: MCMCParameters):
        super().__init__(event_shape, target, kernel, params)

    @property
    def name(self):
        return "Generic MCMC"

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """

        :param torch.Tensor x: current state with shape `(n_chains, *event_shape)`.
        :return: proposed state with shape `(n_chains, *event_shape)`, acceptance mask with shape `(n_chains)`, number
         of evaluated target calls and target gradients, number of divergences.
        """
        raise NotImplementedError

    def update_kernel(self, data: Dict[str, Any]):
        raise NotImplementedError

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               thinning: int = 1,
               time_limit_seconds: int = 3600 * 24) -> MCMCOutput:
        warmup_copy = deepcopy(self)
        warmup_copy.params.tuning = True
        warmup_copy.params.n_iterations = self.params.n_warmup_iterations
        warmup_output = warmup_copy.sample(
            x0,
            show_progress=show_progress,
            thinning=thinning,
            time_limit_seconds=time_limit_seconds
        )

        self.kernel = warmup_copy.kernel
        new_params = warmup_copy.params
        new_params.n_iterations = self.params.n_iterations
        new_params.tuning = self.params.tuning
        self.params = new_params

        return warmup_output

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               thinning: int = 1,
               time_limit_seconds: int = 3600 * 24,
               **kwargs) -> MCMCOutput:
        n_chains, *event_shape = x0.shape
        event_shape = tuple(event_shape)
        out = MCMCOutput(event_shape, store_samples=self.params.store_samples)
        out.running_samples.thinning = thinning
        x = torch.clone(x0).detach()

        bar_label = f'{self.name}'
        if self.params.tuning:
            bar_label = f'{self.name} (tuning)'
        for _ in (pbar := tqdm(range(self.params.n_iterations), desc=bar_label, disable=not show_progress)):
            if out.statistics.elapsed_time_seconds > time_limit_seconds:
                break

            t0 = time.time()
            x_prime, mask, n_calls, n_grads, n_divs = self.propose(x)
            x = x.detach()
            x_prime = x_prime.detach()
            x[mask] = x_prime[mask]

            out.statistics.n_target_calls += n_calls
            out.statistics.n_target_gradient_calls += n_grads
            out.statistics.n_divergences += n_divs
            out.statistics.n_accepted_trajectories += int(torch.sum(mask))
            out.statistics.n_attempted_trajectories += n_chains
            out.statistics.expectations.update(x)

            with torch.no_grad():
                x = x.detach()
                out.running_samples.add(x)

                if self.params.tuning:
                    self.update_kernel({
                        'x': x,
                        'mask': mask
                    })

            out.statistics.elapsed_time_seconds += time.time() - t0
            pbar.set_postfix_str(f'{out.statistics} | {self.kernel}')

        out.kernel = self.kernel
        return out


@dataclass
class MetropolisKernel(MCMCKernel):
    event_size: int
    inv_mass_diag: torch.Tensor = None
    step_size: float = 0.01
    da: DualAveraging = None
    da_params: DualAveragingParams = DualAveragingParams()

    def __post_init__(self):
        super().__post_init__()
        if self.inv_mass_diag is None:
            self.inv_mass_diag = torch.ones(self.event_size)
        else:
            if self.inv_mass_diag.shape != (self.event_size,):
                raise ValueError
        if self.da is None:
            self.da = DualAveraging(self.step_size, self.da_params)


@dataclass
class MetropolisParameters(MCMCParameters):
    tune_inv_mass_diag: bool = False
    tune_step_size: bool = False
    adjustment: bool = True
    imd_adjustment: float = 1e-3


class MetropolisSampler(MCMCSampler):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: MetropolisKernel,
                 params: MetropolisParameters):
        super().__init__(event_shape, target, kernel, params)

    def update_kernel(self, data: Dict[str, Any]):
        self.kernel: MetropolisKernel
        self.params: MetropolisParameters
        x = data['x']
        mask = data['mask']
        n_chains = x.shape[0]

        # Update the inverse mass diagonal
        if n_chains > 1 and self.params.tune_inv_mass_diag:
            # self.kernel.inv_mass_diag = torch.std(x, dim=0)  # root of the preconditioning matrix diagonal
            self.kernel.inv_mass_diag = (
                    self.params.imd_adjustment * torch.var(x.flatten(1, -1), dim=0) +
                    (1 - self.params.imd_adjustment) * self.kernel.inv_mass_diag
            )
        if self.params.tune_step_size and self.params.adjustment:
            # Step size tuning is only possible with adjustment right now
            acc_rate = torch.mean(mask.float())
            error = self.kernel.da_params.target_acceptance_rate - acc_rate
            self.kernel.da.step(error)
            self.kernel.step_size = self.kernel.da.value  # Step size adaptation
