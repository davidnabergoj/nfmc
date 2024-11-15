from typing import Sized, Optional

import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCOutput, MCMCParameters, MCMCKernel, MCMCStatistics
from dataclasses import dataclass


@dataclass
class NUTSKernel(MCMCKernel):
    event_size: int


@dataclass
class NUTSParameters(MCMCParameters):
    n_warmup_iterations: int = 100


class NUTS(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: Optional[NUTSKernel] = None,
                 params: Optional[NUTSParameters] = None):
        if kernel is None:
            kernel = NUTSKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if params is None:
            params = NUTSParameters()
        super().__init__(event_shape, target, kernel, params)

    def sample(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: NUTSKernel
        self.params: NUTSParameters

        from pyro.infer.mcmc import NUTS as NUTSPyro, MCMC as MCMCPyro
        n_chains: int = 1

        def potential_wrapper(x_dict):
            x = torch.column_stack([x_dict[f'd{i}'] for i in range(len(x_dict))])
            return self.target(x)

        initial_params = {f'd{i}': torch.randn(size=(n_chains,)) for i in range(self.kernel.event_size)}
        kernel = NUTSPyro(potential_fn=potential_wrapper)
        mcmc = MCMCPyro(
            kernel,
            num_samples=self.params.n_iterations,
            warmup_steps=self.params.n_warmup_iterations,
            initial_params=initial_params,
            num_chains=n_chains
        )
        mcmc.run()

        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=self.params.store_samples)
        out.running_samples.add(
            torch.concat([mcmc.get_samples()[f'd{i}'][:, :, None] for i in range(self.kernel.event_size)], dim=2)
        )

        # TODO better statistics
        out.statistics.update_counters(
            n_accepted_trajectories=n_chains * self.params.n_iterations,
            n_attempted_trajectories=n_chains * self.params.n_iterations,
        )

        out.kernel = self.kernel
        return out
