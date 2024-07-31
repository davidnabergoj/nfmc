from dataclasses import dataclass
from typing import Sized

import torch

from nfmc.algorithms.sampling.base import Sampler, NFMCKernel, NFMCParameters, MCMCOutput
from nfmc.algorithms.sampling.mcmc import HMC
from nfmc.algorithms.sampling.mcmc.hmc import HMCKernel, HMCParameters


@dataclass
class NeuTraKernel(NFMCKernel):
    pass


@dataclass
class NeuTraParameters(NFMCParameters):
    n_vi_iterations: int = 100
    batch_inverse_size: int = 128


class NeuTraHMC(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 inner_kernel: HMCKernel = None,
                 inner_params: HMCParameters = None,
                 kernel: NeuTraKernel = None,
                 params: NeuTraParameters = None):
        if inner_kernel is None:
            inner_kernel = HMCKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if inner_params is None:
            inner_params = HMCParameters()
        if kernel is None:
            kernel = NeuTraKernel(event_shape)
        if params is None:
            params = NeuTraParameters()
        super().__init__(event_shape, target, kernel, params)
        self.inner_kernel = inner_kernel
        self.inner_params = inner_params
        self.inner_params.n_iterations = self.params.n_iterations  # TODO handle this better

    def sample(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: NeuTraKernel
        self.params: NeuTraParameters
        self.inner_params.n_iterations = self.params.n_iterations  # TODO handle this better

        n_chains, *event_shape = x0.shape

        # Fit flow to target via variational inference
        self.kernel.flow.variational_fit(
            lambda v: -self.target(v),
            n_epochs=self.params.n_vi_iterations,
            show_progress=show_progress
        )

        # Run HMC with target being the flow
        x0 = self.kernel.flow.sample(n_chains)
        z0, _ = self.kernel.flow.bijection.forward(x0)

        def adjusted_target(_z):
            _x, log_det_inverse = self.kernel.flow.bijection.inverse(_z)
            log_prob = -self.target(_x)
            adjusted_log_prob = log_prob + log_det_inverse
            return -adjusted_log_prob

        # TODO handle HMC tuning here
        inner_sampler = HMC(event_shape, adjusted_target, self.inner_kernel, self.inner_params)
        mcmc_output = inner_sampler.sample(z0.detach(), show_progress=show_progress)

        with torch.no_grad():
            xs, _ = self.kernel.flow.bijection.batch_inverse(
                mcmc_output.samples,
                batch_size=self.params.batch_inverse_size
            )
            xs = xs.detach()

        return MCMCOutput(samples=xs, statistics=mcmc_output.statistics)
