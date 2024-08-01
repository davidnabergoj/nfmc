import time
from dataclasses import dataclass
from typing import Sized

import torch

from nfmc.algorithms.sampling.base import Sampler, NFMCKernel, NFMCParameters, MCMCOutput, MCMCStatistics
from nfmc.algorithms.sampling.mcmc import HMC
from nfmc.algorithms.sampling.mcmc.hmc import HMCKernel, HMCParameters


@dataclass
class NeuTraKernel(NFMCKernel):
    pass


@dataclass
class NeuTraParameters(NFMCParameters):
    batch_inverse_size: int = 128
    warmup_fit_kwargs: dict = None

    def __post_init__(self):
        if self.warmup_fit_kwargs is None:
            self.warmup_fit_kwargs = {
                'early_stopping': True,
                'early_stopping_threshold': 50,
                'keep_best_weights': True,
                'n_samples': 1,
                'n_epochs': 500,
                'lr': 0.05
            }


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
        inner_params.n_iterations = self.params.n_iterations
        self.inner_sampler = HMC(event_shape, self.adjusted_target, inner_kernel, inner_params)

    def adjusted_target(self, _z):
        _x, log_det_inverse = self.kernel.flow.bijection.inverse(_z)
        log_prob = -self.target(_x)
        adjusted_log_prob = log_prob + log_det_inverse
        return -adjusted_log_prob

    def warmup(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: NeuTraKernel
        self.params: NeuTraParameters

        # Fit flow to target via variational inference
        self.kernel.flow.variational_fit(
            lambda v: -self.target(v),
            **self.params.warmup_fit_kwargs,
            show_progress=show_progress
        )

        # Tune HMC
        warmup_output = self.inner_sampler.warmup(x0, show_progress=show_progress)
        return warmup_output

    def sample(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: NeuTraKernel
        self.params: NeuTraParameters

        n_chains, *event_shape = x0.shape
        self.inner_sampler.params.n_iterations = self.params.n_iterations
        self.inner_sampler.params.tune_step_size = False
        self.inner_sampler.params.tune_inv_mass_diag = False


        # Run HMC with the adjusted target
        t0 = time.time()
        z0 = self.kernel.flow.base_sample((n_chains,)).detach()
        t1 = time.time()

        mcmc_output = self.inner_sampler.sample(z0, show_progress=show_progress)
        mcmc_output.statistics.elapsed_time_seconds += t1 - t0

        t0 = time.time()
        with torch.no_grad():
            xs, _ = self.kernel.flow.bijection.batch_inverse(
                mcmc_output.samples,
                batch_size=self.params.batch_inverse_size
            )
            xs = xs.detach()
        mcmc_output.statistics.elapsed_time_seconds += time.time() - t0

        return MCMCOutput(samples=xs, statistics=mcmc_output.statistics)
