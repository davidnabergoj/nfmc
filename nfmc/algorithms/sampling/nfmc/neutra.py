import time
from dataclasses import dataclass
from typing import Sized, Union, Tuple, Type

import torch
from prometheus_client.samples import Sample

from nfmc.algorithms.sampling.base import Sampler, NFMCKernel, NFMCParameters, MCMCOutput, MCMCStatistics
from nfmc.algorithms.sampling.mcmc import HMC
from nfmc.algorithms.sampling.mcmc.base import MCMCSampler, MetropolisKernel, MetropolisParameters, MetropolisSampler
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


class NeuTra(Sampler):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 inner_sampler_class: Type[MetropolisSampler],
                 inner_kernel: MetropolisKernel,
                 inner_params: MetropolisParameters,
                 kernel: NeuTraKernel = None,
                 params: NeuTraParameters = None):
        if kernel is None:
            kernel = NeuTraKernel(event_shape)
        if params is None:
            params = NeuTraParameters()
        super().__init__(event_shape, target, kernel, params)
        inner_params.n_iterations = self.params.n_iterations
        self.inner_sampler = inner_sampler_class(
            event_shape,
            self.adjusted_target,
            inner_kernel,
            inner_params
        )

    def adjusted_target(self, _z, return_data: bool = False):
        self.kernel: NFMCKernel
        _x, log_det_inverse = self.kernel.flow.bijection.inverse(_z)
        log_prob = -self.target(_x)
        adjusted_log_prob = log_prob + log_det_inverse
        adjusted_potential = -adjusted_log_prob
        if return_data:
            return adjusted_potential, _x
        else:
            return adjusted_potential

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               **kwargs) -> MCMCOutput:
        self.kernel: NeuTraKernel
        self.params: NeuTraParameters

        # Fit flow to target via variational inference
        self.kernel.flow.variational_fit(
            lambda v: -self.target(v),
            **self.params.warmup_fit_kwargs,
            show_progress=show_progress
        )

        # Tune MCMC
        self.inner_sampler.params.tuning_mode()
        return self.inner_sampler.warmup(x0, show_progress=show_progress, **kwargs)

    def sample(self, x0: torch.Tensor, **kwargs) -> MCMCOutput:
        self.kernel: NeuTraKernel
        self.params: NeuTraParameters

        n_chains = x0.shape[0]
        self.inner_sampler.params.n_iterations = self.params.n_iterations
        self.inner_sampler.params.sampling_mode()
        self.inner_sampler.params.store_samples = self.params.store_samples
        z0 = self.kernel.flow.base_sample((n_chains,)).detach()

        # Run MCMC with the adjusted target, returns output related to the original space (not the latent space)
        return self.inner_sampler.sample(
            z0,
            data_transform=lambda v: self.kernel.flow.bijection.inverse(v)[0].detach(),
            **kwargs
        )


class NeuTraHMC(NeuTra):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 inner_kernel: HMCKernel = None,
                 inner_params: HMCParameters = None,
                 kernel: NeuTraKernel = None,
                 params: NeuTraParameters = None):
        if inner_kernel is None:
            inner_kernel = HMCKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if inner_params is None:
            inner_params = HMCParameters()
        super().__init__(event_shape, target, HMC, inner_kernel, inner_params, kernel, params)
