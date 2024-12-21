import time
from dataclasses import dataclass
from typing import Union, Tuple, Type

import torch

from nfmc.algorithms.sampling.base import Sampler, NFMCKernel, NFMCParameters, MCMCOutput
from nfmc.algorithms.sampling.mcmc.hmc import HMC
from nfmc.algorithms.sampling.mcmc.base import MetropolisKernel, MetropolisParameters, MetropolisSampler
from nfmc.algorithms.sampling.mcmc.hmc import HMCKernel, HMCParameters
from nfmc.algorithms.sampling.mcmc.mh import MHKernel, MHParameters, MH


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
                'early_stopping_threshold': 5000,
                'keep_best_weights': True,
                'n_samples': 1,
                'n_epochs': 50000,
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
        _x, log_det_inverse = self.kernel.flow.bijection.inverse(_z.to(self.kernel.flow.get_device()))
        _x = _x.cpu()
        log_prob = -self.target(_x)
        adjusted_log_prob = log_prob + log_det_inverse.to(log_prob)
        adjusted_potential = -adjusted_log_prob
        if return_data:
            return adjusted_potential, _x
        else:
            return adjusted_potential

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        self.kernel: NeuTraKernel
        self.params: NeuTraParameters

        if time_limit_seconds is not None:
            flow_fit_time_limit = 0.3 * time_limit_seconds
        else:
            flow_fit_time_limit = None

        # Fit flow to target via variational inference
        t0 = time.time()
        self.kernel.flow.variational_fit(
            lambda v: -self.target(v),
            **{
                **dict(time_limit_seconds=flow_fit_time_limit),
                **self.params.warmup_fit_kwargs,
            },
            show_progress=show_progress
        )
        elapsed_time = time.time() - t0

        if time_limit_seconds is not None:
            inner_sampler_warmup_time_limit = time_limit_seconds - elapsed_time
        else:
            inner_sampler_warmup_time_limit = None

        # Tune MCMC
        self.inner_sampler.params.tuning_mode()
        self.inner_sampler.params.store_samples = self.params.store_samples
        self.inner_sampler.params.n_warmup_iterations = self.params.n_warmup_iterations
        return self.inner_sampler.warmup(
            x0,
            show_progress=show_progress,
            time_limit_seconds=inner_sampler_warmup_time_limit
        )

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        self.kernel: NeuTraKernel
        self.params: NeuTraParameters

        self.inner_sampler.params.n_iterations = self.params.n_iterations
        self.inner_sampler.params.sampling_mode()
        self.inner_sampler.params.store_samples = self.params.store_samples

        # Run MCMC with the adjusted target, returns output related to the original space (not the latent space)
        z0 = x0
        self.inner_sampler.data_transform = lambda v: self.kernel.flow.bijection.inverse(v)[0].detach()
        out = self.inner_sampler.sample(
            z0,
            show_progress=show_progress,
            time_limit_seconds=time_limit_seconds
        )
        out.kernel.flow = self.kernel.flow
        return out


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


class NeuTraMH(NeuTra):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 inner_kernel: MHKernel = None,
                 inner_params: MHParameters = None,
                 kernel: NeuTraKernel = None,
                 params: NeuTraParameters = None):
        if inner_kernel is None:
            inner_kernel = MHKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if inner_params is None:
            inner_params = MHParameters()
        super().__init__(event_shape, target, MH, inner_kernel, inner_params, kernel, params)
