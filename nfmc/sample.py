from typing import Union, Tuple, Optional

import torch

from nfmc.algorithms.sampling.base import NFMCKernel, MCMCOutput, Sampler
from nfmc.algorithms.sampling.mcmc.ess import ESSKernel, ESSParameters, ESS
from nfmc.algorithms.sampling.mcmc.hmc import HMCKernel, HMCParameters, UHMC, HMC
from nfmc.algorithms.sampling.mcmc.langevin import LangevinKernel, LangevinParameters, ULA, MALA
from nfmc.algorithms.sampling.mcmc.mh import MHKernel, MHParameters, MH
from nfmc.algorithms.sampling.nfmc.dlmc import DLMCKernel, DLMCParameters, DLMC
from nfmc.algorithms.sampling.nfmc.imh import IMHKernel, IMHParameters, FixedIMH, AdaptiveIMH
from nfmc.algorithms.sampling.nfmc.jump import JumpNFMCParameters, JumpULA, JumpHMC, JumpUHMC, JumpMALA, JumpMH, JumpESS
from nfmc.algorithms.sampling.nfmc.neutra import NeuTraKernel, NeuTraParameters, NeuTraHMC, NeuTraMH
from nfmc.algorithms.sampling.nfmc.tess import TESSKernel, TESSParameters, TESS
from nfmc.util import create_flow_object
from torchflows.flows import Flow
from potentials.base import Potential


def create_sampler(target: callable,
                   event_shape: Optional[Union[torch.Size, Tuple[int]]] = None,
                   flow: Optional[Union[str, Flow]] = 'realnvp',
                   strategy: str = "imh",
                   negative_log_likelihood: callable = None,
                   kernel_kwargs: Optional[dict] = None,
                   param_kwargs: Optional[dict] = None,
                   inner_kernel_kwargs: Optional[dict] = None,
                   inner_param_kwargs: Optional[dict] = None,
                   device: torch.device = torch.device("cpu"),
                   flow_kwargs: Optional[dict] = None) -> Sampler:
    """
    Create the Sampler object.

    :param Union[callable, Potential] target: target distribution, specified by a negative log probability density.
     This function takes as input a batch of tensors with shape `(batch_size, *event_shape)` and outputs a batch with
     shape `(batch_size,)`.
    :param Tuple[int, ...] event_shape: shape of the event tensor. If `target` is an instance of Potential, this
     argument is unused.
    :param Union[str, Flow] flow: normalizing flow used in sampling. Must be provided when using a NFMC sampler.
     Can be either a `Flow` object or a string specifying the architecture.
     See `nfmc.util.get_supported_normalizing_flows` for a list of supported normalizing flows.
    :param str strategy: sampling strategy. See `nfmc.util.get_supported_samplers` for a list of sampling strategies.
    :param Union[callable, Potential] negative_log_likelihood: auxiliary negative log probability density. Used in
     specific samplers like DLMC and TESS. This function takes as input a batch of tensors with shape
     `(batch_size, *event_shape)` and outputs a batch with shape `(batch_size,)`.
    :param dict kernel_kwargs: keyword arguments for the sampler kernel object (subclass of `MCMCKernel`).
    :param dict param_kwargs: keyword arguments for the sampler parameter object (subclass of `MCMCParameters`).
    :param dict inner_kernel_kwargs: keyword arguments for the kernel object of the inner sampler
     (subclass of `MCMCKernel`).
    :param dict inner_param_kwargs: keyword arguments for the parameter object of the inner sampler
     (subclass of `MCMCParameters`).
    :param torch.device device: torch device for normalizing flow operations.
    :param flow_kwargs: keyword arguments for `create_flow_object`.
    :return: Sampler object.
    :rtype: Sampler
    """
    flow_kwargs = flow_kwargs or {}
    kernel_kwargs = kernel_kwargs or {}
    param_kwargs = param_kwargs or {'n_iterations': 100}
    inner_kernel_kwargs = inner_kernel_kwargs or {}
    inner_param_kwargs = inner_param_kwargs or {}

    if flow is not None and not isinstance(flow, str):
        event_shape = flow.event_shape
    elif isinstance(target, Potential):
        event_shape = target.event_shape

    event_size = int(torch.prod(torch.as_tensor(event_shape)))

    if strategy in ['hmc', 'uhmc', 'ula', 'mala', 'mh', 'ess']:
        # MCMC
        if strategy == "hmc":
            kernel = HMCKernel(event_size=event_size, **kernel_kwargs)
            params = HMCParameters(**param_kwargs)
            return HMC(event_shape, target, kernel, params)
        elif strategy == "uhmc":
            kernel = HMCKernel(event_size=event_size, **kernel_kwargs)
            params = HMCParameters(**param_kwargs)
            return UHMC(event_shape, target, kernel, params)
        elif strategy == "mala":
            kernel = LangevinKernel(event_size=event_size, **kernel_kwargs)
            params = LangevinParameters(**param_kwargs)
            return MALA(event_shape, target, kernel, params)
        elif strategy == "ula":
            kernel = LangevinKernel(event_size=event_size, **kernel_kwargs)
            params = LangevinParameters(**param_kwargs)
            return ULA(event_shape, target, kernel, params)
        elif strategy == "mh":
            kernel = MHKernel(event_size=event_size, **kernel_kwargs)
            params = MHParameters(**param_kwargs)
            return MH(event_shape, target, kernel, params)
        elif strategy == "ess":
            if negative_log_likelihood is None:
                raise ValueError("Negative log likelihood must be provided")
            kernel = ESSKernel(event_shape=event_shape, **kernel_kwargs)
            params = ESSParameters(**param_kwargs)
            return ESS(event_shape, target, negative_log_likelihood, kernel, params)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
    elif strategy in [
        "imh", "fixed_imh",  # "imh" means "fixed_imh" by default
        "adaptive_imh",
        "jump_mala",
        "jump_ula",
        "jump_hmc",
        "jump_uhmc",
        "jump_ess",
        "jump_mh",
        "neutra_hmc",
        "neutra_mh",
        "tess",
        "dlmc"
    ]:
        # NFMC
        if flow is None:
            raise ValueError("Flow object must be provided")
        if isinstance(flow, str):
            flow_object = create_flow_object(flow_string=flow, event_shape=event_shape, **flow_kwargs).to(device)
        elif isinstance(flow, Flow):
            flow_object = flow.to(device)
        else:
            raise ValueError(f"Unknown type for normalizing flow: {type(flow)}")
        if strategy in ["imh", "fixed_imh"]:
            kernel = IMHKernel(event_shape, flow=flow_object)
            params = IMHParameters(**param_kwargs)
            return FixedIMH(event_shape, target, kernel, params)
        if strategy == "adaptive_imh":
            kernel = IMHKernel(event_shape, flow=flow_object)
            param_kwargs.update({
                'adaptation': True
            })
            params = IMHParameters(**param_kwargs)
            return AdaptiveIMH(event_shape, target, kernel, params)
        elif strategy == 'jump_mala':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(**param_kwargs)
            inner_kernel = LangevinKernel(event_size=event_size, **inner_kernel_kwargs)
            inner_params = LangevinParameters(**inner_param_kwargs)
            return JumpMALA(
                event_shape,
                target,
                kernel=kernel,
                params=params,
                inner_kernel=inner_kernel,
                inner_params=inner_params
            )
        elif strategy == 'jump_ula':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(**param_kwargs)
            inner_kernel = LangevinKernel(event_size=event_size, **inner_kernel_kwargs)
            inner_params = LangevinParameters(**inner_param_kwargs)
            return JumpULA(
                event_shape,
                target,
                kernel=kernel,
                params=params,
                inner_kernel=inner_kernel,
                inner_params=inner_params
            )
        elif strategy == 'jump_hmc':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(**param_kwargs)
            inner_kernel = HMCKernel(event_size=event_size, **inner_kernel_kwargs)
            if 'n_iterations' not in inner_param_kwargs:
                inner_param_kwargs['n_iterations'] = 5
            inner_params = HMCParameters(**inner_param_kwargs)
            return JumpHMC(
                event_shape,
                target,
                kernel=kernel,
                params=params,
                inner_kernel=inner_kernel,
                inner_params=inner_params
            )
        elif strategy == 'jump_uhmc':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(**param_kwargs)
            inner_kernel = HMCKernel(event_size=event_size, **inner_kernel_kwargs)
            inner_params = HMCParameters(**inner_param_kwargs)
            return JumpUHMC(
                event_shape,
                target,
                kernel=kernel,
                params=params,
                inner_kernel=inner_kernel,
                inner_params=inner_params
            )
        elif strategy == 'jump_mh':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(**param_kwargs)
            inner_kernel = MHKernel(event_size=event_size, **inner_kernel_kwargs)
            inner_params = MHParameters(**inner_param_kwargs)
            return JumpMH(
                event_shape,
                target,
                kernel=kernel,
                params=params,
                inner_kernel=inner_kernel,
                inner_params=inner_params
            )
        elif strategy == 'jump_ess':
            if negative_log_likelihood is None:
                raise ValueError("Negative log likelihood must be provided")
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(**param_kwargs)
            inner_kernel = ESSKernel(event_shape=event_shape, **inner_kernel_kwargs)
            inner_params = ESSParameters(**inner_param_kwargs)
            return JumpESS(
                event_shape,
                target,
                negative_log_likelihood=negative_log_likelihood,
                kernel=kernel,
                params=params,
                inner_kernel=inner_kernel,
                inner_params=inner_params
            )
        elif strategy == "tess":
            if negative_log_likelihood is None:
                raise ValueError("Negative log likelihood must be provided")
            kernel = TESSKernel(event_shape, flow=flow_object)
            params = TESSParameters(**param_kwargs)
            return TESS(event_shape, target, negative_log_likelihood, kernel, params)
        elif strategy == "dlmc":
            if negative_log_likelihood is None:
                raise ValueError("Negative log likelihood must be provided")
            kernel = DLMCKernel(event_shape, flow=flow_object)
            params = DLMCParameters(**param_kwargs)
            return DLMC(event_shape, target, negative_log_likelihood, kernel, params)
        elif strategy == 'neutra_hmc':
            kernel = NeuTraKernel(event_shape, flow=flow_object)
            params = NeuTraParameters(**param_kwargs)
            inner_kernel = HMCKernel(event_size=event_size, **inner_kernel_kwargs)
            inner_params = HMCParameters(**inner_param_kwargs)
            return NeuTraHMC(event_shape, target, inner_kernel, inner_params, kernel, params)
        elif strategy == 'neutra_mh':
            kernel = NeuTraKernel(event_shape, flow=flow_object)
            params = NeuTraParameters(**param_kwargs)
            inner_kernel = MHKernel(event_size=event_size, **inner_kernel_kwargs)
            inner_params = MHParameters(**inner_param_kwargs)
            return NeuTraMH(event_shape, target, inner_kernel, inner_params, kernel, params)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def sample(target: Union[callable, Potential],
           event_shape: Optional[Union[torch.Size, Tuple[int, ...]]] = None,
           flow: Optional[Union[str, Flow]] = 'realnvp',
           strategy: str = "imh",
           n_iterations: int = 100,
           n_warmup_iterations: int = 100,
           n_chains: int = 100,
           x0: torch.Tensor = None,
           warmup: bool = False,
           show_progress: bool = True,
           sampling_time_limit_seconds: Union[float, int] = None,
           warmup_time_limit_seconds: Union[float, int] = None,
           **kwargs) -> MCMCOutput:
    """
    Sample from a target distributions.

    :param Union[callable, Potential] target: target distribution, specified by a negative log probability density.
     This function takes as input a batch of tensors with shape `(batch_size, *event_shape)` and outputs a batch with
     shape `(batch_size,)`.
    :param Tuple[int, ...] event_shape: shape of the event tensor. If `target` is an instance of Potential, this
     argument is unused.
    :param Union[str, Flow] flow: normalizing flow used in sampling. Must be provided when using a NFMC sampler.
     Can be either a `Flow` object or a string specifying the architecture.
     See `nfmc.util.get_supported_normalizing_flows` for a list of supported normalizing flows.
    :param str strategy: sampling strategy. See `nfmc.util.get_supported_samplers` for a list of sampling strategies.
    :param torch.Tensor x0: initial chain states with shape `(n_chains, *event_shape)`.
    :param int n_iterations: number of iterations for sampling.
    :param int n_warmup_iterations: number of iterations for warmup.
    :param int n_chains: number of chains for sampling (and warm-up, if specified). If `x0` is provided, this argument
     is unused.
    :param bool warmup: if True, perform a warm-up phase before sampling.
    :param bool show_progress: if True, display a progress bar during sampling and warmup.
    :param Union[float, int] sampling_time_limit_seconds: time limit for sampling.
    :param Union[float, int] warmup_time_limit_seconds: time limit for warmup.
    :param dict kwargs: keyword arguments for `create_sampler`.
    :return: sampling output object.
    :rtype: MCMCOutput
    """
    if flow is not None and not isinstance(flow, str):
        event_shape = flow.event_shape
    elif isinstance(target, Potential):
        event_shape = target.event_shape
    if 'param_kwargs' not in kwargs:
        kwargs['param_kwargs'] = {}
    kwargs['param_kwargs'] = {
        **kwargs['param_kwargs'],
        **dict(
            n_iterations=n_iterations,
            n_warmup_iterations=n_warmup_iterations
        )
    }

    sampler = create_sampler(
        target=target,
        event_shape=event_shape,
        flow=flow,
        strategy=strategy,
        **kwargs
    )
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *event_shape))

    if warmup:
        warmup_output = sampler.warmup(x0=x0, show_progress=show_progress, time_limit_seconds=warmup_time_limit_seconds)
        if warmup_output.samples is not None:
            x0 = warmup_output.samples.flatten(0, 1)
            x0 = x0[torch.randperm(len(x0))][:n_chains]
        else:
            x0 = warmup_output.running_samples.last_sample
    return sampler.sample(x0=x0, show_progress=show_progress, time_limit_seconds=sampling_time_limit_seconds)
