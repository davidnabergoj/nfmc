from typing import Union, Tuple, Optional, Any
import torch.nn as nn

import torch

from nfmc.algorithms.mh.base import MHSampler

from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel

from nfmc.algorithms.preconditioning.preconditioners import NormalizingFlowPreconditioner
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler

from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from nfmc.algorithms.jump.kernels import NeuTraJumpHMCKernel, NeuTraJumpMALAKernel, NeuTraJumpRWMHKernel
from nfmc.algorithms.sampling.base.sampler import MCMCSampler
from nfmc.algorithms.util.samples import Samples
from nfmc.util import create_flow_object

PRECONDITIONED_SAMPLERS = [
    "imh", "fixed_imh",  # "imh" means "fixed_imh" by default
    "i-sir",
    "jump_hmc",
    "jump_mala",
    "jump_rwmh",
    "neutra_hmc",
    "neutra_mala",
    "neutra_rwmh",
    "ex2_hmc",
    "ex2_mala",
    "ex2_rwmh",
]


def create_sampler(neg_log_prob_target: callable,
                   event_shape: Optional[Union[torch.Size, Tuple[int]]] = None,
                   flow: Optional[Union[str, Any]] = 'realnvp',
                   kernel: str = "imh",
                   kernel_kwargs: Optional[dict] = None,
                   device: torch.device = torch.device("cpu"),
                   flow_kwargs: Optional[dict] = None) -> MCMCSampler:
    """
    Create the Sampler object.

    :param Union[callable, Potential] neg_log_prob_target: target distribution, specified by a negative log probability 
     density. This function takes as input a batch of tensors with shape `(batch_size, *event_shape)` and outputs a 
     batch with shape `(batch_size,)`.
    :param Tuple[int, ...] event_shape: shape of the event tensor. If `target` is an instance of Potential, this
     argument is unused.
    :param Union[str, Flow] flow: normalizing flow used in sampling. Must be provided when using a NFMC sampler.
     Can be either a `Flow` object or a string specifying the architecture.
     See `nfmc.util.get_supported_normalizing_flows` for a list of supported normalizing flows.
    :param str kernel: sampling strategy. See `nfmc.util.get_supported_samplers` for a list of sampling strategies.
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

    if flow is not None and not isinstance(flow, str):
        event_shape = flow.event_shape
    elif hasattr(neg_log_prob_target, "event_shape"):
        event_shape = neg_log_prob_target.event_shape

    if kernel in ['hmc', 'mala', 'rwmh']:
        # MCMC
        if kernel == "hmc":
            kernel_object = HMCKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                **kernel_kwargs
            )
        elif kernel == "mala":
            kernel_object = MALAKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                **kernel_kwargs
            )
        elif kernel == "rwmh":
            kernel_object = RWMHKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                **kernel_kwargs
            )
        else:
            raise ValueError(f"Unsupported sampling strategy: {kernel}")
        return MHSampler(kernel=kernel_object)

    elif kernel in PRECONDITIONED_SAMPLERS:
        # Create NF object
        if flow is None:
            raise ValueError("Flow object must be provided")
        if isinstance(flow, str):
            flow_object = create_flow_object(
                flow_string=flow, event_shape=event_shape, **flow_kwargs).to(device)
        elif isinstance(flow, nn.Module):
            flow_object = flow.to(device)
        else:
            raise ValueError(
                f"Unknown type for normalizing flow: {type(flow)}")

        if kernel in ["imh", "fixed_imh"]:
            kernel_object = IMHKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(flow=flow_object),
                **kernel_kwargs
            )
        elif kernel in ["i-sir", "isir"]:
            kernel_object = IteratedSIRKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(flow=flow_object),
                **kernel_kwargs
            )
        elif kernel == 'jump_hmc':
            kernel_object = NeuTraJumpHMCKernel(
                flow=flow_object,
                neg_log_prob_target=neg_log_prob_target,
                global_kernel='imh',
                **kernel_kwargs
            )
        elif kernel == 'jump_mala':
            kernel_object = NeuTraJumpMALAKernel(
                flow=flow_object,
                neg_log_prob_target=neg_log_prob_target,
                global_kernel='imh',
                **kernel_kwargs
            )
        elif kernel == 'jump_rwmh':
            kernel_object = NeuTraJumpRWMHKernel(
                flow=flow_object,
                neg_log_prob_target=neg_log_prob_target,
                global_kernel='imh',
                **kernel_kwargs
            )
        elif kernel == 'ex2_hmc':
            kernel_object = NeuTraJumpHMCKernel(
                flow=flow_object,
                neg_log_prob_target=neg_log_prob_target,
                global_kernel='i-sir',
                **kernel_kwargs
            )
        elif kernel == 'ex2_mala':
            kernel_object = NeuTraJumpMALAKernel(
                flow=flow_object,
                neg_log_prob_target=neg_log_prob_target,
                global_kernel='i-sir',
                **kernel_kwargs
            )
        elif kernel == 'ex2_rwmh':
            kernel_object = NeuTraJumpRWMHKernel(
                flow=flow_object,
                neg_log_prob_target=neg_log_prob_target,
                global_kernel='i-sir',
                **kernel_kwargs
            )
        elif kernel == 'neutra_hmc':
            kernel_object = HMCKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(
                    flow=flow_object
                ),
                **kernel_kwargs
            )
        elif kernel == 'neutra_mala':
            kernel_object = MALAKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(
                    flow=flow_object
                ),
                **kernel_kwargs
            )
        elif kernel == 'neutra_rwmh':
            kernel_object = RWMHKernel(
                event_shape=event_shape,
                neg_log_prob_target=neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(
                    flow=flow_object
                ),
                **kernel_kwargs
            )
        else:
            raise ValueError(f"Unsupported sampling strategy: {kernel}")
        return PreconditionedMCMCSampler(kernel=kernel_object)
    raise ValueError(f"Unsupported sampling strategy: {kernel}")


def sample(neg_log_prob_target: Union[callable, Any],
           event_shape: Optional[Union[torch.Size, Tuple[int, ...]]] = None,
           flow: Optional[Union[str, Any]] = 'realnvp',
           kernel: str = "imh",
           n_sampling_steps: int = 100,
           n_chains: int = 100,
           x0: torch.Tensor = None,
           warmup: bool = False,
           warmup_kwargs: dict = None,
           show_progress: bool = True,
           sampling_time_limit_seconds: Union[float, int] = None,
           warmup_time_limit_seconds: Union[float, int] = None,
           return_warmup_samples: bool = False,
           **kwargs) -> Union[Samples, Tuple[Samples, Samples]]:
    """
    Sample from a target distributions.

    :param Union[callable, Potential] neg_log_prob_target: target distribution, specified by a negative log probability 
     density. This function takes as input a batch of tensors with shape `(batch_size, *event_shape)` and outputs a batch with
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
    :rtype: Samples
    """
    warmup_kwargs = warmup_kwargs or {}
    if flow == 'None':
        flow = None
    if flow is not None and not isinstance(flow, str):
        event_shape = flow.event_shape
    elif hasattr(neg_log_prob_target, "event_shape"):
        event_shape = neg_log_prob_target.event_shape

    sampler = create_sampler(
        neg_log_prob_target=neg_log_prob_target,
        event_shape=event_shape,
        flow=flow,
        kernel=kernel,
        **kwargs
    )

    # Create initial state
    if x0 is None:
        x0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1

    if kernel in PRECONDITIONED_SAMPLERS:
        # x0 is treated as a latent state
        if warmup:
            warmup_output, _latent_output = sampler.warmup(
                x0,
                show_progress=show_progress,
                time_limit_seconds=warmup_time_limit_seconds,
                return_latent_samples=True,
                **warmup_kwargs
            )
            x0 = _latent_output.last_sample
        else:
            warmup_output = None
        sampling_output = sampler.sample(
            x0,
            n_steps=n_sampling_steps,
            show_progress=show_progress,
            time_limit_seconds=sampling_time_limit_seconds
        )
    else:
        # x0 is treated as the target state
        if warmup:
            warmup_output = sampler.warmup(
                x0=x0,
                show_progress=show_progress,
                time_limit_seconds=warmup_time_limit_seconds,
                **warmup_kwargs,
            )
            x0 = warmup_output.last_sample
        else:
            warmup_output = None
        sampling_output = sampler.sample(
            x0=x0,
            n_steps=n_sampling_steps,
            show_progress=show_progress,
            time_limit_seconds=sampling_time_limit_seconds
        )
    if return_warmup_samples:
        return sampling_output, warmup_output
    return sampling_output
