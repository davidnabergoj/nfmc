from typing import Tuple, Union

import torch
from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from nfmc.algorithms.jump.base import JumpMarkovKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.base import PreconditionedMarkovKernel
from nfmc.algorithms.preconditioning.preconditioners import NormalizingFlowPreconditioner
from torchflows.flows import Flow


def _create_latent_global_kernel(name: str,
                                 event_shape: Union[Tuple[int, ...], torch.Size],
                                 neg_log_prob_target: callable):
    if name == 'imh':
        return IMHKernel(event_shape, neg_log_prob_target)
    elif name == 'i-sir':
        return IteratedSIRKernel(event_shape, neg_log_prob_target)
    else:
        raise ValueError(
            f"Unrecognized global kernel specifier string: {name}"
        )


class JumpRWMHKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local RWMH kernel with a global jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpRWMHKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        """
        latent_global_kernel = _create_latent_global_kernel(
            global_kernel,
            flow.event_shape,
            neg_log_prob_target
        )

        latent_local_kernel = RWMHKernel(flow.event_shape, neg_log_prob_target)
        latent_jump_kernel = JumpMarkovKernel(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_jump_kernel,
            preconditioner=preconditioner,
            **kwargs
        )


class JumpMALAKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local MALA kernel with a global jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpMALAKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        """
        latent_global_kernel = _create_latent_global_kernel(
            global_kernel,
            flow.event_shape,
            neg_log_prob_target
        )

        latent_local_kernel = MALAKernel(flow.event_shape, neg_log_prob_target)
        latent_jump_kernel = JumpMarkovKernel(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_jump_kernel,
            preconditioner=preconditioner,
            **kwargs
        )


class JumpHMCKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local HMC kernel with a global jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpHMCKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        """
        latent_global_kernel = _create_latent_global_kernel(
            global_kernel,
            flow.event_shape,
            neg_log_prob_target
        )

        latent_local_kernel = HMCKernel(flow.event_shape, neg_log_prob_target)
        latent_jump_kernel = JumpMarkovKernel(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_jump_kernel,
            preconditioner=preconditioner,
            **kwargs
        )


class JumpKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned global jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        """
        latent_global_kernel = _create_latent_global_kernel(
            global_kernel,
            flow.event_shape,
            neg_log_prob_target
        )

        latent_global_kernel = IMHKernel(flow.event_shape, neg_log_prob_target)
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_global_kernel,
            preconditioner=preconditioner,
            **kwargs
        )
