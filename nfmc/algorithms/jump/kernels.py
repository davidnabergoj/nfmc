from typing import Tuple, Union

import torch
from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from nfmc.algorithms.jump.base import JumpMarkovKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
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


class JumpRWMHKernel(JumpMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local RWMH kernel with a global jump kernel.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpRWMHKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        """
        latent_global_kernel = _create_latent_global_kernel(
            global_kernel,
            event_shape,
            neg_log_prob_target
        )

        latent_local_kernel = RWMHKernel(event_shape, neg_log_prob_target)
        super().__init__(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )


class JumpMALAKernel(JumpMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local MALA kernel with a global jump kernel.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpMALAKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        """
        latent_global_kernel = _create_latent_global_kernel(
            global_kernel,
            event_shape,
            neg_log_prob_target
        )

        latent_local_kernel = MALAKernel(event_shape, neg_log_prob_target)
        super().__init__(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )


class JumpHMCKernel(JumpMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local HMC kernel with a global jump kernel.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpHMCKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        """
        latent_global_kernel = _create_latent_global_kernel(
            global_kernel,
            event_shape,
            neg_log_prob_target
        )

        latent_local_kernel = HMCKernel(event_shape, neg_log_prob_target)
        super().__init__(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )
