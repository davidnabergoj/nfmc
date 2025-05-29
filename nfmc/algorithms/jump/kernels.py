from typing import Tuple, Union

import torch
from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from nfmc.algorithms.jump.base import JumpMarkovKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.preconditioners import DenseLinearPreconditioner, DiagonalLinearPreconditioner, NormalizingFlowPreconditioner
from torchflows import Flow


def _create_latent_global_kernel(name: str,
                                 event_shape: Union[Tuple[int, ...], torch.Size],
                                 neg_log_prob_target: callable,
                                 **kwargs):
    if name == 'imh':
        return IMHKernel(event_shape, neg_log_prob_target, **kwargs)
    elif name == 'i-sir':
        return IteratedSIRKernel(event_shape, neg_log_prob_target, **kwargs)
    else:
        raise ValueError(
            f"Unrecognized global kernel specifier string: {name}"
        )


def _create_local_kernel(name: str,
                         event_shape: Union[Tuple[int, ...], torch.Size],
                         neg_log_prob_target: callable,
                         _prec: Union[str, NormalizingFlowPreconditioner],
                         **kwargs):
    if _prec == 'diagonal':
        preconditioner = DiagonalLinearPreconditioner(event_shape)
    elif _prec == 'dense':
        preconditioner = DenseLinearPreconditioner(event_shape)
    elif isinstance(_prec, NormalizingFlowPreconditioner):
        preconditioner = _prec
    else:
        raise ValueError("Preconditioner specifier not recognized")

    if name == 'rwmh':
        return RWMHKernel(event_shape, neg_log_prob_target, preconditioner=preconditioner, **kwargs)
    elif name == 'mala':
        return MALAKernel(event_shape, neg_log_prob_target, preconditioner=preconditioner, **kwargs)
    elif name == 'hmc':
        return HMCKernel(event_shape, neg_log_prob_target, preconditioner=preconditioner, **kwargs)
    else:
        raise ValueError(
            f"Unrecognized local kernel specifier string: {name}"
        )


def _create_kernels(neg_log_prob_target: callable,
                    global_kernel: str,
                    flow: Flow,
                    local_kernel: str,
                    local_preconditioner: str,
                    global_kwargs: dict = None,
                    local_kwargs: dict = None):
    global_prec = NormalizingFlowPreconditioner(flow)
    if local_preconditioner == 'nf':
        local_preconditioner = global_prec

    global_kernel = _create_latent_global_kernel(
        global_kernel,
        flow.event_shape,
        neg_log_prob_target,
        preconditioner=global_prec,
        **(global_kwargs or {})
    )
    local_kernel = _create_local_kernel(
        local_kernel,
        flow.event_shape,
        neg_log_prob_target,
        _prec=local_preconditioner,
        **(local_kwargs or {})
    )
    return local_kernel, global_kernel


class NeuTraJumpRWMHKernel(JumpMarkovKernel):
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

        :param Flow flow: normalizing flow for local and global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='rwmh',
                local_preconditioner='nf',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class NeuTraJumpMALAKernel(JumpMarkovKernel):
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

        :param Flow flow: normalizing flow for local and global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='mala',
                local_preconditioner='nf',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class NeuTraJumpHMCKernel(JumpMarkovKernel):
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

        :param Flow flow: normalizing flow for local and global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='hmc',
                local_preconditioner='nf',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class DiagonalJumpRWMHKernel(JumpMarkovKernel):
    """
    Composition of a diagonally-preconditioned local RWMH kernel and a normalizing flow-preconditioned jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpRWMHKernel constructor.

        :param Flow flow: normalizing flow for global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='rwmh',
                local_preconditioner='diagonal',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class DiagonalJumpMALAKernel(JumpMarkovKernel):
    """
    Composition of a diagonally-preconditioned local MALA kernel and a normalizing flow-preconditioned jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpMALAKernel constructor.

        :param Flow flow: normalizing flow for global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='mala',
                local_preconditioner='diagonal',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class DiagonalJumpHMCKernel(JumpMarkovKernel):
    """
    Composition of a diagonally-preconditioned local HMC kernel and a normalizing flow-preconditioned jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpHMCKernel constructor.

        :param Flow flow: normalizing flow for global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='hmc',
                local_preconditioner='diagonal',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class DenseJumpRWMHKernel(JumpMarkovKernel):
    """
    Composition of a densely-preconditioned local RWMH kernel and a normalizing flow-preconditioned jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpRWMHKernel constructor.

        :param Flow flow: normalizing flow for global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='rwmh',
                local_preconditioner='dense',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class DenseJumpMALAKernel(JumpMarkovKernel):
    """
    Composition of a densely-preconditioned local MALA kernel and a normalizing flow-preconditioned jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpMALAKernel constructor.

        :param Flow flow: normalizing flow for global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='mala',
                local_preconditioner='dense',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )


class DenseJumpHMCKernel(JumpMarkovKernel):
    """
    Composition of a densely-preconditioned local HMC kernel and a normalizing flow-preconditioned jump kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh',
                 **kwargs):
        """
        JumpHMCKernel constructor.

        :param Flow flow: normalizing flow for global preconditioning.
        :param neg_log_prob_target: negative log probability density callable.
        :param str global_kernel: type of global kernel. One of ['imh', 'i-sir'].
        :param kwargs: keyword arguments for both the local and global kernels.
        """
        super().__init__(
            *_create_kernels(
                neg_log_prob_target=neg_log_prob_target,
                global_kernel=global_kernel,
                flow=flow,
                local_kernel='hmc',
                local_preconditioner='dense',
                global_kwargs=kwargs,
                local_kwargs=kwargs
            ),
            **kwargs
        )
