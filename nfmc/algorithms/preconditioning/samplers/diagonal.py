from typing import Union, Tuple

import torch

from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.preconditioners import DiagonalLinearPreconditioner
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
from torchflows.flows import Flow


class DiagonalRWMH(PreconditionedMCMCSampler):
    """
    RWMH sampler, preconditioned with a diagonal matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        DiagonalRWMH constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = RWMHKernel(
            event_shape,
            neg_log_prob_target,
            preconditioner=DiagonalLinearPreconditioner(event_shape),
            ** kwargs
        )
        super().__init__(latent_kernel)


class DiagonalMALA(PreconditionedMCMCSampler):
    """
    MALA sampler, preconditioned with a diagonal matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        DiagonalMALA constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = MALAKernel(
            event_shape,
            neg_log_prob_target,
            preconditioner=DiagonalLinearPreconditioner(event_shape),
            **kwargs
        )
        super().__init__(latent_kernel)


class DiagonalHMC(PreconditionedMCMCSampler):
    """
    HMC sampler, preconditioned with a diagonal matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        DiagonalHMC constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = HMCKernel(
            event_shape,
            neg_log_prob_target,
            preconditioner=DiagonalLinearPreconditioner(event_shape),
            **kwargs
        )
        super().__init__(latent_kernel)
