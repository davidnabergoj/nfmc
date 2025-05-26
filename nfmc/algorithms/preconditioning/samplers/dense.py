from typing import Union, Tuple

import torch

from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.preconditioners import DenseLinearPreconditioner
from nfmc.algorithms.preconditioning.base import PreconditionedMCMCSampler
from torchflows import Flow


class DenseRWMH(PreconditionedMCMCSampler):
    """
    RWMH sampler, preconditioned with a matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        DenseRWMH constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = RWMHKernel(
            event_shape,
            neg_log_prob_target,
            **kwargs
        )
        preconditioner = DenseLinearPreconditioner(event_shape)
        super().__init__(latent_kernel, preconditioner)


class DenseMALA(PreconditionedMCMCSampler):
    """
    MALA sampler, preconditioned with a matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        DenseMALA constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = MALAKernel(
            event_shape,
            neg_log_prob_target,
            **kwargs
        )
        preconditioner = DenseLinearPreconditioner(event_shape)
        super().__init__(latent_kernel, preconditioner)


class DenseHMC(PreconditionedMCMCSampler):
    """
    HMC sampler, preconditioned with a matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        DenseHMC constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = HMCKernel(
            event_shape,
            neg_log_prob_target,
            **kwargs
        )
        preconditioner = DenseLinearPreconditioner(event_shape)
        super().__init__(latent_kernel, preconditioner)
