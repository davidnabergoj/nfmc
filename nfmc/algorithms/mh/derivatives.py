from nfmc.algorithms.jump import JumpMarkovKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.base import PreconditionedMarkovKernel
from nfmc.algorithms.preconditioning.preconditioners import NormalizingFlowPreconditioner
from torchflows.flows import Flow


class JumpRWMHKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local RWMH kernel with a global IMH kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        JumpRWMHKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_local_kernel = RWMHKernel(flow.event_shape, neg_log_prob_target)
        latent_global_kernel = IMHKernel(flow.event_shape, neg_log_prob_target)
        latent_jump_kernel = JumpMarkovKernel(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_jump_kernel,
            preconditioner=preconditioner
            ** kwargs
        )


class JumpMALAKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local MALA kernel with a global IMH kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        JumpMALAKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_local_kernel = MALAKernel(flow.event_shape, neg_log_prob_target)
        latent_global_kernel = IMHKernel(flow.event_shape, neg_log_prob_target)
        latent_jump_kernel = JumpMarkovKernel(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_jump_kernel,
            preconditioner=preconditioner
            ** kwargs
        )


class JumpHMCKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned composition of a local HMC kernel with a global IMH kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        JumpHMCKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_local_kernel = HMCKernel(flow.event_shape, neg_log_prob_target)
        latent_global_kernel = IMHKernel(flow.event_shape, neg_log_prob_target)
        latent_jump_kernel = JumpMarkovKernel(
            latent_local_kernel,
            latent_global_kernel,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_jump_kernel,
            preconditioner=preconditioner
            ** kwargs
        )


class FlowIMHKernel(PreconditionedMarkovKernel):
    """
    Normalizing flow-preconditioned IMH kernel.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        JumpHMCKernel constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_global_kernel = IMHKernel(flow.event_shape, neg_log_prob_target)
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            base_kernel=latent_global_kernel,
            preconditioner=preconditioner
            ** kwargs
        )
