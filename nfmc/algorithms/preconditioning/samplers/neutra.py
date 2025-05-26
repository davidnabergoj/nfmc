from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.preconditioners import NormalizingFlowPreconditioner
from nfmc.algorithms.preconditioning.base import PreconditionedMCMCSampler
from torchflows import Flow


class NeuTraRWMH(PreconditionedMCMCSampler):
    """
    Normalizing flow-preconditioned RWMH sampler.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        NeuTraRWMH constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = RWMHKernel(
            flow.event_shape,
            neg_log_prob_target,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(latent_kernel, preconditioner)


class NeuTraMALA(PreconditionedMCMCSampler):
    """
    Normalizing flow-preconditioned MALA sampler.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        NeuTraMALA constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = MALAKernel(
            flow.event_shape,
            neg_log_prob_target,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(latent_kernel, preconditioner)


class NeuTraHMC(PreconditionedMCMCSampler):
    """
    Normalizing flow-preconditioned HMC sampler.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 **kwargs):
        """
        NeuTraHMC constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        latent_kernel = HMCKernel(
            flow.event_shape,
            neg_log_prob_target,
            **kwargs
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(latent_kernel, preconditioner)
