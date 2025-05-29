from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.preconditioners import NormalizingFlowPreconditioner
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
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
        kernel = RWMHKernel(
            flow.event_shape,
            neg_log_prob_target,
            preconditioner = NormalizingFlowPreconditioner(flow),
            **kwargs
        )
        super().__init__(kernel)


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
        kernel = MALAKernel(
            flow.event_shape,
            neg_log_prob_target,
            preconditioner = NormalizingFlowPreconditioner(flow),
            **kwargs
        )
        super().__init__(kernel)


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
        kernel = HMCKernel(
            flow.event_shape,
            neg_log_prob_target,
            preconditioner = NormalizingFlowPreconditioner(flow),
            **kwargs
        )
        super().__init__(kernel)
