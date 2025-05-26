from torchflows.flows import Flow
from nfmc.algorithms.preconditioning.base import PreconditionedMCMCSampler
from nfmc.algorithms.jump.kernels import JumpRWMHKernel, JumpHMCKernel, JumpMALAKernel
from nfmc.algorithms.preconditioning.preconditioners import NormalizingFlowPreconditioner


class JumpRWMH(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = JumpRWMHKernel(
            event_shape=flow.event_shape,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            kernel=kernel,
            preconditioner=preconditioner
        )


class JumpMALA(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = JumpMALAKernel(
            event_shape=flow.event_shape,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            kernel=kernel,
            preconditioner=preconditioner
        )


class JumpHMC(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = JumpHMCKernel(
            event_shape=flow.event_shape,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel
        )
        preconditioner = NormalizingFlowPreconditioner(flow)
        super().__init__(
            kernel=kernel,
            preconditioner=preconditioner
        )
