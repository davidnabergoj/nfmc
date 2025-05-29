from torchflows.flows import Flow
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
from nfmc.algorithms.jump.kernels import DiagonalJumpHMCKernel, DiagonalJumpMALAKernel, DiagonalJumpRWMHKernel, NeuTraJumpRWMHKernel, NeuTraJumpHMCKernel, NeuTraJumpMALAKernel
from nfmc.algorithms.preconditioning.preconditioners import NormalizingFlowPreconditioner


class NeuTraJumpRWMH(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = NeuTraJumpRWMHKernel(
            flow=flow,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel,
        )
        super().__init__(
            kernel=kernel,
        )


class NeuTraJumpMALA(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = NeuTraJumpMALAKernel(
            flow=flow,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel,
        )
        super().__init__(
            kernel=kernel,
        )


class NeuTraJumpHMC(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = NeuTraJumpHMCKernel(
            flow=flow,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel,
        )
        super().__init__(
            kernel=kernel,
        )


class DiagonalJumpRWMH(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = DiagonalJumpRWMHKernel(
            flow=flow,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel,
        )
        super().__init__(
            kernel=kernel,
        )


class DiagonalJumpMALA(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = DiagonalJumpMALAKernel(
            flow=flow,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel,
        )
        super().__init__(
            kernel=kernel,
        )


class DiagonalJumpHMC(PreconditionedMCMCSampler):
    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 global_kernel: str = 'imh'):
        kernel = DiagonalJumpHMCKernel(
            flow=flow,
            neg_log_prob_target=neg_log_prob_target,
            global_kernel=global_kernel,
        )
        super().__init__(
            kernel=kernel,
        )
