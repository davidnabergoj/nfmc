from nfmc.algorithms.mh.base import MHKernel


class JumpMHKernel(MHKernel):
    def __init__(self, event_shape, neg_log_prob_target):
        super().__init__(event_shape, neg_log_prob_target)