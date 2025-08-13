from nfmc.algorithms.kernel import CompositionKernel, MarkovKernel
from nfmc.algorithms.mh.base import MarkovKernel


class JumpMarkovKernel(CompositionKernel):
    """
    Class for Jump Markov kernels.

    Jump Markov kernels are compositions of a local Markov kernel and a global Markov kernel. Global Markov kernel 
     examples include an independent Metropolis-Hastings kernel or an i-SIR kernel.
    """

    def __init__(self,
                 local_kernel: MarkovKernel,
                 global_kernel: MarkovKernel,
                 local_transitions_per_step: int = 20,
                 global_transitions_per_step: int = 1,
                 **kwargs):
        super().__init__(
            [local_kernel, global_kernel],
            mode='cyclic',
            schedule=[local_transitions_per_step, global_transitions_per_step],
            **kwargs
        )

    @property
    def name(self):
        return "Generic jump kernel"
