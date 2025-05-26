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
                 **kwargs):
        super().__init__(
            [local_kernel, global_kernel],
            **kwargs
        )

    @property
    def name(self):
        return "Generic jump kernel"
