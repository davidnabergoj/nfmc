from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.nuts import NUTSKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.iterated_sir import IteratedSIRKernel
import torch
import pytest

@pytest.mark.parametrize("kernel_class", [
    HMCKernel,
    RWMHKernel,
    MALAKernel,
    NUTSKernel,
    IMHKernel,
    IteratedSIRKernel
])
def test_basic(kernel_class):
    n_chains = 4
    kernel = HMCKernel(
        event_shape=(2,),
        neg_log_prob_target=lambda x: 0.5 * (x ** 2).sum(dim=-1)
    )
    x = torch.full(
        size=(n_chains, *kernel.event_shape),
        fill_value=torch.nan
    )

    with pytest.raises(ValueError):
        kernel.step(x)
