import torch
from nfmc.algorithms.sampling.mcmc.mh import MH, MHParameters
import pytest


@pytest.mark.parametrize('n_chains', [1, 2, 7])
@pytest.mark.parametrize('max_samples', [1, 10, 50])
@pytest.mark.parametrize('n_dim', [1, 2, 5])
def test_shape(n_chains, max_samples, n_dim):
    torch.manual_seed(0)
    sampler = MH(
        event_shape=(n_dim,),
        target = lambda x: torch.sum(x ** 2, dim=-1),
        params=MHParameters(max_samples=max_samples, n_iterations=max_samples + 10)
    )
    out = sampler.sample(torch.randn(n_chains, n_dim))

    assert torch.isfinite(out.samples).all()
    assert out.samples.shape == (max_samples, n_chains, n_dim)