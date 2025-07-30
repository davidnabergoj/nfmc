from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from nfmc.algorithms.mh.base import MHSampler
from nfmc.algorithms.mh.imh import IMHKernel
from test.util import DiagonalGaussian, StandardGaussian


import pytest
import torch


@pytest.mark.local_only
def test_imh_sampling():
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 10
    target = DiagonalGaussian(event_shape, mu=0.5, std=0.5)
    kernel = IMHKernel(event_shape, target.neg_log_prob)
    sampler = MHSampler(kernel)

    x0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    sampling_draws = sampler.sample(x0, n_steps=2000)

    assert torch.allclose(
        sampling_draws.first_moment.as_tensor(),
        target.first_moment,
        rtol=0.2
    )
    assert torch.allclose(
        sampling_draws.second_moment.as_tensor(),
        target.second_moment,
        rtol=0.2
    )


@pytest.mark.local_only
def test_isir_sampling():
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 10
    target = DiagonalGaussian(event_shape, mu=0.5, std=0.5)
    kernel = IteratedSIRKernel(event_shape, target.neg_log_prob)
    sampler = MHSampler(kernel)

    x0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    sampling_draws = sampler.sample(x0, n_steps=2000)

    assert torch.allclose(
        sampling_draws.first_moment.as_tensor(),
        target.first_moment,
        rtol=0.2
    )
    assert torch.allclose(
        sampling_draws.second_moment.as_tensor(),
        target.second_moment,
        rtol=0.2
    )


def test_isir_step():
    torch.manual_seed(0)

    event_shape = (2, 3, 4)
    n_chains = 5

    x0 = torch.zeros(size=(n_chains, *event_shape))
    kernel = IteratedSIRKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    x_new = kernel.step(x0)

    assert x_new is not x0
    assert x_new.shape == x0.shape
    assert x_new.dtype == x0.dtype
    assert torch.isfinite(x_new).all()
