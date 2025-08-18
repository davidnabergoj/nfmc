from nfmc.algorithms.mh.base import MHSampler
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.nuts import NUTSKernel
from nfmc.algorithms.util.samples import Samples
from test.util import DiagonalGaussian, StandardGaussian


import pytest
import torch


@pytest.mark.parametrize('event_shape', [(1,), (4,), (2, 3)])
@pytest.mark.parametrize('kernel_class', [
    RWMHKernel,
    HMCKernel,
    IMHKernel,  # add IMH
    MALAKernel,
    NUTSKernel
])
@pytest.mark.parametrize('n_chains', [1, 4])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_step(event_shape, kernel_class, n_chains, dtype):
    torch.manual_seed(0)

    x_current = torch.randn(size=(n_chains, *event_shape), dtype=dtype)
    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    x_new = kernel.step(x_current)

    assert x_new is not x_current
    assert not x_new.requires_grad
    assert x_new.shape == x_current.shape
    assert torch.isfinite(x_new).all()
    assert x_current.dtype == x_new.dtype


@pytest.mark.local_only
@pytest.mark.parametrize('event_shape', [(1,), (4,), (2, 3)])
@pytest.mark.parametrize('kernel_class', [
    RWMHKernel,
    HMCKernel,
    MALAKernel,
    NUTSKernel
])
@pytest.mark.parametrize('n_chains', [1, 4])
@pytest.mark.parametrize('n_steps', [1, 4])
def test_sample(event_shape, kernel_class, n_chains, n_steps):
    torch.manual_seed(0)

    x_initial = torch.randn(size=(n_chains, *event_shape))
    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    sampler = MHSampler(kernel)
    samples = sampler.sample(
        x_initial,
        n_steps=n_steps,
        show_progress=False
    )

    assert isinstance(samples, Samples)
    assert torch.isfinite(samples.as_tensor()).all()
    assert samples.as_tensor().shape == (n_steps, n_chains, *event_shape)
    assert samples.as_tensor().dtype == x_initial.dtype


@pytest.mark.local_only
@pytest.mark.parametrize('event_shape', [(1,), (4,), (2, 3)])
@pytest.mark.parametrize('kernel_class', [
    RWMHKernel,
    HMCKernel,
    MALAKernel,
    NUTSKernel
])
@pytest.mark.parametrize('n_chains', [1, 4])
@pytest.mark.parametrize('n_steps', [1, 4])
def test_warmup(event_shape, kernel_class, n_chains, n_steps):
    torch.manual_seed(0)

    x_initial = torch.randn(size=(n_chains, *event_shape))
    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    sampler = MHSampler(kernel)
    samples = sampler.warmup(
        x_initial,
        n_steps=n_steps,
        show_progress=False
    )

    assert isinstance(samples, Samples)
    assert torch.isfinite(samples.as_tensor()).all()
    assert samples.as_tensor().shape == (n_steps, n_chains, *event_shape)
    assert samples.as_tensor().dtype == x_initial.dtype


@pytest.mark.local_only
@pytest.mark.parametrize("kernel_class", [
    RWMHKernel,
    MALAKernel,
    HMCKernel,
    NUTSKernel
])
def test_warmup_and_sample(kernel_class):
    torch.manual_seed(0)

    event_shape = (4,)
    target = DiagonalGaussian(event_shape)

    kernel = kernel_class(event_shape, neg_log_prob_target=target.neg_log_prob)
    sampler = MHSampler(kernel)

    x0 = torch.rand(size=(1, *event_shape)) * 2 - 1
    warmup_draws = sampler.warmup(
        x0=x0,
        n_steps=500 if kernel_class == HMCKernel else 2000
    )
    sampling_draws = sampler.sample(
        x0=warmup_draws.last_sample, n_steps=2000
    )

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
