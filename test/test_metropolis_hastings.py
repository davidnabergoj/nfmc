import pytest
import torch
from nfmc.algorithms.mh.base import MHSampler
from nfmc.algorithms.mh.rwmh import RWMHKernel
from nfmc.algorithms.mh.hmc import HMCKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.mala import MALAKernel
from nfmc.algorithms.util.samples import Samples
from test.util import standard_gaussian_neg_log_prob


@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, IMHKernel, MALAKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
def test_kernel_step(event_shape, kernel_class, n_chains):
    torch.manual_seed(0)

    x_current = torch.randn(size=(n_chains, *event_shape))
    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=standard_gaussian_neg_log_prob
    )
    x_prime, acceptance_mask = kernel.step(x_current)

    assert not x_prime.requires_grad
    assert x_prime.shape == x_current.shape
    assert acceptance_mask.shape == (n_chains,)
    assert torch.isfinite(x_prime).all()
    assert torch.isfinite(acceptance_mask).all()
    assert acceptance_mask.dtype == torch.bool
    assert x_current.dtype == x_prime.dtype


@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, IMHKernel, MALAKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
def test_sampling(event_shape, kernel_class, n_chains, n_steps):
    torch.manual_seed(0)

    x_initial = torch.randn(size=(n_chains, *event_shape))
    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=standard_gaussian_neg_log_prob
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
