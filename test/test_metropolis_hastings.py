import pytest
import torch
from nfmc.algorithms.mh.base import MHSampler
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
from nfmc.algorithms.preconditioning.preconditioners import DenseLinearPreconditioner, DiagonalLinearPreconditioner, NormalizingFlowPreconditioner
from nfmc.algorithms.util.samples import Samples
from nfmc.util import create_flow_object
from test.util import StandardGaussian


@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, IMHKernel, MALAKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
def test_local_mh_kernel_step(event_shape, kernel_class, n_chains):
    torch.manual_seed(0)

    x_current = torch.randn(size=(n_chains, *event_shape))
    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    x_new = kernel.step(x_current)

    assert not x_new.requires_grad
    assert x_new.shape == x_current.shape
    assert torch.isfinite(x_new).all()
    assert x_current.dtype == x_new.dtype


@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, MALAKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
def test_local_mh_sampling(event_shape, kernel_class, n_chains, n_steps):
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


@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, MALAKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
@pytest.mark.parametrize('preconditioner_class', [
    DiagonalLinearPreconditioner,
    DenseLinearPreconditioner,
])
def test_linear_preconditioned_sampling(event_shape, kernel_class, n_chains, n_steps, preconditioner_class):
    torch.manual_seed(0)

    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob,
        preconditioner=preconditioner_class(event_shape)
    )
    sampler = PreconditionedMCMCSampler(kernel)

    z_initial = torch.randn(size=(n_chains, *event_shape))
    samples = sampler.sample(
        z_initial,
        n_steps=n_steps,
        show_progress=False
    )

    assert isinstance(samples, Samples)
    assert torch.isfinite(samples.as_tensor()).all()
    assert samples.as_tensor().shape == (n_steps, n_chains, *event_shape)
    assert samples.as_tensor().dtype == z_initial.dtype


@pytest.mark.parametrize('event_shape', [(2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, MALAKernel, IMHKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
def test_flow_preconditioned_sampling(event_shape, kernel_class, n_chains, n_steps):
    torch.manual_seed(0)

    flow = create_flow_object('realnvp', event_shape)

    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob,
        preconditioner=NormalizingFlowPreconditioner(flow)
    )
    sampler = PreconditionedMCMCSampler(kernel)

    z_initial = torch.randn(size=(n_chains, *event_shape))
    samples = sampler.sample(
        z_initial,
        n_steps=n_steps,
        show_progress=False
    )

    assert isinstance(samples, Samples)
    assert torch.isfinite(samples.as_tensor()).all()
    assert samples.as_tensor().shape == (n_steps, n_chains, *event_shape)
    assert samples.as_tensor().dtype == z_initial.dtype


@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, MALAKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
def test_local_mh_warmup(event_shape, kernel_class, n_chains, n_steps):
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
