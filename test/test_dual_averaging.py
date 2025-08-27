import math
import pytest

import torch

from nfmc.algorithms.jump.kernels import DiagonalJumpRWMHKernel
from nfmc.algorithms.mh.local.dual_averaging import DualAveraging
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.mh.base import MHSampler
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
from test.util import DiagonalGaussian

from torchflows.bijections.finite.autoregressive.layers import ElementwiseAffine
from torchflows.flows import Flow


@pytest.mark.parametrize('kernel_class', [
    RWMHKernel,
    DiagonalJumpRWMHKernel
])
def test_constructed_during_warmup(kernel_class):
    torch.manual_seed(0)
    event_shape = (4,)
    n_chains = 3

    if kernel_class == RWMHKernel:
        kernel = kernel_class(
            event_shape=event_shape,
            neg_log_prob_target=lambda v: torch.sum(v ** 2, dim=-1)
        )
    else:
        flow = Flow(ElementwiseAffine(event_shape=event_shape))
        kernel = kernel_class(
            flow=flow,
            neg_log_prob_target=lambda v: torch.sum(v ** 2, dim=-1)
        )
    
    sampler = PreconditionedMCMCSampler(kernel=kernel)
    sampler.warmup(
        z0=torch.rand(size=(n_chains, *event_shape)) * 2 - 1,
        n_cycles=1,
        cycle_length=1
    )

    if kernel_class == RWMHKernel:
        assert kernel._dual_averaging is not None
        assert isinstance(kernel._dual_averaging, DualAveraging)
    else:
        assert kernel.kernels[0]._dual_averaging is not None
        assert isinstance(kernel.kernels[0]._dual_averaging, DualAveraging)
        
        assert not hasattr(kernel.kernels[1], '_dual_averaging')


@pytest.mark.parametrize('n_chains', [1, 2, 10])
def test_step(n_chains):
    torch.manual_seed(0)
    target_acceptance_rate = 0.5

    initial_step_size = 1.0
    da = DualAveraging(
        initial_step_size=initial_step_size,
        n_chains=n_chains
    )
    h = target_acceptance_rate - torch.rand(size=(n_chains,))
    da.step(h)

    assert torch.isfinite(da.value).all()
    assert (da.value > 0).all()
    assert (da.value != initial_step_size).all()
    assert da.value.shape == (n_chains,)


@pytest.mark.parametrize('n_chains', [2, 4, 5])
@pytest.mark.parametrize('n_updated_chains', [2])
@pytest.mark.parametrize('store_history', [True, False])
def test_step_update_few(n_chains, n_updated_chains, store_history):
    torch.manual_seed(0)
    target_acceptance_rate = 0.5

    initial_step_size = 1.0
    da = DualAveraging(
        initial_step_size=initial_step_size,
        n_chains=n_chains,
        store_errors=store_history,
        store_step_sizes=store_history
    )
    h = target_acceptance_rate - torch.rand(size=(n_updated_chains,))
    update_mask = torch.arange(n_chains) < n_updated_chains
    da.step(h, mask=update_mask)

    assert torch.isfinite(da.value).all()
    assert (da.value > 0).all()
    assert (da.value[update_mask] != initial_step_size).all()
    assert da.value.shape == (n_chains,)


@pytest.mark.parametrize('n_chains', [1, 2, 10])
def test_history(n_chains):
    torch.manual_seed(0)

    initial_step_size = 1.0
    target_acc_rate = 0.5

    da = DualAveraging(
        initial_step_size=initial_step_size,
        n_chains=n_chains,
        store_step_sizes=True,
        store_errors=True,
    )

    n_steps = 50

    for _ in range(n_steps):
        h = target_acc_rate - torch.rand(size=(n_chains,))
        da.step(h)

    assert len(da.step_size_history) == n_steps
    assert len(da.error_history) == n_steps

    for i in range(n_steps):
        assert torch.isfinite(da.step_size_history[i]).all()
        assert (da.step_size_history[i] > 0).all()
        assert da.step_size_history[i].shape == (n_chains,)
        if n_chains > 2:
            assert len(torch.unique(da.step_size_history[i])) > 1


@pytest.mark.local_only
@pytest.mark.parametrize('kernel_class', [MALAKernel, RWMHKernel, HMCKernel])
def test_reach_target_acceptance_rate(kernel_class):
    torch.manual_seed(0)

    target_acc_rate = 0.764321
    event_shape = (4,)
    n_chains = 1

    target = DiagonalGaussian(event_shape)
    kernel = kernel_class(
        event_shape,
        neg_log_prob_target=target.neg_log_prob,
        target_acceptance_rate=target_acc_rate
    )
    sampler = MHSampler(kernel)

    warmup_samples = sampler.warmup(
        x0=torch.rand(size=(n_chains, *event_shape)) * 2 - 1,
        n_steps=1000,
    )

    assert math.isclose(
        sampler.kernel.acceptance_rate,
        target_acc_rate,
        rel_tol=0.05
    )

    assert isinstance(
        sampler.kernel.step_size,
        torch.Tensor
    )


@pytest.mark.local_only
@pytest.mark.parametrize('kernel_class', [MALAKernel, RWMHKernel])
def test_persist_step_size(kernel_class):
    torch.manual_seed(0)

    target_acc_rate = 0.764321
    event_shape = (4,)
    n_chains = 1

    target = DiagonalGaussian(event_shape)
    kernel = kernel_class(
        event_shape,
        neg_log_prob_target=target.neg_log_prob,
        target_acceptance_rate=target_acc_rate
    )
    sampler = MHSampler(kernel)

    sampler.warmup(
        x0=torch.rand(size=(n_chains, *event_shape)) * 2 - 1,
        n_steps=1000,
    )
    tuned_step_size = sampler.kernel.step_size

    sampler.sample(
        x0=torch.rand(size=(n_chains, *event_shape)) * 2 - 1,
        n_steps=3
    )
    assert math.isclose(
        tuned_step_size,
        sampler.kernel.step_size
    )

    assert isinstance(
        sampler.kernel.step_size,
        torch.Tensor
    )
