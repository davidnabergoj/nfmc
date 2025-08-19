import math
import pytest

import torch

from nfmc.algorithms.mh.local.dual_averaging import DualAveraging
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.mh.base import MHSampler
from test.util import DiagonalGaussian


@pytest.mark.parametrize('n_chains', [1, 2, 10])
def test_step(n_chains):
    torch.manual_seed(0)
    target_acceptance_rate = 0.5

    initial_step_size = 1.0
    da = DualAveraging(
        initial_step_size=initial_step_size,
    )
    h = target_acceptance_rate - torch.rand(size=(n_chains,))
    da.step(h)

    assert torch.isfinite(da.value).all()
    assert (da.value > 0).all()
    assert (da.value != initial_step_size).all()
    assert da.value.shape == (n_chains,)


@pytest.mark.parametrize('n_chains', [1, 2, 10])
def test_history(n_chains):
    torch.manual_seed(0)

    initial_step_size = 1.0
    target_acc_rate = 0.5

    da = DualAveraging(
        initial_step_size=initial_step_size,
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


@pytest.mark.parametrize('kernel_class', [MALAKernel, RWMHKernel, HMCKernel])
def test_reach_target_acceptance_rate(kernel_class):
    torch.manual_seed(0)

    target_acc_rate = 0.764321
    event_shape = (4,)

    target = DiagonalGaussian(event_shape)
    kernel = kernel_class(
        event_shape,
        target.neg_log_prob,
        target_acceptance_rate=target_acc_rate
    )
    sampler = MHSampler(kernel)

    warmup_samples = sampler.warmup(
        x0=torch.rand(size=(1, *event_shape)) * 2 - 1,
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


@pytest.mark.parametrize('kernel_class', [MALAKernel, RWMHKernel])
def test_persist_step_size(kernel_class):
    torch.manual_seed(0)

    target_acc_rate = 0.764321
    event_shape = (4,)

    target = DiagonalGaussian(event_shape)
    kernel = kernel_class(
        event_shape,
        target.neg_log_prob,
        target_acceptance_rate=target_acc_rate
    )
    sampler = MHSampler(kernel)

    sampler.warmup(
        x0=torch.rand(size=(1, *event_shape)) * 2 - 1,
        n_steps=1000,
    )
    tuned_step_size = sampler.kernel.step_size

    sampler.sample(
        x0=torch.rand(size=(1, *event_shape)) * 2 - 1,
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
