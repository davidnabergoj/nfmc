import math
import pytest

import numpy as np
import torch

from nfmc.algorithms.mh.local.dual_averaging import DualAveraging
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.mh.base import MHSampler
from test.util import DiagonalGaussian


def test_step():
    initial_step_size = 1.0
    da = DualAveraging(initial_step_size=initial_step_size)
    da.step(0.2)

    assert np.isfinite(da.value)
    assert da.value > 0
    assert da.value != initial_step_size


def test_history():
    initial_step_size = 1.0
    da = DualAveraging(
        initial_step_size=initial_step_size,
        store_step_sizes=True,
        store_errors=True
    )

    for _ in range(10):
        da.step(0.001)

    assert len(da.step_size_history) == 10
    assert len(da.error_history) == 10

    for i in range(10):
        assert np.isfinite(da.step_size_history[i])
        assert da.step_size_history[i] > 0
        assert da.error_history[i] == 0.001

@pytest.mark.parametrize('kernel_class', [MALAKernel, RWMHKernel])
def test_reach_target_acceptance_rate(kernel_class):
    torch.manual_seed(0)
    target_acc_rate = 0.764321
    event_shape = (4,)

    target = DiagonalGaussian(event_shape)
    kernel = kernel_class(event_shape, target.neg_log_prob, target_acceptance_rate=target_acc_rate)
    sampler = MHSampler(kernel)

    warmup_samples = sampler.warmup(
        x0=torch.rand(size=(1, *event_shape)) * 2 - 1,
        n_steps=1000,
    )

    assert math.isclose(sampler.acceptance_rate, target_acc_rate, rel_tol=0.05)

@pytest.mark.parametrize('kernel_class', [MALAKernel, RWMHKernel])
def test_persist_step_size(kernel_class):
    torch.manual_seed(0)
    target_acc_rate = 0.764321
    event_shape = (4,)

    target = DiagonalGaussian(event_shape)
    kernel = kernel_class(event_shape, target.neg_log_prob, target_acceptance_rate=target_acc_rate)
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
    assert math.isclose(tuned_step_size, sampler.kernel.step_size)
