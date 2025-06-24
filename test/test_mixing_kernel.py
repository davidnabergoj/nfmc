import pytest
import torch

from nfmc.algorithms.kernel import MixingKernel
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from test.util import StandardGaussian


@pytest.mark.parametrize('event_shape', [(1,), (4,), (2, 3)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, IMHKernel, MALAKernel])
@pytest.mark.parametrize('n_chains', [1, 4])
def test_step(event_shape, kernel_class, n_chains):
    torch.manual_seed(0)

    x_current = torch.randn(size=(n_chains, *event_shape))
    kernel1 = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    kernel2 = RWMHKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    mixing_kernel = MixingKernel(
        kernels=[kernel1, kernel2],
        selection_probabilities=[0.5, 0.5]
    )

    x_new = mixing_kernel.step(x_current)

    assert not x_new.requires_grad
    assert x_new.shape == x_current.shape
    assert torch.isfinite(x_new).all()
    assert x_current.dtype == x_new.dtype


def test_set_selection_probabilities_valid():
    torch.manual_seed(0)

    event_shape = (2, 3)

    x_current = torch.randn(size=(1, *event_shape))
    kernel1 = RWMHKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    kernel2 = HMCKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    mixing_kernel = MixingKernel(
        kernels=[kernel1, kernel2],
    )

    mixing_kernel.set_selection_probabilities([0.3, 0.7])

    x_new = mixing_kernel.step(x_current)

    assert not x_new.requires_grad
    assert x_new.shape == x_current.shape
    assert torch.isfinite(x_new).all()
    assert x_current.dtype == x_new.dtype


def test_set_selection_probabilities_invalid():
    torch.manual_seed(0)

    event_shape = (2, 3)

    kernel1 = RWMHKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    kernel2 = HMCKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    mixing_kernel = MixingKernel(
        kernels=[kernel1, kernel2],
    )

    with pytest.raises(ValueError):
        # Invalid probabilities (not summing to 1)
        mixing_kernel.set_selection_probabilities([0.3, 0.8])


def test_set_selection_probabilities_invalid_length():
    torch.manual_seed(0)

    event_shape = (2, 3)

    kernel1 = RWMHKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    kernel2 = HMCKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    mixing_kernel = MixingKernel(
        kernels=[kernel1, kernel2],
    )

    with pytest.raises(ValueError):
        mixing_kernel.set_selection_probabilities([0.3, 0.7, 0.1])
