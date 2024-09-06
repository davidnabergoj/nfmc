import pytest
import torch

from nfmc import sample
from nfmc.algorithms.sampling import HMC, NeuTraHMC, JumpHMC, AdaptiveIMH
from nfmc.util import get_supported_samplers
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian1


@pytest.mark.parametrize('sampler_class', [HMC, NeuTraHMC, JumpHMC, AdaptiveIMH])
def test_basic(sampler_class):
    torch.manual_seed(0)
    target = DiagonalGaussian1()
    sampler = sampler_class(target.event_shape, target)
    sampler.params.n_iterations = 3
    if isinstance(sampler, JumpHMC):
        sampler.inner_sampler.params.n_iterations = 3
    x0 = torch.randn(size=(100, *target.event_shape))
    output = sampler.sample(x0)

    assert output.statistics.running_first_moment.shape == target.event_shape
    assert output.statistics.running_second_moment.shape == target.event_shape

    assert output.statistics.running_first_moment.isfinite().all()
    assert output.statistics.running_second_moment.isfinite().all()


@pytest.mark.parametrize('strategy', get_supported_samplers())
def test_full(strategy: str):
    torch.manual_seed(0)
    event_shape = (10,)
    output = sample(
        target=lambda x: torch.sum(x ** 2, dim=1),
        negative_log_likelihood=lambda x: torch.sum(x ** 2, dim=1),
        event_shape=event_shape,
        strategy=strategy,
        n_iterations=3,
        n_warmup_iterations=3
    )

    assert output.mean.shape == event_shape
    assert output.second_moment.shape == event_shape
    assert output.variance.shape == event_shape
    assert output.mean.isfinite().all()
    assert output.second_moment.isfinite().all()
    assert output.variance.isfinite().all()
