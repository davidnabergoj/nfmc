import pytest
import torch

from nfmc.algorithms.sampling import HMC, NeuTraHMC, JumpHMC, AdaptiveIMH
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian1


@pytest.mark.parametrize('sampler_class', [HMC, NeuTraHMC, JumpHMC, AdaptiveIMH])
def test_basic(sampler_class):
    torch.manual_seed(0)
    target = DiagonalGaussian1()
    sampler = sampler_class(target.event_shape, target)
    sampler.params.estimate_running_moments = True
    sampler.params.n_iterations = 3
    if isinstance(sampler, JumpHMC):
        sampler.inner_sampler.params.n_iterations = 3
    x0 = torch.randn(size=(100, *target.event_shape))
    output = sampler.sample(x0)

    assert output.statistics.running_first_moment.shape == target.event_shape
    assert output.statistics.running_second_moment.shape == target.event_shape

    assert output.statistics.running_first_moment.isfinite().all()
    assert output.statistics.running_second_moment.isfinite().all()

    print(output.statistics.running_first_moment)
    print(output.statistics.running_second_moment)
