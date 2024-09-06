import pytest
import torch
from nfmc.sample import create_sampler
from nfmc.util import get_supported_samplers


@pytest.mark.parametrize('strategy', get_supported_samplers())
def test_warmup(strategy: str):
    if 'jump' in strategy:
        return
    torch.manual_seed(0)
    n_chains = 20
    event_shape = (10,)
    sampler = create_sampler(
        target=lambda x: torch.sum(x ** 2, dim=1),
        event_shape=event_shape,
        strategy=strategy,
        param_kwargs={
            'store_samples': False,
        },
        negative_log_likelihood=lambda x: torch.sum(x ** 2, dim=1),
    )
    x0 = torch.randn(size=(n_chains, *event_shape))
    out = sampler.warmup(x0, time_limit_seconds=1.0)
    assert out.samples is None
    assert out.running_samples.last_sample is not None
    assert out.running_samples.last_sample.shape == (n_chains, *event_shape)


@pytest.mark.parametrize('strategy', get_supported_samplers())
def test_sampling(strategy: str):
    torch.manual_seed(0)
    n_chains = 20
    event_shape = (10,)
    sampler = create_sampler(
        target=lambda x: torch.sum(x ** 2, dim=1),
        event_shape=event_shape,
        strategy=strategy,
        param_kwargs={
            'store_samples': False,
        },
        negative_log_likelihood=lambda x: torch.sum(x ** 2, dim=1),
    )
    x0 = torch.randn(size=(n_chains, *event_shape))
    out = sampler.sample(x0, time_limit_seconds=1.0)
    assert out.samples is None
    assert out.running_samples.last_sample is not None
    assert out.running_samples.last_sample.shape == (n_chains, *event_shape)
