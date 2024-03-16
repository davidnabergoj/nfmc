from nfmc.sampling_implementations import nf_imh, nf_ula, nf_mala, tess, neutra_hmc, dlmc
import torch
import pytest

from nfmc.util import get_supported_normalizing_flows
from potentials.synthetic.gaussian.unit import StandardGaussian

all_event_shapes = [(2,), (5,), (2, 3, 7)]
all_n_chains = [1, 5, 50]
all_n_iterations = [5, 1, 50]
all_flows = get_supported_normalizing_flows()
all_jump_periods = [10, 50]


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_imh(event_shape, flow, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = nf_imh(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
@pytest.mark.parametrize('jump_period', all_jump_periods)
def test_ula(event_shape, flow, n_chains, n_iterations, jump_period):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = nf_ula(target, flow, n_chains, n_iterations, jump_period)
    assert draws.shape == (n_iterations * jump_period, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
@pytest.mark.parametrize('jump_period', all_jump_periods)
def test_mala(event_shape, flow, n_chains, n_iterations, jump_period):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = nf_mala(target, flow, n_chains, n_iterations, jump_period)
    assert draws.shape == (n_iterations * jump_period, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_tess(event_shape, flow, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = tess(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_neutra(event_shape, flow, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = neutra_hmc(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_dlmc(event_shape, n_chains, flow, n_iterations):
    torch.manual_seed(0)
    prior = StandardGaussian(event_shape)
    target = StandardGaussian(event_shape)
    draws = dlmc(prior, target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))
