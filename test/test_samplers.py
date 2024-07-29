from nfmc.sampling_implementations import nf_imh, nf_ula, nf_mala, tess, neutra_hmc, dlmc
import torch
import pytest
from potentials.synthetic.gaussian.unit import StandardGaussian
from normalizing_flows.flows import Flow
from normalizing_flows.architectures import RealNVP

all_event_shapes = [(2,), (3,), (7,)]
all_n_chains = [1, 5]
all_n_iterations = [1, 5]
all_jump_periods = [1, 5]


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_imh(event_shape, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(*event_shape)
    flow = Flow(RealNVP(target.event_shape))
    draws = nf_imh(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
@pytest.mark.parametrize('jump_period', all_jump_periods)
def test_ula(event_shape, n_chains, n_iterations, jump_period):
    torch.manual_seed(0)
    target = StandardGaussian(*event_shape)
    flow = Flow(RealNVP(target.event_shape))
    draws = nf_ula(target, flow, n_chains, n_iterations, jump_period)
    assert draws.shape == (n_iterations * jump_period, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
@pytest.mark.parametrize('jump_period', all_jump_periods)
def test_mala(event_shape, n_chains, n_iterations, jump_period):
    torch.manual_seed(0)
    target = StandardGaussian(*event_shape)
    flow = Flow(RealNVP(target.event_shape))
    draws = nf_mala(target, flow, n_chains, n_iterations, jump_period)
    assert draws.shape == (n_iterations * jump_period, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_tess(event_shape, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(*event_shape)
    flow = Flow(RealNVP(target.event_shape))
    draws = tess(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_neutra(event_shape, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(*event_shape)
    flow = Flow(RealNVP(target.event_shape))
    draws = neutra_hmc(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_dlmc(event_shape, n_chains, n_iterations):
    torch.manual_seed(0)
    prior = StandardGaussian(*event_shape)
    target = StandardGaussian(*event_shape)
    flow = Flow(RealNVP(target.event_shape))
    draws = dlmc(prior, target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))
