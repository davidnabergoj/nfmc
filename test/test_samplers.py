from nfmc.sampling import imh, ula, mala, tess, neutra
import torch
import pytest

from nfmc.util import get_supported_normalizing_flows
from synthetic.gaussian.unit import StandardGaussian

all_event_shapes = [(2, 3, 7), (2,), (5,)]
all_n_chains = [1, 5, 50]
all_n_iterations = [1, 5, 50]
all_flows = get_supported_normalizing_flows()


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_imh(event_shape, flow, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = imh(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_ula(event_shape, flow, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = ula(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_chains', all_n_chains)
@pytest.mark.parametrize('flow', all_flows)
@pytest.mark.parametrize('n_iterations', all_n_iterations)
def test_mala(event_shape, flow, n_chains, n_iterations):
    torch.manual_seed(0)
    target = StandardGaussian(event_shape)
    draws = mala(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
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
    draws = neutra(target, flow, n_chains, n_iterations)
    assert draws.shape == (n_iterations, n_chains, *event_shape)
    assert torch.all(~torch.isnan(draws))
    assert torch.all(torch.isfinite(draws))