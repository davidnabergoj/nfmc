from nfmc.transport import ns, snf, aft, dlmc, craft

import torch
import pytest

from nfmc.util import get_supported_normalizing_flows
from synthetic.gaussian.unit import StandardGaussian

all_event_shapes = [(2,), (5,), (2, 3, 7)]
all_n_particles = [1, 5, 50]
all_n_iterations = [5, 1, 50]
all_flows = get_supported_normalizing_flows()
all_jump_periods = [10, 50]


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_particles', all_n_particles)
@pytest.mark.parametrize('flow', all_flows)
def test_dlmc(event_shape, n_particles, flow):
    torch.manual_seed(0)
    prior = StandardGaussian(event_shape)
    target = StandardGaussian(event_shape)
    particle_history = dlmc(prior, target, flow, n_particles)
    assert particle_history.shape[1:] == (n_particles, *event_shape)
    assert torch.all(~torch.isnan(particle_history))
    assert torch.all(torch.isfinite(particle_history))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_particles', all_n_particles)
@pytest.mark.parametrize('flow', all_flows)
def test_aft(event_shape, n_particles, flow):
    torch.manual_seed(0)
    prior = StandardGaussian(event_shape)
    target = StandardGaussian(event_shape)
    particle_history = aft(prior, target, flow, n_particles)
    assert particle_history.shape[1:] == (n_particles, *event_shape)
    assert torch.all(~torch.isnan(particle_history))
    assert torch.all(torch.isfinite(particle_history))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_particles', all_n_particles)
@pytest.mark.parametrize('flow', all_flows)
def test_craft(event_shape, n_particles, flow):
    torch.manual_seed(0)
    prior = StandardGaussian(event_shape)
    target = StandardGaussian(event_shape)
    particle_history = craft(prior, target, flow, n_particles)
    assert particle_history.shape[1:] == (n_particles, *event_shape)
    assert torch.all(~torch.isnan(particle_history))
    assert torch.all(torch.isfinite(particle_history))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_particles', all_n_particles)
@pytest.mark.parametrize('flow', all_flows)
def test_snf(event_shape, n_particles, flow):
    torch.manual_seed(0)
    prior = StandardGaussian(event_shape)
    target = StandardGaussian(event_shape)
    particle_history = snf(prior, target, flow, n_particles)
    assert particle_history.shape[1:] == (n_particles, *event_shape)
    assert torch.all(~torch.isnan(particle_history))
    assert torch.all(torch.isfinite(particle_history))


@pytest.mark.parametrize('event_shape', all_event_shapes)
@pytest.mark.parametrize('n_particles', all_n_particles)
@pytest.mark.parametrize('flow', all_flows)
def test_ns(event_shape, n_particles, flow):
    torch.manual_seed(0)
    prior = StandardGaussian(event_shape)
    target = StandardGaussian(event_shape)
    particle_history = ns(prior, target, flow, n_particles)
    assert particle_history.shape[1:] == (n_particles, *event_shape)
    assert torch.all(~torch.isnan(particle_history))
    assert torch.all(torch.isfinite(particle_history))
