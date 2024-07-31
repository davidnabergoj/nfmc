import torch
import pytest
from nfmc.algorithms.sampling.mcmc import (
    MH,
    RandomWalk,
    HMC,
    UHMC,
    NUTS,
    MALA,
    ULA,
    ESS
)
from nfmc.algorithms.sampling.nfmc import (
    JumpMALA,
    JumpMH,
    JumpHMC,
    JumpUHMC,
    JumpULA,
    JumpESS,
    JumpNUTS,
    TESS,
    DLMC,
    NeuTraHMC,
    FixedIMH,
    AdaptiveIMH
)
from potentials.synthetic.gaussian.unit import StandardGaussian
from nfmc.algorithms.base import MCMCOutput


@pytest.mark.parametrize('sampler_class', [
    MH,
    RandomWalk,
    HMC,
    UHMC,
    # NUTS,  # only supports one chain at the moment
    MALA,
    ULA
])
def test_mcmc(sampler_class):
    torch.manual_seed(0)
    n_iterations = 3
    n_chains = 4
    event_shape = (5,)
    sampler = sampler_class(event_shape=event_shape, target=StandardGaussian(n_dim=event_shape[0]))
    sampler.params.n_iterations = n_iterations
    x0 = torch.randn(size=(n_chains, *event_shape))
    output = sampler.sample(x0=x0, show_progress=False)

    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, *event_shape)
    assert torch.isfinite(output.samples).all()


def test_nuts():
    torch.manual_seed(0)
    n_iterations = 3
    n_chains = 1
    event_shape = (5,)
    sampler = NUTS(event_shape=event_shape, target=StandardGaussian(n_dim=event_shape[0]))
    sampler.params.n_iterations = n_iterations
    x0 = torch.randn(size=(n_chains, *event_shape))
    output = sampler.sample(x0=x0, show_progress=False)

    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, *event_shape)
    assert torch.isfinite(output.samples).all()


def test_ess():
    torch.manual_seed(0)
    n_iterations = 3
    n_chains = 4
    event_shape = (5,)
    sampler = ESS(
        event_shape=event_shape,
        target=StandardGaussian(n_dim=event_shape[0]),
        negative_log_likelihood=StandardGaussian(n_dim=event_shape[0])
    )
    sampler.params.n_iterations = n_iterations
    x0 = torch.randn(size=(n_chains, *event_shape))
    output = sampler.sample(x0=x0, show_progress=False)

    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, *event_shape)
    assert torch.isfinite(output.samples).all()


def test_jump_ess():
    torch.manual_seed(0)
    n_iterations = 3
    n_chains = 4
    event_shape = (5,)
    sampler = JumpESS(
        event_shape=event_shape,
        target=StandardGaussian(n_dim=event_shape[0]),
        negative_log_likelihood=StandardGaussian(n_dim=event_shape[0])
    )
    sampler.params.n_iterations = n_iterations
    x0 = torch.randn(size=(n_chains, *event_shape))
    output = sampler.sample(x0=x0, show_progress=False)

    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (
        n_iterations * (sampler.inner_sampler.params.n_iterations + 1), n_chains, *event_shape)
    assert torch.isfinite(output.samples).all()


@pytest.mark.parametrize('sampler_class', [TESS, DLMC])
def test_nfmc_with_nll(sampler_class):
    torch.manual_seed(0)
    n_iterations = 3
    n_chains = 4
    event_shape = (5,)
    sampler = sampler_class(
        event_shape=event_shape,
        target=StandardGaussian(n_dim=event_shape[0]),
        negative_log_likelihood=StandardGaussian(n_dim=event_shape[0])
    )
    sampler.params.n_iterations = n_iterations
    x0 = torch.randn(size=(n_chains, *event_shape))
    output = sampler.sample(x0=x0, show_progress=False)

    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (
        n_iterations,
        n_chains,
        *event_shape
    )
    assert torch.isfinite(output.samples).all()


@pytest.mark.parametrize('sampler_class', [
    JumpMALA,
    JumpMH,
    JumpHMC,
    JumpUHMC,
    JumpULA,
    # JumpNUTS,  # does not work with multiple chains
])
def test_jump_nfmc(sampler_class):
    torch.manual_seed(0)
    n_iterations = 3
    n_chains = 4
    event_shape = (5,)
    sampler = sampler_class(event_shape=event_shape, target=StandardGaussian(n_dim=event_shape[0]))
    sampler.params.n_iterations = n_iterations
    x0 = torch.randn(size=(n_chains, *event_shape))
    output = sampler.sample(x0=x0, show_progress=False)

    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (
        n_iterations * (sampler.inner_sampler.params.n_iterations + 1),
        n_chains,
        *event_shape
    )
    assert torch.isfinite(output.samples).all()


@pytest.mark.parametrize('sampler_class', [
    NeuTraHMC,
    FixedIMH,
    AdaptiveIMH
])
def test_other_nfmc(sampler_class):
    torch.manual_seed(0)
    n_iterations = 3
    n_chains = 4
    event_shape = (5,)
    sampler = sampler_class(
        event_shape=event_shape,
        target=StandardGaussian(n_dim=event_shape[0]),
    )
    sampler.params.n_iterations = n_iterations
    x0 = torch.randn(size=(n_chains, *event_shape))
    output = sampler.sample(x0=x0, show_progress=False)

    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (
        n_iterations,
        n_chains,
        *event_shape
    )
    assert torch.isfinite(output.samples).all()
