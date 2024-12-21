import pytest
import torch

from nfmc.algorithms.sampling.mcmc.ess import ESS
from nfmc.algorithms.sampling.mcmc.hmc import UHMC, HMC
from nfmc.algorithms.sampling.mcmc.langevin import MALA, ULA
from nfmc.algorithms.sampling.mcmc.mh import MH, RandomWalk
from nfmc.algorithms.sampling.nfmc.imh import FixedIMH, AdaptiveIMH
from nfmc.algorithms.sampling.nfmc.jump import JumpESS, JumpMALA, JumpUHMC, JumpHMC, JumpULA, JumpMH
from nfmc.algorithms.sampling.nfmc.neutra import NeuTraHMC
from test.util import standard_gaussian_potential


@pytest.mark.parametrize('sampler_class', [
    # NUTS,  # not supported yet
    MALA,
    MH,
    UHMC,
    HMC,
    ULA,
    RandomWalk,
])
def test_warmup_mcmc(sampler_class):
    torch.manual_seed(0)
    n_iterations = 7
    n_dim = 5
    n_chains = 3

    sampler = sampler_class(event_shape=(n_dim,), target=standard_gaussian_potential)
    sampler.params.n_warmup_iterations = n_iterations

    x0 = torch.randn(size=(n_chains, n_dim))
    warmup_output = sampler.warmup(x0, show_progress=False)
    assert warmup_output.samples.shape == (n_iterations, n_chains, n_dim)
    assert torch.isfinite(warmup_output.samples).all()


@pytest.mark.skip("Not implemented")
def test_warmup_ess():
    torch.manual_seed(0)
    n_iterations = 7
    n_dim = 5
    n_chains = 3

    sampler = ESS(event_shape=(n_dim,), target=standard_gaussian_potential, negative_log_likelihood=standard_gaussian_potential)
    sampler.params.n_warmup_iterations = n_iterations

    x0 = torch.randn(size=(n_chains, n_dim))
    warmup_output = sampler.warmup(x0, show_progress=False)
    assert warmup_output.samples.shape == (n_iterations, n_chains, n_dim)
    assert torch.isfinite(warmup_output.samples).all()


@pytest.mark.parametrize('sampler_class', [
    # JumpNUTS,  # not supported yet
    JumpMH,
    JumpULA,
    JumpHMC,
    JumpUHMC,
    JumpMALA,
])
def test_warmup_jump_nfmc(sampler_class):
    torch.manual_seed(0)
    n_dim = 5
    n_chains = 3

    sampler = sampler_class(event_shape=(n_dim,), target=standard_gaussian_potential)

    x0 = torch.randn(size=(n_chains, n_dim))
    warmup_output = sampler.warmup(x0, show_progress=False)
    assert warmup_output.samples.shape[1:] == (n_chains, n_dim)
    assert len(warmup_output.samples.shape) == 3
    assert torch.isfinite(warmup_output.samples).all()


@pytest.mark.skip("Not implemented")
def test_warmup_jump_ess():
    torch.manual_seed(0)
    n_dim = 5
    n_chains = 3

    sampler = JumpESS(
        event_shape=(n_dim,),
        target=standard_gaussian_potential,
        negative_log_likelihood=standard_gaussian_potential
    )
    x0 = torch.randn(size=(n_chains, n_dim))
    warmup_output = sampler.warmup(x0, show_progress=False)
    assert warmup_output.samples.shape == (1, n_chains, n_dim)
    assert torch.isfinite(warmup_output.samples).all()


@pytest.mark.parametrize('sampler_class', [
    AdaptiveIMH,
    FixedIMH,
])
def test_warmup_imh(sampler_class):
    torch.manual_seed(0)
    n_dim = 5
    n_chains = 3

    sampler = sampler_class(event_shape=(n_dim,), target=standard_gaussian_potential)

    x0 = torch.randn(size=(n_chains, n_dim))
    warmup_output = sampler.warmup(x0, show_progress=False)
    assert warmup_output.samples.shape == (1, n_chains, n_dim)
    assert torch.isfinite(warmup_output.samples).all()


def test_warmup_neutra():
    torch.manual_seed(0)
    n_dim = 5
    n_chains = 3

    sampler = NeuTraHMC(event_shape=(n_dim,), target=standard_gaussian_potential)

    x0 = torch.randn(size=(n_chains, n_dim))
    warmup_output = sampler.warmup(x0, show_progress=False)
    assert warmup_output.samples.shape == (sampler.inner_sampler.params.n_warmup_iterations, n_chains, n_dim)
    assert torch.isfinite(warmup_output.samples).all()
