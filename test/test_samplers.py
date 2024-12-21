import torch
import pytest

from nfmc import sample
from nfmc.algorithms.sampling.base import MCMCOutput
from nfmc.algorithms.sampling.mcmc.ess import ESS
from nfmc.algorithms.sampling.mcmc.hmc import HMC, UHMC
from nfmc.algorithms.sampling.mcmc.langevin import MALA, ULA
from nfmc.algorithms.sampling.mcmc.mh import MH, RandomWalk
from nfmc.algorithms.sampling.mcmc.nuts import NUTS
from nfmc.algorithms.sampling.nfmc.dlmc import DLMC
from nfmc.algorithms.sampling.nfmc.imh import FixedIMH, AdaptiveIMH
from nfmc.algorithms.sampling.nfmc.jump import JumpESS, JumpMALA, JumpMH, JumpHMC, JumpUHMC, JumpULA
from nfmc.algorithms.sampling.nfmc.neutra import NeuTraHMC
from nfmc.algorithms.sampling.nfmc.tess import TESS
from test.util import standard_gaussian_potential


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
    sampler = sampler_class(event_shape=event_shape, target=standard_gaussian_potential)
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
    sampler = NUTS(event_shape=event_shape, target=standard_gaussian_potential)
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
        target=standard_gaussian_potential,
        negative_log_likelihood=standard_gaussian_potential
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
        target=standard_gaussian_potential,
        negative_log_likelihood=standard_gaussian_potential
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
        target=standard_gaussian_potential,
        negative_log_likelihood=standard_gaussian_potential
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
    sampler = sampler_class(event_shape=event_shape, target=standard_gaussian_potential)
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
        target=standard_gaussian_potential,
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


@pytest.mark.parametrize('strategy', [
    'hmc',
    'uhmc',
    'ula',
    'mala',
    'mh',
    "imh",
    "neutra_hmc"
])
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_sample_wrapper_no_jump(strategy: str, device: str):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = 3, 4, 5

    target = standard_gaussian_potential
    output = sample(
        target,
        event_shape=(n_dim,),
        strategy=strategy,
        n_chains=n_chains,
        n_iterations=n_iterations,
        device=torch.device(device),
    )
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, n_dim)
    assert torch.isfinite(output.samples).all()


@pytest.mark.parametrize('strategy', ['dlmc', 'tess', "ess"])
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_sample_wrapper_nll(strategy: str, device: str):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = 3, 4, 5

    target = standard_gaussian_potential
    output = sample(
        target,
        event_shape=(n_dim,),
        strategy=strategy,
        negative_log_likelihood=standard_gaussian_potential,
        n_chains=n_chains,
        n_iterations=n_iterations,
        device=torch.device(device)
    )
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, n_dim)
    assert torch.isfinite(output.samples).all()


@pytest.mark.parametrize('strategy', ["jump_mala", "jump_ula", "jump_hmc", "jump_uhmc", "jump_mh"])
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_sample_wrapper_jump(strategy: str, device: str):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = 3, 4, 5
    n_trajectories_per_jump = 7

    target = standard_gaussian_potential
    output = sample(
        target,
        event_shape=(n_dim,),
        strategy=strategy,
        n_chains=n_chains,
        n_iterations=n_iterations,
        inner_param_kwargs={'n_iterations': n_trajectories_per_jump},
        device=torch.device(device)
    )
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations * (n_trajectories_per_jump + 1), n_chains, n_dim)
    assert torch.isfinite(output.samples).all()


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_sample_wrapper_jump_ess(device: str):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = 3, 4, 5
    n_trajectories_per_jump = 7

    target = standard_gaussian_potential
    output = sample(
        target,
        event_shape=(n_dim,),
        strategy='jump_ess',
        n_chains=n_chains,
        n_iterations=n_iterations,
        negative_log_likelihood=standard_gaussian_potential,
        inner_param_kwargs={'n_iterations': n_trajectories_per_jump},
        device=torch.device(device)
    )
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations * (n_trajectories_per_jump + 1), n_chains, n_dim)
    assert torch.isfinite(output.samples).all()
