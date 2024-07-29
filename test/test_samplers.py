from nfmc import sample
from normalizing_flows.flows import Flow
from normalizing_flows.architectures import RealNVP
from potentials.synthetic.gaussian.unit import StandardGaussian
import pytest
from nfmc.util import MCMCOutput
import torch


@pytest.mark.parametrize('strategy', ['hmc', 'uhmc', 'ula', 'mala', 'mh'])
def test_mcmc(strategy):
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = (5, 7, 10)
    target = StandardGaussian(n_dim)
    output = sample(target, strategy=strategy, n_iterations=n_iterations, n_chains=n_chains)
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, n_dim)


@pytest.mark.parametrize('strategy', ["imh", "fixed_imh"])
def test_imh(strategy):
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = (5, 7, 10)
    target = StandardGaussian(n_dim)
    flow = Flow(RealNVP(target.event_shape))
    output = sample(target, flow=flow, strategy=strategy, n_iterations=n_iterations, n_chains=n_chains)
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, n_dim)


@pytest.mark.parametrize('strategy', ["jump_mala", "jump_ula", "jhmc", "fixed_jhmc", "jump_uhmc"])
def test_nfmc_jumps(strategy):
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = (5, 7, 10)
    n_trajectories_per_jump = 3
    target = StandardGaussian(n_dim)
    flow = Flow(RealNVP(target.event_shape))
    output = sample(
        target,
        flow=flow,
        strategy=strategy,
        n_iterations=n_iterations,
        n_trajectories_per_jump=n_trajectories_per_jump,
        n_chains=n_chains
    )
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations * (n_trajectories_per_jump + 1), n_chains, n_dim)


@pytest.mark.parametrize('strategy', ["neutra_hmc", "tess"])
def test_nfmc_preconditioning(strategy):
    torch.manual_seed(0)
    n_iterations, n_chains, n_dim = (5, 7, 10)
    target = StandardGaussian(n_dim)
    flow = Flow(RealNVP(target.event_shape))
    output = sample(
        target,
        flow=flow,
        strategy=strategy,
        n_iterations=n_iterations,
        n_chains=n_chains
    )
    assert isinstance(output, MCMCOutput)
    assert output.samples.shape == (n_iterations, n_chains, n_dim)
