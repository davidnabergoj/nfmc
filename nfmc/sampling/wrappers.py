import torch

from nfmc.sampling.jump.hamiltonian_monte_carlo import adjusted_hmc_base, unadjusted_hmc_base
from potentials.base import Potential
from potentials.synthetic.gaussian.unit import StandardGaussian
from nfmc.sampling.neutra import neutra_hmc_base
from nfmc.sampling.elliptical import transport_elliptical_slice_sampler, elliptical_slice_sampler
from nfmc.sampling.independent_metropolis_hastings import independent_metropolis_hastings_base
from nfmc.sampling.jump.langevin_monte_carlo import metropolis_adjusted_langevin_algorithm_base, \
    unadjusted_langevin_algorithm_base
from nfmc.sampling.deterministic_langevin import deterministic_langevin_monte_carlo_base
from nfmc.util import create_flow_object


def nf_hmc(target: Potential,
           flow: str,
           n_chains: int = 100,
           n_iterations: int = 100,
           n_mcmc_steps_per_iteration: int = 50,
           x0: torch.Tensor = None,
           **kwargs):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *target.event_shape))
    return adjusted_hmc_base(
        x0,
        flow_object,
        target,
        n_jumps=n_iterations,
        jump_period=n_mcmc_steps_per_iteration,
        **kwargs
    )


def nf_uhmc(target: Potential,
            flow: str,
            n_chains: int = 100,
            n_iterations: int = 100,
            n_mcmc_steps_per_iteration: int = 50,
            x0: torch.Tensor = None,
            **kwargs):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *target.event_shape))
    return unadjusted_hmc_base(
        x0,
        flow_object,
        target,
        n_jumps=n_iterations,
        n_mcmc_steps_per_jump=n_mcmc_steps_per_iteration,
        **kwargs
    )


def nf_mala(target: Potential, flow: str, n_chains: int = 100, n_iterations: int = 100, jump_period: int = 50,
            x0: torch.Tensor = None):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *target.event_shape))
    return metropolis_adjusted_langevin_algorithm_base(
        x0,
        flow_object,
        target,
        n_jumps=n_iterations,
        jump_period=jump_period
    )


def nf_ula(target: Potential,
           flow: str,
           n_chains: int = 100,
           n_iterations: int = 100,
           jump_period: int = 50,
           x0: torch.Tensor = None,
           **kwargs):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *target.event_shape))
    return unadjusted_langevin_algorithm_base(
        x0,
        flow_object,
        target,
        n_jumps=n_iterations,
        jump_period=jump_period,
        **kwargs
    )


def nf_imh(target: Potential,
           flow: str,
           n_chains: int = 100,
           n_iterations: int = 1000,
           x0: torch.Tensor = None):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *target.event_shape))
    else:
        # n_chains = x0.shape[0]
        assert x0.shape[1:] == target.event_shape
    return independent_metropolis_hastings_base(x0, flow_object, target, n_iterations=n_iterations)


def neutra_hmc(target: callable, flow: str, event_shape, n_chains: int = 100, n_iterations: int = 1000):
    """

    :param target: target potential.
    :param flow:
    :param event_shape:
    :param n_chains:
    :param n_iterations:
    :return:
    """
    flow_object = create_flow_object(flow_name=flow, event_shape=event_shape)
    return neutra_hmc_base(flow_object, target, n_chains, n_vi_iterations=n_iterations, n_hmc_iterations=n_iterations)


def tess(
        negative_log_likelihood: Potential,
        flow: str,
        n_chains: int = 100,
        n_iterations: int = 1000,
        show_progress: bool = True,
        **kwargs
):
    flow_object = create_flow_object(flow_name=flow, event_shape=negative_log_likelihood.event_shape)
    return transport_elliptical_slice_sampler(
        flow_object,
        negative_log_likelihood,
        show_progress=show_progress,
        n_sampling_iterations=n_iterations,
        n_warmup_iterations=n_iterations
    )


def ess(
        negative_log_likelihood: Potential,
        n_chains: int = 100,
        n_iterations: int = 1000,
        show_progress=True,
        **kwargs
):
    """
    Elliptical slice sampler.
    Samples from distribution defined by the density P(x)*L(x) where P is a multivariate normal prior with zero mean and
     cov covariance (can be specified in kwargs) and L is the likelihood.
    """
    return elliptical_slice_sampler(
        negative_log_likelihood,
        event_shape=negative_log_likelihood.event_shape,
        n_chains=n_chains,
        n_iterations=n_iterations,
        show_progress=show_progress,
        **kwargs
    )


def dlmc(target: Potential, flow: str, prior: Potential = None, n_particles: int = 100, n_iterations: int = 500,
         show_progress: bool = True, **kwargs):
    # Defaults to a standard normal prior
    if prior is None:
        prior = StandardGaussian(event_shape=target.event_shape)
    x_prior = prior.sample(batch_shape=(n_particles,))
    flow_object = create_flow_object(flow, prior.event_shape)
    return deterministic_langevin_monte_carlo_base(
        x_prior,
        lambda x: target(x) + prior(x),
        target,
        flow_object,
        full_output=True,
        show_progress=show_progress,
        n_iterations=n_iterations,
        **kwargs
    )
