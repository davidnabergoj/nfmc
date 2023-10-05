import torch

from potentials.base import Potential
from nfmc.sampling.neutra import neutra_hmc_base
from nfmc.sampling.elliptical import transport_elliptical_slice_sampler, elliptical_slice_sampler
from nfmc.sampling.independent_metropolis_hastings import independent_metropolis_hastings_base
from nfmc.sampling.langevin_algorithm import metropolis_adjusted_langevin_algorithm_base, \
    unadjusted_langevin_algorithm_base
from nfmc.util import create_flow_object


def mala(target: Potential, flow: str, n_chains: int = 100, n_iterations: int = 100, jump_period: int = 50):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    x0 = torch.randn(size=(n_chains, *target.event_shape))
    return metropolis_adjusted_langevin_algorithm_base(
        x0,
        flow_object,
        target,
        n_jumps=n_iterations,
        jump_period=jump_period
    )


def ula(target: Potential, flow: str, n_chains: int = 100, n_iterations: int = 100, jump_period: int = 50):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    x0 = torch.randn(size=(n_chains, *target.event_shape))
    return unadjusted_langevin_algorithm_base(
        x0,
        flow_object,
        target,
        n_jumps=n_iterations,
        jump_period=jump_period
    )


def imh(target: Potential, flow: str, n_chains: int = 100, n_iterations: int = 1000):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    x0 = torch.randn(size=(n_chains, *target.event_shape))
    return independent_metropolis_hastings_base(x0, flow_object, target, n_iterations=n_iterations)


def neutra_hmc(target: Potential, flow: str, n_chains: int = 100, n_iterations: int = 1000):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
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
