import torch

from potentials.base import Potential
from nfmc.sampling.neutra import neutra_hmc_base
from nfmc.sampling.elliptical import transport_elliptical_slice_sampling_base, elliptical_slice_sampler
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


def tess(target: Potential, flow: str, n_particles: int = 100, n_iterations: int = 1000):
    flow_object = create_flow_object(flow_name=flow, event_shape=target.event_shape)
    x0 = torch.randn(size=(n_particles, *target.event_shape))
    return transport_elliptical_slice_sampling_base(
        x0,
        flow_object,
        target,
        n_sampling_iterations=n_iterations,
        n_warmup_iterations=n_iterations
    )


def ess(target: Potential, n_chains: int = 100, n_iterations: int = 1000, show_progress=True, **kwargs):
    """
    Elliptical slice sampler.
    """
    return elliptical_slice_sampler(
        target,
        n_chains=n_chains,
        n_iterations=n_iterations,
        show_progress=show_progress,
        **kwargs
    )
