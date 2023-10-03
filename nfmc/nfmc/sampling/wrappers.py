import torch

from potentials.base import Potential
from nfmc.nfmc.sampling import neutra_hmc_base
from nfmc.nfmc.sampling import transport_elliptical_slice_sampling_base
from nfmc.nfmc.sampling.independent_metropolis_hastings import independent_metropolis_hastings_base
from nfmc.nfmc.sampling.langevin_algorithm import metropolis_adjusted_langevin_algorithm_base, unadjusted_langevin_algorithm_base
from nfmc.util import create_flow_object


def mala(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_chains, target.n_dim))
    return metropolis_adjusted_langevin_algorithm_base(x0, flow_object, target)


def ula(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_chains, target.n_dim))
    return unadjusted_langevin_algorithm_base(x0, flow_object, target)


def imh(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_chains, target.n_dim))
    return independent_metropolis_hastings_base(x0, flow_object, target)


def neutra_hmc(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    return neutra_hmc_base(flow_object, target, n_chains)


def tess(target: Potential, flow: str, n_particles: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_particles, target.n_dim))
    return transport_elliptical_slice_sampling_base(x0, flow_object, target)
