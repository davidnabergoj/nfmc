import torch

from potentials.base import Potential
from nfmc.nfmc.sampling import neutra_hmc
from nfmc.nfmc.sampling import tess
from nfmc.nfmc.sampling.independent_metropolis_hastings import imh
from nfmc.nfmc.sampling.langevin_algorithm import mala, ula
from nfmc.util import create_flow_object


def mala_wrapper(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_chains, target.n_dim))
    return mala(x0, flow_object, target)


def ula_wrapper(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_chains, target.n_dim))
    return ula(x0, flow_object, target)


def imh_wrapper(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_chains, target.n_dim))
    return imh(x0, flow_object, target)


def neutra_hmc_wrapper(target: Potential, flow: str, n_chains: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    return neutra_hmc(flow_object, target, n_chains)


def tess_wrapper(target: Potential, flow: str, n_particles: int = 100):
    flow_object = create_flow_object(flow_name=flow)
    x0 = torch.randn(size=(n_particles, target.n_dim))
    return tess(x0, flow_object, target)
