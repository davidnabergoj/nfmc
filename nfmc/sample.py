from typing import Union, List, Tuple

import torch

from nfmc.sampling_implementations.elliptical import transport_elliptical_slice_sampler
from nfmc.sampling_implementations.independent_metropolis_hastings import independent_metropolis_hastings_base
from nfmc.sampling_implementations.jump.hamiltonian_monte_carlo import unadjusted_hmc_base, adjusted_hmc_base
from nfmc.sampling_implementations.jump.langevin_monte_carlo import unadjusted_langevin_algorithm_base, \
    metropolis_adjusted_langevin_algorithm_base
from nfmc.sampling_implementations.neutra import neutra_hmc_base
from nfmc.util import create_flow_object
from normalizing_flows import Flow


def sample(target: callable,
           flow: Union[str, Flow],
           event_shape,
           strategy: str = "imh",
           n_chains: int = 100,
           n_iterations: int = 100,
           x0: torch.Tensor = None,
           edge_list: List[Tuple[int, int]] = None,
           device: torch.device = torch.device("cpu"),
           **kwargs):
    # strategy_choices = ["imh", "jump_mala", "jump_ula", "jump_hmc", "jump_uhmc", "tess", "neutra_hmc"]

    # Create the flow object if necessary
    if isinstance(flow, str):
        flow_object = create_flow_object(flow_name=flow, event_shape=event_shape, edge_list=edge_list).to(device)
    else:
        flow_object = flow.to(device)

    # Create a set of initial chain states if necessary
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *event_shape))

    if strategy == "imh":
        return independent_metropolis_hastings_base(
            x0,
            flow_object,
            target,
            n_iterations=n_iterations,
            **kwargs
        )
    elif strategy == "jump_mala":
        return metropolis_adjusted_langevin_algorithm_base(
            x0,
            flow_object,
            target,
            n_jumps=n_iterations,
            **kwargs
        )
    elif strategy == "jump_ula":
        return unadjusted_langevin_algorithm_base(
            x0,
            flow_object,
            target,
            n_jumps=n_iterations,
            **kwargs
        )
    elif strategy == "jump_hmc":
        return adjusted_hmc_base(
            x0,
            flow_object,
            target,
            n_jumps=n_iterations,
            **kwargs
        )
    elif strategy == "jump_uhmc":
        return unadjusted_hmc_base(
            x0,
            flow_object,
            target,
            n_jumps=n_iterations,
            **kwargs
        )
    elif strategy == "tess":
        return transport_elliptical_slice_sampler(
            flow_object,
            target,
            n_chains=n_chains,
            n_sampling_iterations=n_iterations,
            n_warmup_iterations=n_iterations,
            **kwargs
        )
    elif strategy == "neutra_hmc":
        return neutra_hmc_base(
            flow_object,
            target,
            n_chains,
            n_vi_iterations=n_iterations,
            n_hmc_iterations=n_iterations,
            **kwargs
        )
    else:
        raise ValueError
