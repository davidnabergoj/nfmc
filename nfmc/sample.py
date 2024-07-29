from typing import Union, List, Tuple, Optional

import torch

from nfmc.mcmc.langevin_algorithm import ula
from nfmc.sampling_implementations.elliptical import transport_elliptical_slice_sampler
from nfmc.sampling_implementations.independent_metropolis_hastings import imh, imh_fixed_flow
from nfmc.sampling_implementations.jump.hamiltonian_monte_carlo import jhmc
from nfmc.sampling_implementations.jump.langevin_monte_carlo import unadjusted_langevin_algorithm_base, \
    metropolis_adjusted_langevin_algorithm_base
from nfmc.sampling_implementations.neutra import neutra_hmc_base
from nfmc.util import create_flow_object
from normalizing_flows import Flow
from nfmc.mcmc import hmc, mh, nuts, mala
from potentials.base import Potential
from nfmc.util import MCMCOutput


def sample(target: callable,
           event_shape: Optional[Union[torch.Size, Tuple[int]]] = None,
           flow: Optional[Union[str, Flow]] = None,
           strategy: str = "imh",
           n_chains: int = 100,
           n_iterations: int = 100,
           x0: torch.Tensor = None,
           edge_list: List[Tuple[int, int]] = None,
           device: torch.device = torch.device("cpu"),
           **kwargs) -> Union[MCMCOutput, torch.Tensor]:
    if flow is not None:
        event_shape = flow.event_shape
    elif isinstance(target, Potential):
        event_shape = target.event_shape
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *event_shape))

    if strategy in ['hmc', 'uhmc', 'ula', 'mala', 'mh']:
        # MCMC
        base_kwargs = {'x0': x0, 'target': target, 'n_iterations': n_iterations}
        if strategy == "hmc":
            return hmc(**base_kwargs, adjustment=True, **kwargs)
        elif strategy == "uhmc":
            return hmc(**base_kwargs, adjustment=False, **kwargs)
        elif strategy == "mala":
            return mala(**base_kwargs, **kwargs)
        elif strategy == "ula":
            return ula(**base_kwargs, **kwargs)
        elif strategy == "mh":
            return mh(**base_kwargs, **kwargs)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
    elif strategy in ["imh", "fixed_imh", "jump_mala", "jump_ula", "jhmc", "fixed_jhmc", "jump_uhmc",
                      "neutra_hmc", "tess"]:
        # NFMC
        if flow is None:
            raise ValueError("Flow object must be provided")
        if isinstance(flow, str):
            flow_object = create_flow_object(flow_name=flow, event_shape=event_shape, edge_list=edge_list).to(device)
        elif isinstance(flow, Flow):
            flow_object = flow.to(device)
        else:
            raise ValueError(f"Unknown type for normalizing flow: {type(flow)}")
        base_kwargs = {'x0': x0, 'target': target, 'flow': flow_object}
        if strategy == "imh":
            return imh(**base_kwargs, n_iterations=n_iterations, **kwargs)
        if strategy == "fixed_imh":
            return imh_fixed_flow(**base_kwargs, n_iterations=n_iterations, **kwargs)
        elif strategy == 'jump_mala':
            return metropolis_adjusted_langevin_algorithm_base(**base_kwargs, n_jumps=n_iterations, **kwargs)
        elif strategy == 'jump_ula':
            return unadjusted_langevin_algorithm_base(**base_kwargs, n_jumps=n_iterations, **kwargs)
        elif strategy == 'jhmc':
            return jhmc(**base_kwargs, n_jumps=n_iterations, jump_hmc_kwargs={'adjusted_jumps': True}, **kwargs)
        elif strategy == 'fixed_jhmc':
            return jhmc(
                **base_kwargs,
                n_jumps=n_iterations,
                jump_hmc_kwargs={'adjusted_jumps': True, 'fit_nf': False},
                skip_burnin_1=True,
                skip_tuning=True,
                skip_burnin_2=True,
                skip_nf_fit=True,
                **kwargs
            )
        elif strategy == 'jump_uhmc':
            # return jump_uhmc(**base_kwargs, n_jumps=n_iterations, jump_hmc_kwargs={'adjusted_jumps': False}, **kwargs)
            raise NotImplementedError
        elif strategy == "tess":
            return transport_elliptical_slice_sampler(
                flow=base_kwargs['flow'],
                negative_log_likelihood=base_kwargs["target"],
                n_chains=n_chains,
                n_sampling_iterations=n_iterations,
                n_warmup_iterations=n_iterations,
                **kwargs
            )
        elif strategy == 'neutra_hmc':
            return neutra_hmc_base(
                flow=base_kwargs['flow'],
                potential=base_kwargs["target"],
                n_chains=n_chains,
                n_vi_iterations=n_iterations,
                n_hmc_iterations=n_iterations,
                **kwargs
            )
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
