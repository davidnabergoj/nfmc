from typing import Union, List, Tuple, Optional

import torch

from nfmc.algorithms.base import NFMCKernel, MCMCOutput
from nfmc.algorithms.sampling.mcmc.ess import ESSKernel, ESSParameters, ESS
from nfmc.algorithms.sampling.mcmc.hmc import HMCKernel, HMCParameters, UHMC, HMC
from nfmc.algorithms.sampling.mcmc.langevin import LangevinKernel, LangevinParameters, ULA, MALA
from nfmc.algorithms.sampling.mcmc.mh import MHKernel, MHParameters, MH
from nfmc.algorithms.sampling.nfmc.imh import IMHKernel, IMHParameters, AdaptiveIMH
from nfmc.algorithms.sampling.nfmc.jump import JumpNFMCParameters, JumpULA, JumpHMC, JumpUHMC, JumpMALA
from nfmc.algorithms.sampling.nfmc.neutra import NeuTraKernel, NeuTraParameters, NeuTraHMC
from nfmc.algorithms.sampling.nfmc.tess import TESSKernel, TESSParameters, TESS
from nfmc.util import create_flow_object
from normalizing_flows import Flow
from potentials.base import Potential


def sample(target: callable,
           event_shape: Optional[Union[torch.Size, Tuple[int]]] = None,
           flow: Optional[Union[str, Flow]] = None,
           strategy: str = "imh",
           n_chains: int = 100,
           n_iterations: int = 100,
           x0: torch.Tensor = None,
           edge_list: List[Tuple[int, int]] = None,
           negative_log_likelihood: callable = None,
           kernel_kwargs: Optional[dict] = None,
           param_kwargs: Optional[dict] = None,
           inner_kernel_kwargs: Optional[dict] = None,
           inner_param_kwargs: Optional[dict] = None,
           device: torch.device = torch.device("cpu"),
           **kwargs) -> Union[MCMCOutput, torch.Tensor]:
    if flow is not None:
        event_shape = flow.event_shape
    elif isinstance(target, Potential):
        event_shape = target.event_shape
    if x0 is None:
        x0 = torch.randn(size=(n_chains, *event_shape))
    if kernel_kwargs is None:
        kernel_kwargs = dict()
    if param_kwargs is None:
        param_kwargs = dict()
    if inner_kernel_kwargs is None:
        inner_kernel_kwargs = dict()
    if inner_param_kwargs is None:
        inner_param_kwargs = dict()

    if strategy in ['hmc', 'uhmc', 'ula', 'mala', 'mh', 'ess']:
        # MCMC
        if strategy == "hmc":
            kernel = HMCKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))), **kernel_kwargs)
            params = HMCParameters(n_iterations=n_iterations, **param_kwargs)
            return HMC(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == "uhmc":
            kernel = HMCKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))), **kernel_kwargs)
            params = HMCParameters(n_iterations=n_iterations, **param_kwargs)
            return UHMC(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == "mala":
            kernel = LangevinKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))), **kernel_kwargs)
            params = LangevinParameters(n_iterations=n_iterations, **param_kwargs)
            return MALA(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == "ula":
            kernel = LangevinKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))), **kernel_kwargs)
            params = LangevinParameters(n_iterations=n_iterations, **param_kwargs)
            return ULA(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == "mh":
            kernel = MHKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))), **kernel_kwargs)
            params = MHParameters(n_iterations=n_iterations, **param_kwargs)
            return MH(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == "ess":
            if negative_log_likelihood is None:
                raise ValueError("Negative log likelihood must be provided")
            kernel = ESSKernel(event_shape=event_shape, **kernel_kwargs)
            params = ESSParameters(n_iterations=n_iterations, **param_kwargs)
            return ESS(event_shape, target, negative_log_likelihood, kernel, params).sample(x0, **kwargs)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
    elif strategy in ["imh", "jump_mala", "jump_ula", "jump_hmc", "jump_uhmc", "neutra_hmc", "tess"]:
        # NFMC
        if flow is None:
            raise ValueError("Flow object must be provided")
        if isinstance(flow, str):
            flow_object = create_flow_object(flow_name=flow, event_shape=event_shape, edge_list=edge_list).to(device)
        elif isinstance(flow, Flow):
            flow_object = flow.to(device)
        else:
            raise ValueError(f"Unknown type for normalizing flow: {type(flow)}")
        if strategy == "imh":
            kernel = IMHKernel(event_shape, flow=flow_object)
            params = IMHParameters(n_iterations=n_iterations, **param_kwargs)
            return AdaptiveIMH(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == 'jump_mala':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(n_iterations=n_iterations, **param_kwargs)
            return JumpMALA(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == 'jump_ula':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(n_iterations=n_iterations, **param_kwargs)
            return JumpULA(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == 'jump_hmc':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(n_iterations=n_iterations, **param_kwargs)
            return JumpHMC(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == 'jump_uhmc':
            kernel = NFMCKernel(event_shape, flow=flow_object)
            params = JumpNFMCParameters(n_iterations=n_iterations, **param_kwargs)
            return JumpUHMC(event_shape, target, kernel, params).sample(x0, **kwargs)
        elif strategy == "tess":
            if negative_log_likelihood is None:
                raise ValueError("Negative log likelihood must be provided")
            kernel = TESSKernel(event_shape, flow=flow_object)
            params = TESSParameters(n_iterations=n_iterations, **param_kwargs)
            return TESS(event_shape, target, negative_log_likelihood, kernel, params).sample(x0, **kwargs)
        elif strategy == 'neutra_hmc':
            kernel = NeuTraKernel(event_shape, flow=flow_object)
            params = NeuTraParameters(n_iterations=n_iterations, **param_kwargs)
            inner_kernel = HMCKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))), **inner_kernel_kwargs)
            inner_params = HMCParameters(**inner_param_kwargs)
            return NeuTraHMC(event_shape, target, inner_kernel, inner_params, kernel, params).sample(x0, **kwargs)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
