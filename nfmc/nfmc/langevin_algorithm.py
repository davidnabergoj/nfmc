from copy import deepcopy

import torch
from tqdm import tqdm
from normalizing_flows import Flow
from nfmc.mcmc.langevin_algorithm import base as base_langevin


def base(x0: torch.Tensor,
         flow: Flow,
         potential: callable,
         n_jumps: int = 25,
         jump_period: int = 100,
         batch_size: int = 128,
         burnin: int = 1000,
         full_output: bool = False,
         nf_adjustment: bool = False,
         show_progress: bool = True,
         **kwargs):
    n_chains, n_dim = x0.shape

    xs = []
    x = deepcopy(x0)

    # Burnin to get to the typical set
    x = base_langevin(
        x0=x,
        n_iterations=burnin,
        full_output=False,
        potential=potential,
        **kwargs
    )

    # Langevin with NF jumps
    if show_progress:
        iterator = tqdm(range(n_jumps), desc='NF Langevin algorithm')
    else:
        iterator = range(n_jumps)

    for _ in iterator:
        x_lng = base_langevin(
            x0=x,
            n_iterations=jump_period - 1,
            full_output=True,
            potential=potential,
            **kwargs
        )
        if full_output:
            xs.append(x_lng)

        x_train = x_lng.view(-1, n_dim)
        flow.fit(x_train, n_epochs=1, batch_size=batch_size, shuffle=False)
        x_proposed = flow.sample(n_chains).detach().cpu()
        if nf_adjustment:
            x_current = x_train[-1]
            t0 = potential(x_current)
            t1 = potential(x_proposed)
            t2 = flow.log_prob(x_current)
            t3 = flow.log_prob(x_proposed)
            log_alpha = (t0 - t1 + t2 - t3).cpu()
            log_u = torch.rand_like(log_alpha).log()
            acceptance_mask = log_u > log_alpha
            x[acceptance_mask] = x_proposed[acceptance_mask]
        else:
            x = x_proposed

        if full_output:
            xs.append(deepcopy(x.unsqueeze(0)))

    if full_output:
        return torch.cat(xs, dim=0)
    else:
        return x


def ula(*args, **kwargs):
    return base(*args, **kwargs, adjustment=False)


def mala(*args, **kwargs):
    return base(*args, **kwargs, adjustment=True)
