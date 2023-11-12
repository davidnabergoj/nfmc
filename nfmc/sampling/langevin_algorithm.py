from copy import deepcopy

import torch
from tqdm import tqdm

from nfmc.util import metropolis_acceptance_log_ratio
from normalizing_flows import Flow
from nfmc.mcmc.langevin_algorithm import base as base_langevin


def langevin_algorithm_base(x0: torch.Tensor,
                            flow: Flow,
                            potential: callable,
                            n_jumps: int = 25,
                            jump_period: int = 500,
                            batch_size: int = 128,
                            burnin: int = 1000,
                            nf_adjustment: bool = True,
                            show_progress: bool = True,
                            **kwargs):
    n_chains, *event_shape = x0.shape

    x = deepcopy(x0)

    # Burnin to get to the typical set
    x = base_langevin(
        x0=x,
        n_iterations=burnin,
        full_output=False,
        potential=potential,
        **kwargs
    )

    assert torch.all(torch.isfinite(x))

    # In the burnin stage, fit the flow to the typical set data
    flow.fit(x)
    # Note: in practice, it is quite useful to have a decent fit at this point.

    xs = []

    # Langevin with NF jumps
    if show_progress:
        iterator = tqdm(range(n_jumps), desc='NF Langevin algorithm')
    else:
        iterator = range(n_jumps)

    accepted = 0
    total = 0

    for _ in iterator:
        assert torch.all(torch.isfinite(x))
        x_lng = base_langevin(
            x0=x,
            n_iterations=jump_period - 1,
            potential=potential,
            **kwargs
        )  # (n_steps, n_chains, *event_shape)
        assert torch.all(torch.isfinite(x_lng))
        xs.append(x_lng)

        x_train = x_lng.view(-1, *event_shape)  # (n_steps * n_chains, *event_shape)
        flow.fit(x_train, n_epochs=1, batch_size=batch_size, shuffle=False)
        x_proposed = flow.sample(n_chains).detach().cpu()  # (n_chains, *event_shape)
        if nf_adjustment:
            x_current = x_lng[-1]
            u_x = potential(x_current)
            u_x_prime = potential(x_proposed)
            f_x = flow.log_prob(x_current)
            f_x_prime = flow.log_prob(x_proposed)
            log_alpha = metropolis_acceptance_log_ratio(
                log_prob_curr=-u_x,
                log_prob_prime=-u_x_prime,
                log_proposal_curr=f_x,
                log_proposal_prime=f_x_prime
            ).cpu()
            acceptance_mask = torch.rand_like(log_alpha).log() < log_alpha
            x[acceptance_mask] = x_proposed[acceptance_mask]
            accepted += int(torch.sum(torch.as_tensor(acceptance_mask).float()))
        else:
            x = x_proposed
            accepted += n_chains
        total += n_chains

        if show_progress:
            iterator.set_postfix_str(f'accept-frac: {accepted / total:.4f}')

        # x.shape = (n_chains, n_dim)

        xs.append(deepcopy(x)[None])

    return torch.cat(xs, dim=0)


def unadjusted_langevin_algorithm_base(*args, **kwargs):
    return langevin_algorithm_base(*args, **kwargs, adjustment=False)


def metropolis_adjusted_langevin_algorithm_base(*args, **kwargs):
    return langevin_algorithm_base(*args, **kwargs, adjustment=True)
