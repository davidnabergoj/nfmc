from copy import deepcopy

import torch
from tqdm import tqdm

from nfmc.sampling.jump.base import JumpMCMC
from nfmc.util import metropolis_acceptance_log_ratio
from normalizing_flows import Flow
from nfmc.mcmc.langevin_algorithm import base as base_langevin


class NFLMC(JumpMCMC):
    def __init__(self,
                 target_potential: callable,
                 flow: Flow,
                 mcmc_kwargs: dict = None,
                 inv_mass_diag: torch.Tensor = None,
                 step_size: float = None,
                 **kwargs):
        self.inv_mass_diag = inv_mass_diag
        self.step_size = step_size
        if mcmc_kwargs is None:
            mcmc_kwargs = dict()
        self.mcmc_kwargs = mcmc_kwargs

        super().__init__(target_potential, flow, **kwargs)

    @property
    def name(self) -> str:
        return "NF-LMC"

    def sample_mcmc(self, x: torch.Tensor):
        x, kernel_params = base_langevin(
            x0=x,
            n_iterations=self.jump_period - 1,
            potential=self.target_potential,
            return_kernel_parameters=True,
            **{**self.mcmc_kwargs, **{
                "inv_mass_diag": self.inv_mass_diag,
                "step_size": self.step_size}
               }  # Reuse old kernel parameters and have them overwrite whatever is in kwargs
        )  # (n_steps, n_chains, *event_shape) where n_steps = self.jump_period - 1
        self.inv_mass_diag = kernel_params["inv_mass_diag"]
        self.step_size = kernel_params["step_size"]
        return x


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
    # Burnin with standard LMC
    x, kernel_params = base_langevin(
        x0=x0,
        n_iterations=burnin,
        full_output=False,
        potential=potential,
        return_kernel_parameters=True,
        **kwargs
    )

    nf_lmc = NFLMC(
        target_potential=potential,
        flow=flow,
        n_jumps=n_jumps,
        jump_period=jump_period,
        show_progress=show_progress,
        flow_adjustment=nf_adjustment,
        inv_mass_diag=kernel_params["inv_mass_diag"],
        step_size=kernel_params["step_size"]
    )

    x = nf_lmc.sample(x, flow_fit_kwargs={"batch_size": batch_size})
    return x


def langevin_algorithm_base_(x0: torch.Tensor,
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
    n_dim = int(torch.prod(torch.as_tensor(event_shape)))

    x = deepcopy(x0)

    # Burnin to get to the typical set
    x, kernel_params = base_langevin(
        x0=x,
        n_iterations=burnin,
        full_output=False,
        potential=potential,
        return_kernel_parameters=True,
        **kwargs
    )

    assert torch.all(torch.isfinite(x))

    # In the burnin stage, fit the flow to the typical set data
    flow.fit(x)
    # Note: in practice, it is quite useful to have a decent fit at this point.

    xs = []

    # Langevin with NF jumps
    if show_progress:
        iterator = tqdm(
            range(n_jumps),
            desc=f'NF-LMC ({n_chains} chains, '
                 f'{n_dim} dimensions, '
                 f'{jump_period} LMC iterations per jump, '
                 f'{"adjusted" if nf_adjustment else "unadjusted"})'
        )
    else:
        iterator = range(n_jumps)

    accepted = 0
    total = 0

    for _ in iterator:
        assert torch.all(torch.isfinite(x))
        x_lng, kernel_params = base_langevin(
            x0=x,
            n_iterations=jump_period - 1,
            potential=potential,
            return_kernel_parameters=True,
            **{**kwargs, **kernel_params}  # Reuse old kernel parameters and have them overwrite whatever is in kwargs
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
            n_current_accepted = int(torch.sum(torch.as_tensor(acceptance_mask).float()))
            accepted += n_current_accepted
        else:
            x = x_proposed
            n_current_accepted = n_chains
            accepted += n_current_accepted
        total += n_chains

        if show_progress:
            iterator.set_postfix_str(
                f'current accepted fraction: {n_current_accepted / n_chains:.3f}, '
                f'total accepted fraction: {accepted / total:.3f}, '
                f'step: {kernel_params["step_size"]:.3f}, '
                f'norm(inv_mass_diag): {torch.linalg.norm(kernel_params["inv_mass_diag"]):.3f}'
            )

        # x.shape = (n_chains, n_dim)

        xs.append(deepcopy(x)[None])

    return torch.cat(xs, dim=0)


def unadjusted_langevin_algorithm_base(*args, **kwargs):
    return langevin_algorithm_base(*args, **kwargs, adjustment=False)


def metropolis_adjusted_langevin_algorithm_base(*args, **kwargs):
    return langevin_algorithm_base(*args, **kwargs, adjustment=True)
