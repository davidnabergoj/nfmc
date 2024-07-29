import torch
from tqdm import tqdm

from nfmc.mcmc import hmc
from nfmc.util import metropolis_acceptance_log_ratio
from normalizing_flows import Flow


def jump_hmc_no_burn_in(x0: torch.Tensor,
                        potential: callable,
                        flow: Flow,
                        inv_mass_diag: torch.Tensor,
                        step_size: float,
                        n_jumps: int = 100,
                        n_trajectories_per_jump: int = 10,
                        n_leapfrog_steps: int = 15,
                        max_train_size: int = 1000,
                        max_val_size: int = 1000,
                        train_fraction: float = 0.8,
                        n_epochs: int = 50,
                        early_stopping_threshold: int = 10,
                        fit_nf: bool = False,
                        adjusted_jumps: bool = True,
                        show_progress: bool = False,
                        **kwargs):
    n_total_accepted_jumps = 0
    n_total_hmc_divergences = 0
    n_total_hmc_accepts = 0

    n_chains, *event_shape = x0.shape
    x = torch.clone(x0)
    xs = torch.zeros(
        size=(n_jumps * (n_trajectories_per_jump + 1), *x0.shape),
        requires_grad=False
    )  # (steps, chains, *event)
    step_index = 0
    for i in (pbar := tqdm(range(n_jumps), desc='Jump HMC', disable=not show_progress)):
        # HMC trajectories
        pbar.set_description_str(f'JHMC (HMC sampling)')
        hmc_output = hmc(
            target=potential,
            x0=x,
            step_size=step_size,
            inv_mass_diag=inv_mass_diag,
            n_iterations=n_trajectories_per_jump,
            tune_step_size=False,
            tune_inv_mass_diag=False,
            show_progress=False,
            full_output=True,
            n_leapfrog_steps=n_leapfrog_steps,
            **kwargs
        )
        xs[step_index:step_index + len(hmc_output['samples'])] = hmc_output['samples'].detach()
        step_index += len(hmc_output['samples'])
        n_total_hmc_divergences += hmc_output['statistics']['n_divergences']
        n_total_hmc_accepts += hmc_output['statistics']['n_accepted']

        # Fit flow
        if fit_nf:
            n_data = step_index
            perm = torch.randperm(n_data)
            n_train = int(train_fraction * n_data)
            x_train = xs[:n_data][perm].flatten(0, 1)[:n_train][:max_train_size]
            x_val = xs[:n_data][perm].flatten(0, 1)[n_train:][:max_val_size]
            pbar.set_description_str(f'JHMC (fitting NF)')
            flow.fit(
                x_train=x_train,
                x_val=x_val,
                early_stopping=True,
                n_epochs=n_epochs,
                early_stopping_threshold=early_stopping_threshold
            )

        # Adjusted jump
        pbar.set_description_str(f'JHMC (jumping)')
        x_prime = flow.sample(n=n_chains).detach()

        x = hmc_output['samples'][-1]
        if adjusted_jumps:
            try:
                u_x = potential(x)
                u_x_prime = potential(x_prime)
                f_x = flow.log_prob(x)
                f_x_prime = flow.log_prob(x_prime)
                log_alpha = metropolis_acceptance_log_ratio(
                    log_prob_curr=-u_x,
                    log_prob_prime=-u_x_prime,
                    log_proposal_curr=f_x,
                    log_proposal_prime=f_x_prime
                )
                acceptance_mask = torch.rand_like(log_alpha).log() < log_alpha
            except ValueError:
                acceptance_mask = torch.zeros(size=x.shape[:-len(event_shape)], dtype=torch.bool)

            x[acceptance_mask] = x_prime[acceptance_mask]
            n_current_accepted_jumps = int(torch.sum(acceptance_mask))
        else:
            x = x_prime
            n_current_accepted_jumps = n_chains

        n_total_accepted_jumps += n_current_accepted_jumps

        hmc_total_acc_rate = n_total_hmc_accepts / (n_trajectories_per_jump * (i + 1) * n_chains)
        hmc_current_acc_rate = hmc_output["statistics"]["acceptance_rate"]
        hmc_current_divergences = hmc_output["statistics"]["n_divergences"]
        jump_total_acc_rate = n_total_accepted_jumps / ((i + 1) * n_chains)
        jump_current_acc_rate = n_current_accepted_jumps / n_chains
        total_acc_rate = (n_total_hmc_accepts + n_current_accepted_jumps) / (
                (n_trajectories_per_jump + 1) * (i + 1) * n_chains)
        current_acc_rate = (n_current_accepted_jumps + hmc_output['statistics']['n_accepted']) / (
                n_chains * (n_trajectories_per_jump + 1))
        pbar.set_postfix_str(
            f'Acceptance rate (total: {total_acc_rate:.3f}, current: {current_acc_rate:.3f}) | '
            f'HMC acceptance rate (total: {hmc_total_acc_rate:.3f}, current: {hmc_current_acc_rate:.3f}) | '
            f'Jump acceptance rate (total: {jump_total_acc_rate:.3f}, current: {jump_current_acc_rate:.3f}) | '
            f'HMC divergences (total: {n_total_hmc_divergences}, current: {hmc_current_divergences})'
        )
        xs[step_index] = x
        step_index += 1
    return xs


def jhmc(x0: torch.Tensor,
         flow: Flow,
         target: callable,
         n_burn_in_iterations: int = 400,
         n_tuning_iterations: int = 1000,
         n_jumps: int = 250,
         n_trajectories_per_jump: int = 20,
         show_progress: bool = True,
         n_leapfrog_steps: int = 15,
         step_size: float = 0.01,
         inv_mass_diag: torch.Tensor = None,
         pct_train: float = 0.8,
         initial_fit_kwargs: dict = None,
         jump_hmc_kwargs: dict = None,
         skip_burnin_1: bool = False,
         skip_tuning: bool = False,
         skip_burnin_2: bool = False,
         skip_nf_fit: bool = False,
         **kwargs):
    """

    :param x0:
    :param flow:
    :param target: target potential.
    :param n_burn_in_iterations:
    :param n_tuning_iterations:
    :param n_jumps:
    :param n_trajectories_per_jump:
    :param show_progress:
    :param n_leapfrog_steps:
    :param step_size:
    :param inv_mass_diag:
    :param pct_train:
    :param initial_fit_kwargs:
    :param jump_hmc_kwargs:
    :param skip_burnin_1:
    :param skip_tuning:
    :param skip_burnin_2:
    :param skip_nf_fit:
    :param kwargs:
    :return:
    """
    if inv_mass_diag is None:
        inv_mass_diag = torch.ones(size=(flow.event_size,))
    if initial_fit_kwargs is None:
        initial_fit_kwargs = dict()
    if jump_hmc_kwargs is None:
        jump_hmc_kwargs = dict()

    # burn-in phase 1
    if skip_burnin_1:
        x_burn_in_1 = x0[None]
    else:
        x_burn_in_1 = hmc(
            target=target,
            x0=x0,
            n_iterations=max(n_burn_in_iterations // 2, 1),
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=step_size,
            inv_mass_diag=inv_mass_diag,
            tune_step_size=False,
            tune_inv_mass_diag=False,
            show_progress=show_progress,
            **kwargs
        ).detach()

    # tuning
    if skip_tuning:
        x_tuning = x_burn_in_1
        kernel = {
            'inv_mass_diag': inv_mass_diag,
            'step_size': step_size,
            'n_leapfrog_steps': n_leapfrog_steps
        }
    else:
        hmc_tuning_output = hmc(
            target=target,
            x0=x_burn_in_1[-1],
            n_iterations=n_tuning_iterations,
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=step_size,
            inv_mass_diag=inv_mass_diag,
            tune_step_size=True,
            tune_inv_mass_diag=True,
            show_progress=show_progress,
            full_output=True,
            **kwargs
        )
        x_tuning = hmc_tuning_output['samples'].detach()
        kernel = hmc_tuning_output['kernel']

    # burn-in phase 2
    if skip_burnin_2:
        x_burn_in_2 = x_tuning
    else:
        x_burn_in_2 = hmc(
            target=target,
            x0=x_tuning[-1],
            n_iterations=max(n_burn_in_iterations // 2, 1),
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=kernel['step_size'],
            inv_mass_diag=kernel['inv_mass_diag'],
            tune_step_size=False,
            tune_inv_mass_diag=False,
            show_progress=show_progress,
            **kwargs
        ).detach()

    # NF fit
    if not skip_nf_fit:
        x_burn_in_2_flat = x_burn_in_2.flatten(0, 1)
        n_data = len(x_burn_in_2_flat)
        n_train = int(n_data * pct_train)
        x_burn_in_2_flat = x_burn_in_2_flat[torch.randperm(n_data)]  # shuffle data
        x_train = x_burn_in_2_flat[:n_train]
        x_val = x_burn_in_2_flat[n_train:]
        flow.fit(x_train=x_train, x_val=x_val, early_stopping=True, show_progress=show_progress, **initial_fit_kwargs)

    # sampling
    return jump_hmc_no_burn_in(
        x0=x_burn_in_2[-1],
        potential=target,
        flow=flow,
        n_jumps=n_jumps,
        n_trajectories_per_jump=n_trajectories_per_jump,
        step_size=kernel['step_size'],
        inv_mass_diag=kernel['inv_mass_diag'],
        n_leapfrog_steps=kernel['n_leapfrog_steps'],
        show_progress=show_progress,
        **jump_hmc_kwargs
    )
