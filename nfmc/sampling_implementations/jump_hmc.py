from nfmc import sample
from nfmc.util import metropolis_acceptance_log_ratio
import torch

from normalizing_flows import Flow
from tqdm import tqdm


def jump_hmc(x0: torch.Tensor,
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
             fit_nf: bool = True,
             adjusted_jumps: bool = True,
             **kwargs):
    n_chains, *event_shape = x0.shape
    x = torch.clone(x0)
    xs = torch.zeros(size=(n_jumps * (n_trajectories_per_jump + 1), *x0.shape),
                     requires_grad=False)  # (steps, chains, *event)
    step_index = 0
    for _ in (pbar := tqdm(range(n_jumps), desc='Jump HMC')):
        # HMC trajectories
        pbar.set_postfix_str(f'HMC sampling')
        xs_hmc = sample(
            potential,
            event_shape,
            x0=x,
            step_size=step_size,
            inv_mass_diag=inv_mass_diag,
            n_iterations=n_trajectories_per_jump,
            strategy='hmc',
            tune_step_size=False,
            tune_inv_mass_diag=False,
            show_progress=False,
            n_leapfrog_steps=n_leapfrog_steps,
            **kwargs
        ).detach()
        xs[step_index:step_index + len(xs_hmc)] = xs_hmc
        step_index += len(xs_hmc)

        # Fit flow
        if fit_nf:
            n_data = step_index
            perm = torch.randperm(n_data)
            n_train = int(train_fraction * n_data)
            x_train = xs[:n_data][perm].flatten(0, 1)[:n_train][:max_train_size]
            x_val = xs[:n_data][perm].flatten(0, 1)[n_train:][:max_val_size]
            pbar.set_postfix_str(f'Fitting NF')
            flow.fit(
                x_train=x_train,
                x_val=x_val,
                early_stopping=True,
                n_epochs=n_epochs,
                early_stopping_threshold=early_stopping_threshold
            )

        # Adjusted jump
        x_prime = flow.sample(n=n_chains).detach()

        x = xs_hmc[-1]
        if adjusted_jumps:
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
            x[acceptance_mask] = x_prime[acceptance_mask]

        xs[step_index] = x
        step_index += 1
    return xs

# def sample_jump_hmc(target,
#                     step_size: float = 0.01,
#                     n_burnin: int = 1000,
#                     n_iterations: int = 3000,
#                     n_leapfrog_steps: int = 15,
#                     target_acceptance_rate: float = 0.651):
#     x_hmc_burnin, kernel = sample(
#         target,
#         target.event_shape,
#         step_size=step_size,
#         n_iterations=n_burnin,
#         n_leapfrog_steps=n_leapfrog_steps,
#         target_acceptance_rate=target_acceptance_rate,
#         strategy='hmc',
#         full_output=True
#     )
#
#     x_hmc = sample(
#         target,
#         target.event_shape,
#         x0=x_hmc_burnin[-1],
#         step_size=kernel['step_size'],
#         inv_mass_diag=kernel['inv_mass_diag'],
#         n_iterations=n_iterations,
#         n_leapfrog_steps=n_leapfrog_steps,
#         strategy='hmc',
#         tune_step_size=False,
#         tune_inv_mass_diag=False
#     )
#
#     return x_hmc
