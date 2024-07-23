import math
from copy import deepcopy
from tqdm import tqdm
import torch

from nfmc.util import metropolis_acceptance_log_ratio, DualAveraging
from normalizing_flows.utils import sum_except_batch  # Remove this dependence


def mass_matrix_multiply(x: torch.Tensor, inverse_mass_matrix_diagonal: torch.Tensor, event_shape):
    # x.shape = (*batch_shape, *event_shape)
    # event_size = prod(event_shape)
    # inverse_mass_matrix_diagonal.shape = (event_size,)
    # output_shape = x.shape
    # output = x * inverse_mass_matrix_diagonal
    batch_shape = x.shape[:-len(event_shape)]
    event_size = int(torch.prod(torch.tensor(event_shape)))
    x_reshaped = x.view(*batch_shape, event_size)
    x_reshaped_multiplied = torch.einsum('...i,i->...i', x_reshaped, inverse_mass_matrix_diagonal)
    x_multiplied = x_reshaped_multiplied.view_as(x)
    return x_multiplied


def grad_potential(x: torch.Tensor, potential: callable):
    with torch.enable_grad():
        x.requires_grad_(True)
        grad = torch.autograd.grad(potential(x).sum(), x)[0]
        x = x.detach()
        x.requires_grad_(False)
        grad = grad.detach()
        grad.requires_grad_(False)
    return grad


def hmc_step_b(x: torch.Tensor, momentum: torch.Tensor, step_size: float, potential: callable):
    # momentum update
    return momentum - step_size / 2 * grad_potential(x, potential)


def hmc_step_a(x: torch.Tensor, momentum: torch.Tensor, inv_mass_diag, step_size: float, event_shape):
    # position update
    return x + step_size * mass_matrix_multiply(momentum, inv_mass_diag, event_shape)


def hmc_trajectory(x: torch.Tensor,
                   momentum: torch.Tensor,
                   event_shape,
                   inv_mass_diag: torch.Tensor,
                   step_size: float,
                   n_leapfrog_steps: int,
                   potential: callable,
                   full_output: bool = False):
    xs = []
    for j in range(n_leapfrog_steps):
        momentum = hmc_step_b(x, momentum, step_size, potential)
        x = hmc_step_a(x, momentum, inv_mass_diag, step_size, event_shape)
        momentum = hmc_step_b(x, momentum, step_size, potential)
        if full_output:
            xs.append(x)

    if full_output:
        return torch.stack(xs), x, momentum
    return x, momentum


def hmc(x0: torch.Tensor,
        potential: callable,
        n_iterations: int = 100,
        n_leapfrog_steps: int = 50,
        step_size: float = 0.01,
        target_acceptance_rate: float = 0.651,
        da_kwargs: dict = None,
        inv_mass_diag: torch.Tensor = None,
        tune_inv_mass_diag: bool = True,
        tune_step_size: bool = True,
        show_progress: bool = True,
        adjustment: bool = True,
        full_output: bool = False,
        imd_adjustment: float = 1e-3):
    if da_kwargs is None:
        da_kwargs = dict()

    n_chains, *event_shape = x0.shape
    xs = torch.zeros(size=(n_iterations, n_chains, *event_shape))
    da = DualAveraging(initial_step_size=step_size, **da_kwargs)
    x = torch.clone(x0).detach()
    if inv_mass_diag is None:
        inv_mass_diag = torch.ones(size=event_shape)

    n_divergences = 0
    accepted_total = 0
    for i in (pbar := tqdm(range(n_iterations), desc='HMC', disable=not show_progress)):
        p = mass_matrix_multiply(
            torch.randn_like(x),
            1 / inv_mass_diag.sqrt(),
            event_shape
        )

        try:
            x_prime, p_prime = hmc_trajectory(
                x,
                p,
                event_shape=event_shape,
                inv_mass_diag=inv_mass_diag,
                step_size=step_size,
                n_leapfrog_steps=n_leapfrog_steps,
                potential=potential
            )

            with torch.no_grad():
                if adjustment:
                    hamiltonian_start = potential(x) + 0.5 * sum_except_batch(
                        mass_matrix_multiply(p ** 2, inv_mass_diag, event_shape),
                        event_shape
                    )
                    hamiltonian_end = potential(x_prime) + 0.5 * sum_except_batch(
                        mass_matrix_multiply(p_prime ** 2, inv_mass_diag, event_shape),
                        event_shape
                    )
                    log_prob_accept = -hamiltonian_end - (-hamiltonian_start)
                    log_u = torch.rand_like(log_prob_accept).log()  # n_chains
                    accepted_mask = (log_u < log_prob_accept)  # n_chains
                else:
                    accepted_mask = torch.ones(size=(n_chains,), dtype=torch.bool)

                x[accepted_mask] = x_prime[accepted_mask]
        except ValueError:
            accepted_mask = torch.zeros(size=(n_chains,), dtype=torch.bool)
            n_divergences += 1

        with torch.no_grad():
            accepted_total += int(torch.sum(accepted_mask))
            x = x.detach()
            xs[i] = x
            acc_rate = torch.mean(accepted_mask.float())

            if tune_inv_mass_diag:
                # inv_mass_diag = torch.var(x.flatten(1, -1), dim=0)  # Mass matrix adaptation
                inv_mass_diag = imd_adjustment * torch.var(x.flatten(1, -1), dim=0) + (
                            1 - imd_adjustment) * inv_mass_diag
            if tune_step_size and adjustment:  # Step size tuning is only possible with adjustment right now
                error = target_acceptance_rate - acc_rate
                da.step(error)
                step_size = da.value  # Step size adaptation
            pbar.set_postfix_str(
                f'acc: {accepted_total / (n_chains * (i + 1)):.3f} [{acc_rate:.3f}], '
                f'log step: {math.log(step_size):.2f}, '
                f'imd_max_norm: {torch.max(torch.abs(inv_mass_diag)):.2f}, '
                f'da_error: {da.error_sum:.2f}, '
                # f'da_log_step: {da.log_step:.2f}, '
                f'n_divergences: {n_divergences}'
            )
    if full_output:
        return xs, {
            'inv_mass_diag': inv_mass_diag,
            'step_size': step_size,
            'n_leapfrog_steps': n_leapfrog_steps
        }
    return xs
