import math
from copy import deepcopy
from tqdm import tqdm
import torch

from nfmc.util import metropolis_acceptance_log_ratio, DualAveraging
from normalizing_flows.utils import sum_except_batch


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


def hmc_step_a(x: torch.Tensor, momentum: torch.Tensor, inv_mass_diag, step_size: float):
    # position update
    return x + step_size * inv_mass_diag * momentum


def hmc_trajectory(x: torch.Tensor,
                   momentum: torch.Tensor,
                   inv_mass_diag: torch.Tensor,
                   step_size: float,
                   n_leapfrog_steps: int,
                   potential: callable,
                   full_output: bool = False):
    xs = []
    for j in range(n_leapfrog_steps):
        momentum = hmc_step_b(x, momentum, step_size, potential)
        x = hmc_step_a(x, momentum, inv_mass_diag, step_size)
        momentum = hmc_step_b(x, momentum, step_size, potential)
        if full_output:
            xs.append(x)

    if full_output:
        return torch.stack(xs), x, momentum
    return x, momentum


def hmc(x0: torch.Tensor,
        potential: callable,
        n_iterations: int = 1000,
        n_leapfrog_steps: int = 15,
        step_size: float = None,
        full_output: bool = False,
        target_acceptance_rate: float = 0.651,
        show_progress: bool = False):
    n_chains, *event_shape = x0.shape

    xs = torch.zeros(size=(n_iterations, *x0.shape), dtype=x0.dtype)

    x = deepcopy(x0)
    n_dim = int(torch.prod(torch.as_tensor(event_shape)))
    if step_size is None:
        step_size = n_dim ** (-1 / 4)
    dual_avg = DualAveraging(math.log(step_size))
    inv_mass_diag = torch.std(x0, dim=0)  # has shape: event_shape

    if show_progress:
        iterator = tqdm(range(n_iterations), desc='HMC')
    else:
        iterator = range(n_iterations)

    accepted = 0
    total = 0

    for i in iterator:
        initial_momentum = torch.randn_like(x) / inv_mass_diag.sqrt()[None]
        x_prime, momentum_prime = hmc_trajectory(
            x,
            initial_momentum,
            inv_mass_diag,
            step_size,
            n_leapfrog_steps,
            potential
        )
        with torch.no_grad():
            log_alpha = metropolis_acceptance_log_ratio(
                log_prob_curr=-potential(x) - 0.5 * sum_except_batch(
                    initial_momentum ** 2 * inv_mass_diag,
                    event_shape=event_shape
                ),
                log_prob_prime=-potential(x_prime) - 0.5 * sum_except_batch(
                    momentum_prime ** 2 * inv_mass_diag,
                    event_shape=event_shape
                ),
                log_proposal_curr=0.0,
                log_proposal_prime=0.0
            )  # batch_shape
            log_u = torch.rand_like(log_alpha).log()  # n_chains
            accepted_mask = torch.less(log_u, log_alpha)  # n_chains
            x[accepted_mask] = x_prime[accepted_mask]
            x = x.detach()

            inv_mass_diag = torch.std(x0, dim=0)
            acceptance_rate = float(torch.mean(accepted_mask.float()))
            dual_avg.step(target_acceptance_rate - acceptance_rate)
            step_size = math.exp(dual_avg.value)

            accepted += int(torch.sum(accepted_mask))
            total += n_chains

            if show_progress:
                iterator.set_postfix_str(f'accept-frac: {accepted / total:.4f}')

            if full_output:
                xs[i] = deepcopy(x)

    if full_output:
        return xs.detach()
    else:
        return x.detach()
