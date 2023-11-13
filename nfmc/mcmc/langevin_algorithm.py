from copy import deepcopy
import math
import torch

from nfmc.util import metropolis_acceptance_log_ratio, DualAveraging


@torch.no_grad()
def proposal_potential(x_prime: torch.Tensor,
                       x: torch.Tensor,
                       grad_u_x: torch.Tensor,
                       a_diag: torch.Tensor,
                       tau: float):
    """
    Compute the Langevin algorithm proposal potential q(x_prime | x).
    """
    assert x_prime.shape == x.shape == grad_u_x.shape
    term = x_prime - x + tau * a_diag.view(1, -1) * grad_u_x
    return (term * (1 / a_diag.view(1, -1)) * term).sum(dim=-1) / (4 * tau)


def base(x0: torch.Tensor,
         potential: callable,
         n_iterations: int = 1000,
         step_size: float = None,
         full_output: bool = True,
         adjustment: bool = False,
         inv_mass_diag: torch.Tensor = None,
         tune_diagonal_preconditioning: bool = True,
         target_acceptance_rate: float = 0.574,
         return_kernel_parameters: bool = False):
    assert torch.all(torch.isfinite(x0))

    # TODO tune tau in MALA
    n_chains, *event_shape = x0.shape
    n_dim = int(torch.prod(torch.as_tensor(event_shape)))
    if step_size is None:
        step_size = n_dim ** (-1 / 3)
    xs = []

    if inv_mass_diag is None:
        # Compute standard deviation of chain states unless using a single chain
        if x0.shape[0] > 1 and tune_diagonal_preconditioning:
            inv_mass_diag = torch.std(x0, dim=0)  # root of the mass matrix diagonal
        else:
            inv_mass_diag = torch.ones(*event_shape)

    dual_avg = DualAveraging(math.log(step_size))

    x = deepcopy(x0)
    for i in range(n_iterations):
        noise = torch.randn_like(x)

        # Compute potential and gradient at current state
        x.requires_grad_(True)
        u_x = potential(x)
        grad_u_x, = torch.autograd.grad(u_x.sum(), x)
        x.requires_grad_(False)  # Clear gradients
        x = x.detach()
        x.grad = None  # Clear gradients

        # Compute new state
        grad_term = -step_size / inv_mass_diag[None].square() * grad_u_x
        noise_term = math.sqrt(2 * step_size) / inv_mass_diag[None] * noise
        x_prime = x + grad_term + noise_term

        if adjustment:
            # Compute potential and gradient at proposed state
            x_prime.requires_grad_(True)
            u_x_prime = potential(x_prime)
            grad_u_x_prime, = torch.autograd.grad(u_x_prime.sum(), x_prime)
            x_prime.requires_grad_(False)  # Clear gradients
            x_prime = x_prime.detach()
            x_prime.grad = None  # Clear gradients

            # Perform metropolis adjustment (MALA)
            log_ratio = metropolis_acceptance_log_ratio(
                log_prob_curr=-u_x,
                log_prob_prime=-u_x_prime,
                log_proposal_curr=-proposal_potential(x, x_prime, grad_u_x_prime, 1 / inv_mass_diag ** 2, step_size),
                log_proposal_prime=-proposal_potential(x_prime, x, grad_u_x, 1 / inv_mass_diag ** 2, step_size)
            )
            adjustment_mask = torch.as_tensor(torch.log(torch.rand(n_chains)) < log_ratio)
            acceptance_rate = float(torch.mean(adjustment_mask.float()))
            dual_avg.step(target_acceptance_rate - acceptance_rate)
            step_size = math.exp(dual_avg.value)
        else:
            # No adjustment (ULA)
            adjustment_mask = torch.ones(n_chains, dtype=torch.bool)
        x[adjustment_mask] = x_prime[adjustment_mask]

        # Update the inerse mass diagonal
        if x0.shape[0] > 1 and tune_diagonal_preconditioning:
            inv_mass_diag = torch.std(x, dim=0)  # root of the preconditioning matrix diagonal

        if full_output:
            xs.append(deepcopy(x))

    if full_output:
        x = torch.stack(xs)

    if return_kernel_parameters:
        return x, {
            "inv_mass_diag": inv_mass_diag,
            "step_size": step_size
        }
    else:
        return x


def mala(*args, **kwargs):
    return base(*args, **kwargs, adjustment=True)


def ula(*args, **kwargs):
    return base(*args, **kwargs, adjustment=False)


def exponential_decay_mu_scheduler(iteration: int, init: float = 1.0, decay: float = 1e-4):
    return init * decay ** iteration


def smooth_lmc(x0: torch.Tensor,
               potential: callable,
               n_iterations: int = 1000,
               step_size: float = None,
               adjustment: bool = False,
               mu_scheduler: callable = None):
    """
    Langevin Monte Carlo with target smoothing for multimodal sampling.
    """
    assert torch.all(torch.isfinite(x0))

    n_chains, *event_shape = x0.shape
    n_dim = int(torch.prod(torch.as_tensor(event_shape)))
    if step_size is None:
        step_size = n_dim ** (-1 / 3)
    xs = []

    # Compute standard deviation of chain states unless using a single chain
    sqrt_a = torch.ones(*event_shape)

    x = deepcopy(x0)
    for i in range(n_iterations):
        noise = torch.randn_like(x)

        # Compute potential and gradient at current state
        x.requires_grad_(True)
        u_x = potential(x + mu_scheduler(i) * torch.randn_like(x))
        grad_u_x, = torch.autograd.grad(u_x.sum(), x)
        x.requires_grad_(False)  # Clear gradients
        x = x.detach()
        x.grad = None  # Clear gradients

        # Compute new state
        x_prime = x - step_size * sqrt_a[None].square() * grad_u_x + math.sqrt(2 * step_size) * sqrt_a[None] * noise

        if adjustment:
            # Compute potential and gradient at proposed state
            x_prime.requires_grad_(True)
            u_x_prime = potential(x_prime)
            grad_u_x_prime, = torch.autograd.grad(u_x_prime.sum(), x_prime)
            x_prime.requires_grad_(False)  # Clear gradients
            x_prime = x_prime.detach()
            x_prime.grad = None  # Clear gradients

            # Perform metropolis adjustment (MALA)
            log_ratio = metropolis_acceptance_log_ratio(
                log_prob_curr=-u_x,
                log_prob_prime=-u_x_prime,
                log_proposal_curr=-proposal_potential(x, x_prime, grad_u_x_prime, sqrt_a ** 2, step_size),
                log_proposal_prime=-proposal_potential(x_prime, x, grad_u_x, sqrt_a ** 2, step_size)
            )
            adjustment_mask = torch.as_tensor(torch.log(torch.rand(n_chains)) < log_ratio)
            acceptance_rate = float(torch.mean(adjustment_mask.float()))
        else:
            # No adjustment (ULA)
            adjustment_mask = torch.ones(n_chains, dtype=torch.bool)
        x[adjustment_mask] = x_prime[adjustment_mask]

        xs.append(deepcopy(x))

    return torch.stack(xs)
