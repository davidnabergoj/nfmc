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
         tau: float = None,
         full_output: bool = False,
         adjustment: bool = False,
         target_acceptance_rate: float = 0.574):
    # TODO tune tau in MALA
    assert len(x0.shape) == 2
    n_chains, n_dim = x0.shape
    if tau is None:
        tau = n_dim ** (-1 / 3)
    xs = []
    sqrt_a = torch.std(x0, dim=0)  # root of the preconditioning matrix diagonal
    dual_avg = DualAveraging(math.log(tau))

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
        x_prime = x - tau * sqrt_a.view(1, -1).square() * grad_u_x + math.sqrt(2 * tau) * sqrt_a.view(1, -1) * noise

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
                log_proposal_curr=-proposal_potential(x, x_prime, grad_u_x_prime, sqrt_a ** 2, tau),
                log_proposal_prime=-proposal_potential(x_prime, x, grad_u_x, sqrt_a ** 2, tau)
            )
            adjustment_mask = torch.as_tensor(torch.log(torch.rand(n_chains)) < log_ratio)
            acceptance_rate = float(torch.mean(adjustment_mask.float()))
            dual_avg.step(target_acceptance_rate - acceptance_rate)
            tau = math.exp(dual_avg())
        else:
            # No adjustment (ULA)
            adjustment_mask = torch.ones(n_chains, dtype=torch.bool)
        x[adjustment_mask] = x_prime[adjustment_mask]

        sqrt_a = torch.std(x, dim=0)

        if full_output:
            xs.append(deepcopy(x))

    if full_output:
        return torch.stack(xs)
    else:
        return x


def mala(*args, **kwargs):
    return base(*args, **kwargs, adjustment=True)


def ula(*args, **kwargs):
    return base(*args, **kwargs, adjustment=False)
