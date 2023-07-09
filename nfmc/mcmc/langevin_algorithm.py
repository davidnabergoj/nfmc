from copy import deepcopy
import math
import torch


@torch.no_grad()
def proposal_potential(x_prime: torch.Tensor,
                       x: torch.Tensor,
                       grad_u_x: torch.Tensor,
                       tau: float):
    """
    Compute the Langevin algorithm proposal potential q(x_prime | x).
    """
    assert x_prime.shape == x.shape == grad_u_x.shape
    norm = torch.linalg.norm(x_prime - x - tau * grad_u_x)
    return norm ** 2 / (4 * tau)


def base(x0: torch.Tensor,
         potential: callable,
         n_iterations: int = 1000,
         tau: float = None,
         full_output: bool = False,
         adjustment: bool = False):
    assert len(x0.shape) == 2
    n_chains, n_dim = x0.shape
    if tau is None:
        tau = n_dim ** (-1 / 3)
    xs = []

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
        x_prime = x - tau * grad_u_x + math.sqrt(2 * tau) * noise

        if adjustment:
            # Compute potential and gradient at proposed state
            x_prime.requires_grad_(True)
            u_x_prime = potential(x_prime)
            grad_u_x_prime, = torch.autograd.grad(u_x_prime.sum(), x_prime)
            x_prime.requires_grad_(False)  # Clear gradients
            x_prime = x_prime.detach()
            x_prime.grad = None  # Clear gradients

            # Perform metropolis adjustment (MALA)
            log_ratio = (
                    - u_x_prime
                    + u_x
                    - proposal_potential(x, x_prime, grad_u_x_prime, tau)
                    + proposal_potential(x_prime, x, grad_u_x, tau)
            )
            adjustment_mask = torch.log(torch.rand(n_chains)) < log_ratio
        else:
            # No adjustment (ULA)
            adjustment_mask = torch.ones(n_chains, dtype=torch.bool)
        x[adjustment_mask] = x_prime[adjustment_mask]

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