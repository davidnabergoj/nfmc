from copy import deepcopy

import torch


def hmc(x0: torch.Tensor,
        potential: callable,
        n_iterations: int = 1000,
        n_leapfrog_steps: int = 15,
        step_size: int = 1.0,
        full_output: bool = False):
    x = deepcopy(x0)
    xs = [x0]
    for i in range(n_iterations):
        x_proposed = deepcopy(x)
        initial_momentum = torch.randn_like(x_proposed)
        momentum = deepcopy(initial_momentum)
        for j in range(n_leapfrog_steps):
            grad = torch.autograd.grad(potential(x_proposed).sum(), x_proposed)[0]
            momentum = momentum - step_size / 2 * grad
            x_proposed = x_proposed + step_size * momentum
            grad = torch.autograd.grad(potential(x_proposed).sum(), x_proposed)[0]
            momentum = momentum - step_size / 2 * grad

        with torch.no_grad():
            log_alpha = (
                    - potential(x_proposed)
                    - 0.5 * torch.linalg.norm(momentum, dim=1) ** 2
                    + potential(x)
                    + 0.5 * torch.linalg.norm(initial_momentum, dim=1) ** 2
            )
            log_u = torch.rand_like(log_alpha).log()
            accepted_mask = torch.where(torch.less(log_u, log_alpha))
            x[accepted_mask] = x_proposed[accepted_mask]

            if full_output:
                xs.append(deepcopy(x))

    if full_output:
        return torch.stack(xs)
    else:
        return x
