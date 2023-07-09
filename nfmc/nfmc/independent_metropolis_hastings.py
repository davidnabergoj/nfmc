from copy import deepcopy

import torch

from normalizing_flows import Flow


def imh(x0: torch.Tensor,
        flow: Flow,
        potential: callable,
        n_iterations: int = 1000,
        full_output: bool = False,
        adaptation_dropoff: float = 0.99):
    # Exponentially diminishing adaptation probability sequence

    xs = []

    n_chains, n_dim = x0.shape
    x = deepcopy(x0)
    for i in range(n_iterations):
        x_proposed = flow.sample(n_chains)
        log_u = torch.rand(n_chains).log()
        log_alpha = (
                + potential(x)
                - potential(x_proposed)
                + flow.log_prob(x)
                - flow.log_prob(x_proposed)
        )
        accepted_mask = torch.where(torch.less(log_u, log_alpha))
        x[accepted_mask] = x_proposed[accepted_mask]

        xs.append(deepcopy(x))

        u_prime = torch.rand()
        alpha_prime = adaptation_dropoff ** i
        if u_prime < alpha_prime:
            k = int(torch.randint(low=0, high=len(xs), size=()))
            x_train = xs[k]
            flow.fit(x_train, n_epochs=1)

    if full_output:
        return torch.stack(xs)
    else:
        return x
