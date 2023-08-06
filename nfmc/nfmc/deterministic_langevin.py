from copy import deepcopy

import torch

from nfmc.util import metropolis_acceptance_log_ratio, compute_grad
from normalizing_flows import Flow


def dlmc(x_prior: torch.Tensor,
         potential: callable,
         negative_log_likelihood: callable,
         flow: Flow,
         step_size: float = 1.0,
         n_iterations: int = 100,
         latent_updates: bool = False,
         full_output: bool = False):
    n_chains, n_dim = x_prior.shape

    # Initial update
    grad = compute_grad(negative_log_likelihood, x_prior)
    x = x_prior + step_size * grad

    xs = []
    for i in range(n_iterations):
        print(f'{i = }')
        flow.fit(x.detach())
        if latent_updates:
            z, _ = flow.forward(x)
            grad = compute_grad(potential, x)
            z = z - step_size * (grad - z)
            x, _ = flow.inverse(z)
        else:
            grad = compute_grad(lambda v: potential(v) + flow.log_prob(v), x)
            x = x - step_size * grad

        x_tilde = flow.sample(n_chains)
        log_alpha = metropolis_acceptance_log_ratio(
            log_prob_curr=-potential(x),
            log_prob_prime=-potential(x_tilde),
            log_proposal_curr=flow.log_prob(x),
            log_proposal_prime=flow.log_prob(x_tilde)
        )
        log_u = torch.rand(n_chains).log().to(log_alpha)
        accepted_mask = torch.where(torch.less(log_u, log_alpha))
        x[accepted_mask] = x_tilde[accepted_mask]

        if full_output:
            xs.append(deepcopy(x))

    if full_output:
        return xs
    return x
