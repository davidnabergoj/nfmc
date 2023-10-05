from copy import deepcopy

import torch

from nfmc.util import metropolis_acceptance_log_ratio, compute_grad
from normalizing_flows import Flow


def deterministic_langevin_monte_carlo_base(x_prior: torch.Tensor,
                                            potential: callable,
                                            negative_log_likelihood: callable,
                                            flow: Flow,
                                            step_size: float = 1.0,
                                            n_iterations: int = 20,
                                            latent_updates: bool = False,
                                            full_output: bool = False):
    # FIXME latent = True does not work
    n_chains, *event_shape = x_prior.shape

    # Initial update
    grad = compute_grad(negative_log_likelihood, x_prior)
    x = x_prior - step_size * grad
    x.requires_grad_(False)

    xs = [x_prior, x]
    for i in range(n_iterations):
        flow.fit(x.detach(), n_epochs=100)
        if latent_updates:
            z, _ = flow.bijection.forward(x)
            grad = compute_grad(potential, x)
            z = z - step_size * (grad - z)
            x, _ = flow.bijection.inverse(z)
        else:
            grad = compute_grad(lambda v: potential(v) + flow.log_prob(v), x)
            x = x - step_size * grad

        x_tilde = flow.sample(n_chains, no_grad=True)
        log_alpha = metropolis_acceptance_log_ratio(
            log_prob_curr=-potential(x),
            log_prob_prime=-potential(x_tilde),
            log_proposal_curr=flow.log_prob(x),
            log_proposal_prime=flow.log_prob(x_tilde)
        )
        log_u = torch.rand(n_chains).log().to(log_alpha)
        accepted_mask = torch.where(torch.less(log_u, log_alpha))
        x[accepted_mask] = x_tilde[accepted_mask]
        x = x.detach()

        if full_output:
            xs.append(deepcopy(x.detach()))

    if full_output:
        return torch.stack(xs)
    return x
