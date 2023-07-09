from typing import Any

import torch

from normalizing_flows import Flow


def ns(n_live_points: int,
       prior: Any,
       log_likelihood: callable,
       flow: Flow,
       n_iterations: int = 1000,
       latent_scale: float = 1.0,  # instead of radius
       n_population_draws: int = 50):
    n_rs_iterations = 2 * n_live_points

    x = prior.sample(n_live_points)
    n_dim = x.shape[-1]

    for i in range(n_iterations):
        x_ll = log_likelihood(x)
        worst_idx = torch.argmin(x_ll)

        if i < n_rs_iterations:
            worst_replacement = rejection_sampling(prior)
        else:
            z_worst, _ = flow.forward(x[worst_idx])
            z_replacements = torch.randn(n_population_draws, n_dim) * latent_scale + z_worst
            x_replacements, _ = flow.inverse(z_replacements)
            xr_ll = log_likelihood(x_replacements)
            candidates = x_replacements[xr_ll > x_ll[worst_idx]]
            worst_replacement = candidates[torch.randint(low=0, high=len(candidates), size=())]
        x[worst_idx] = worst_replacement
        if i % n_live_points == 0:
            flow.fit(x)
