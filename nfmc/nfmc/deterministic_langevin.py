import torch

from normalizing_flows import Flow


def dlmc(x_prior: torch.Tensor,
         potential: callable,
         negative_log_likelihood: callable,
         flow: Flow,
         step_size: float = 1.0,
         n_iterations: int = 1000,
         latent_updates: bool = False):
    n_chains, n_dim = x_prior.shape

    # Initial update
    grad = torch.autograd.grad(
        negative_log_likelihood(x_prior).sum(),
        x_prior
    )[0]
    x = x_prior + step_size * grad

    for i in range(n_iterations):
        flow.fit(x)
        if latent_updates:
            z, _ = flow.forward(x)
            grad = torch.autograd.grad(potential(x), x)[0]
            z = z - step_size * (grad - z)
            x, _ = flow.inverse(z)
        else:
            grad = torch.autograd.grad(
                potential(x) + flow.log_prob(x),
                x
            )[0]
            x = x - step_size * grad

        x_tilde = flow.sample(n_chains)
        log_u = torch.rand(n_chains).log()
        log_alpha = (
                + potential(x)
                - potential(x_tilde)
                + flow.log_prob(x)
                - flow.log_prob(x_tilde)
        )
        accepted_mask = torch.where(torch.less(log_u, log_alpha))
        x[accepted_mask] = x_tilde[accepted_mask]

    return x
