import torch

from nfmc.mcmc.hmc import hmc
from normalizing_flows import Flow


def neutra_hmc_base(flow: Flow,
                    potential: callable,
                    n_chains: int,
                    n_vi_iterations: int = 100,
                    n_hmc_iterations: int = 100,
                    show_progress: bool = True):
    # Fit flow to target via variational inference
    flow.variational_fit(lambda v: -potential(v), n_epochs=n_vi_iterations, show_progress=show_progress)

    # Run HMC with target being the flow
    x0 = flow.sample(n_chains)
    z0, _ = flow.bijection.forward(x0)

    def adjusted_potential(_z):
        _x, log_det_inverse = flow.bijection.inverse(_z)
        log_prob = -potential(_x)
        adjusted_log_prob = log_prob + log_det_inverse
        return -adjusted_log_prob

    z = hmc(
        z0.detach(),
        adjusted_potential,
        full_output=True,
        n_iterations=n_hmc_iterations,
        show_progress=show_progress
    )

    with torch.no_grad():
        x, _ = flow.bijection.batch_inverse(z, batch_size=128)
        x = x.detach()
    return x
