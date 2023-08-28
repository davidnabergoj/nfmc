import torch

from nfmc.mcmc.hmc import hmc
from normalizing_flows import Flow


def neutra_hmc(flow: Flow,
               potential: callable,
               n_chains: int,
               n_vi_iterations: int = 100,
               n_hmc_iterations: int = 100):
    # Fit flow to target via variational inference
    print('Fitting NF')
    flow.variational_fit(potential, n_epochs=n_vi_iterations)

    # Run HMC with target being the flow
    x0 = flow.sample(n_chains)
    z0, _ = flow.bijection.forward(x0)

    def adjusted_potential(_z):
        _x, log_det_inverse = flow.bijection.inverse(_z)
        log_prob = -potential(_x)
        adjusted_log_prob = log_prob + log_det_inverse
        return -adjusted_log_prob

    print('Running HMC')
    z = hmc(z0.detach(), adjusted_potential, full_output=True, n_iterations=n_hmc_iterations)

    print('Generating final samples')
    with torch.no_grad():
        x, _ = flow.bijection.batch_inverse(z, batch_size=128)
        x = x.detach()
    return x
