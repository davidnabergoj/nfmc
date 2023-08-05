import torch

from nfmc.mcmc.hmc import hmc
from normalizing_flows import Flow


def neutra_hmc(flow: Flow,
               potential: callable,
               n_chains: int):
    # Fit flow to target via variational inference
    print('Fitting NF')
    flow.variational_fit(potential)

    # Run HMC with target being the flow
    x0 = flow.sample(n_chains)
    z0, _ = flow.bijection.forward(x0)

    def adjusted_potential(latent_z):
        target_x, log_det_inverse = flow.bijection.inverse(latent_z)
        adjusted_log_prob = -potential(target_x) + log_det_inverse
        return -adjusted_log_prob

    print('Running HMC')
    z = hmc(z0.detach(), adjusted_potential, full_output=True)

    print('Generating final samples')
    with torch.no_grad():
        x, _ = flow.bijection.batch_inverse(z, batch_size=128)
        x = x.detach()
    return x
