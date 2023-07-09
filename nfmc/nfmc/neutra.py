from nfmc.mcmc.hmc import hmc
from normalizing_flows import Flow


def neutra_hmc(flow: Flow,
               potential: callable,
               n_chains: int):
    # Fit flow to target via variational inference
    flow.variational_fit(potential)

    # Run HMC with target being the flow
    x0 = flow.sample(n_chains)
    z0, _ = flow.forward(x0)

    def adjusted_potential(latent_z):
        target_x, log_det_inverse = flow.inverse(latent_z)
        return potential(target_x) - log_det_inverse

    z = hmc(z0, adjusted_potential)
    x, _ = flow.inverse(z)
    return x
