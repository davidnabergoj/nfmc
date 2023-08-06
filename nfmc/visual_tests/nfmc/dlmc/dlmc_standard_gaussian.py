import torch

from nfmc.nfmc.deterministic_langevin import dlmc
from normalizing_flows import Flow, RealNVP
from potentials.synthetic.gaussian import StandardGaussian, DiagonalGaussian

if __name__ == '__main__':
    torch.manual_seed(0)

    # Prior = N(5, I)
    n_dim = 2
    n_chains = 100

    negative_log_prior = DiagonalGaussian(mu=torch.zeros(n_dim) + 5, sigma=torch.ones(n_dim)).cuda()
    negative_log_likelihood = StandardGaussian(event_shape=(n_dim,)).cuda()
    x_prior = negative_log_prior.sample(batch_shape=(n_chains,))


    def potential(x):
        return negative_log_prior(x) + negative_log_likelihood(x)


    flow = Flow(RealNVP(n_dim)).cuda()

    ret = dlmc(x_prior, potential, negative_log_likelihood, flow, step_size=0.01)
    print(f'{ret = }')
