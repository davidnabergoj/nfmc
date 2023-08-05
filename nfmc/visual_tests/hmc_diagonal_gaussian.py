import matplotlib.pyplot as plt
import torch

from nfmc.mcmc.hmc import hmc
from potentials.synthetic.gaussian import DiagonalGaussian

if __name__ == '__main__':
    torch.manual_seed(0)
    n_chains, n_dim = 200, 2
    x0 = torch.randn(size=(n_chains, n_dim))
    mu = torch.zeros(n_dim)
    sigma = torch.linspace(-3, 3, n_dim).exp()
    potential = DiagonalGaussian(mu, sigma)
    print(x0[0])
    ret = hmc(x0, potential, full_output=True).detach()
    print(f'{ret.shape = }')

    gt = potential.sample((10000,))

    chain_id = 0
    plt.figure()
    plt.scatter(gt[:, 0], gt[:, -1], label='Ground truth', alpha=0.3)
    plt.plot(ret[:, chain_id, 0], ret[:, chain_id, -1], '-o', label='HMC chain', c='tab:orange')
    plt.legend()
    plt.tight_layout()
    plt.show()
