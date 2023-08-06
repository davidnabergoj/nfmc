import torch

from nfmc.nfmc.neutra import neutra_hmc
from normalizing_flows import Flow, IAF
from potentials.synthetic.gaussian import DiagonalGaussian
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)

    n_dim = 2
    n_chains = 200
    mu = torch.zeros(n_dim)
    sigma = torch.linspace(-3, 3, n_dim).exp()
    potential = DiagonalGaussian(mu, sigma).cuda()
    gt = potential.sample((10000,)).cpu()
    flow = Flow(IAF(n_dim, n_layers=3)).cuda()

    ret = neutra_hmc(flow, potential, n_chains).cpu()
    print(f'{ret.shape = }')

    xf = flow.sample(1000).detach().cpu()
    plt.figure()
    plt.scatter(gt[:, 0], gt[:, 1], label='Ground truth', alpha=0.5)
    plt.scatter(xf[:, 0], xf[:, 1], label='Flow', s=5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    chain_id = 0

    plt.figure()
    plt.scatter(gt[:, 0], gt[:, 1], label='Ground truth')
    plt.scatter(ret[:, chain_id, 0], ret[:, chain_id, 1], label='NeuTra HMC')
    plt.legend()
    plt.tight_layout()
    plt.show()
