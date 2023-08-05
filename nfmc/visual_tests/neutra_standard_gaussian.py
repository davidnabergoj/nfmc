import torch

from nfmc.nfmc.neutra import neutra_hmc
from normalizing_flows import Flow, IAF
from potentials.synthetic.gaussian import StandardGaussian
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)

    n_dim = 100
    n_chains = 200
    potential = StandardGaussian(event_shape=(n_dim,))
    gt = potential.sample((10000,))
    flow = Flow(IAF(n_dim)).cuda()

    ret = neutra_hmc(flow, potential, n_chains).cpu()

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
    plt.scatter(xf[:, 0], xf[:, 1], label='Flow')
    plt.scatter(ret[chain_id, 0], ret[chain_id, 1], label='NeuTra HMC')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f'{ret.shape = }')
