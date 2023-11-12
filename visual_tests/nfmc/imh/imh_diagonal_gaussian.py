import torch
import matplotlib.pyplot as plt
from nfmc.nfmc.independent_metropolis_hastings import imh
from normalizing_flows import Flow, RealNVP
from potentials.synthetic.gaussian import DiagonalGaussian

if __name__ == '__main__':
    torch.manual_seed(0)
    n_chains = 100
    n_dim = 2
    mu = torch.tensor([5.0, 5.0])
    sigma = torch.ones(2)
    potential = DiagonalGaussian(mu, sigma).cuda()
    x0 = torch.randn(size=(n_chains, n_dim)).cuda()

    flow = Flow(RealNVP(n_dim, n_layers=3)).cuda()  # Many layers make the flow unstable
    ret = imh(x0, flow, potential, full_output=True, n_iterations=3000).cpu()

    print(f'{ret.shape = }')

    gt = potential.sample((1000,)).cpu()
    xf = flow.sample(1000, no_grad=True).cpu()
    d0, d1 = 0, -1
    chain_id = 0

    fig, ax = plt.subplots()
    ax.scatter(gt[:, d0], gt[:, d1], label='Ground truth', s=6)
    ax.scatter(ret[:, chain_id, d0], ret[:, chain_id, d1], label='IMH', s=6)
    ax.scatter(xf[:, d0], xf[:, d1], label='Flow', s=6, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()
