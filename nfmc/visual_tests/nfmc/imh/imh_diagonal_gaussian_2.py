import torch
import matplotlib.pyplot as plt
from nfmc.nfmc.independent_metropolis_hastings import imh
from nfmc.util import relative_error
from normalizing_flows import Flow, RealNVP
from potentials.synthetic.gaussian import DiagonalGaussian

if __name__ == '__main__':
    # Bounded geometric train distribution is better than the uniform.

    torch.manual_seed(0)
    n_chains = 100  # Worse geometry needs more chains
    n_dim = 2
    mu = torch.tensor([0.0, 0.0])
    sigma = torch.tensor([1e-4, 1e4])
    potential = DiagonalGaussian(mu, sigma).cuda()
    x0 = torch.randn(size=(n_chains, n_dim)).cuda()

    torch.manual_seed(0)
    flow = Flow(RealNVP(n_dim, n_layers=3)).cuda()
    ret_geom_approx = imh(x0, flow, potential, full_output=True, n_iterations=500,
                          train_dist='bounded_geom_approx').cpu()

    gt = potential.sample((1000,)).cpu()
    xf = flow.sample(1000, no_grad=True).cpu()
    d0, d1 = 0, -1
    chain_id = 0

    burnin_idx = 150

    #############################################################

    fig, ax = plt.subplots()
    ax.scatter(ret_geom_approx[burnin_idx:, :, d0].reshape(-1, n_dim),
               ret_geom_approx[burnin_idx:, :, d1].reshape(-1, n_dim),
               label='IMH (approx geometric)', s=6)
    ax.scatter(gt[:, d0], gt[:, d1], label='Ground truth', s=6)
    # ax.scatter(xf[:, d0], xf[:, d1], label='Flow', s=6, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()

    #############################################################

    torch.manual_seed(0)
    flow = Flow(RealNVP(n_dim, n_layers=3)).cuda()  # Many layers make the flow unstable
    ret_unif = imh(x0, flow, potential, full_output=True, n_iterations=500, train_dist='uniform').cpu()

    fig, ax = plt.subplots()
    ax.scatter(ret_unif[burnin_idx:, :, d0].reshape(-1, n_dim), ret_unif[burnin_idx:, :, d1].reshape(-1, n_dim),
               label='IMH (uniform)', s=6)
    ax.scatter(gt[:, d0], gt[:, d1], label='Ground truth', s=6)
    ax.legend()
    fig.tight_layout()
    plt.show()

    #############################################################

    torch.manual_seed(0)
    flow = Flow(RealNVP(n_dim, n_layers=3)).cuda()
    ret_geom = imh(x0, flow, potential, full_output=True, n_iterations=500, train_dist='bounded_geom').cpu()

    fig, ax = plt.subplots()
    ax.scatter(ret_geom[burnin_idx:, :, d0].reshape(-1, n_dim), ret_geom[burnin_idx:, :, d1].reshape(-1, n_dim),
               label='IMH (geometric)', s=6)
    ax.scatter(gt[:, d0], gt[:, d1], label='Ground truth', s=6)
    ax.legend()
    fig.tight_layout()
    plt.show()

    #############################################################

    fig, ax = plt.subplots()
    ax.scatter(gt[:, d0], gt[:, d1], label='Ground truth', s=6)
    ax.scatter(ret_unif[burnin_idx:, chain_id, d0], ret_unif[burnin_idx:, chain_id, d1], label='IMH (uniform)', s=6)
    ax.scatter(ret_geom[burnin_idx:, chain_id, d0], ret_geom[burnin_idx:, chain_id, d1], label='IMH (geometric)', s=6)
    ax.scatter(ret_geom_approx[burnin_idx:, chain_id, d0], ret_geom_approx[burnin_idx:, chain_id, d1],
               label='IMH (approx geometric)', s=6)
    ax.legend()
    fig.tight_layout()
    plt.show()

    print(f'Uniform, variance relative error: {relative_error(sigma ** 2, ret_unif[burnin_idx:].var(dim=(0, 1)))}')
    print(f'Approx geometric, variance relative error: {relative_error(sigma ** 2, ret_geom_approx[burnin_idx:].var(dim=(0, 1)))}')
    print(f'Geometric, variance relative error: {relative_error(sigma ** 2, ret_geom[burnin_idx:].var(dim=(0, 1)))}')

    print(f'Uniform, variance: {ret_unif[burnin_idx:].var(dim=(0, 1))}')
    print(f'Approx geometric, variance: {ret_geom_approx[burnin_idx:].var(dim=(0, 1))}')
    print(f'Geometric, variance: {ret_geom[burnin_idx:].var(dim=(0, 1))}')
