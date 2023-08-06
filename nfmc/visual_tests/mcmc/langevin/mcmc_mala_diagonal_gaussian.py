import torch
import matplotlib.pyplot as plt
from potentials.synthetic.gaussian import DiagonalGaussian
from nfmc.mcmc.langevin_algorithm import mala, ula

if __name__ == '__main__':
    """
    Note: more NF layers make its distribution broader.
    """

    torch.manual_seed(0)
    n_dim = 2
    n_chains = 100

    mu = torch.ones(n_dim)
    sigma = torch.linspace(-3, 3, n_dim).exp()

    x0 = torch.randn(size=(n_chains, n_dim))
    target = DiagonalGaussian(mu, sigma)

    ret = mala(x0, target, full_output=True, tau=1e-3, n_iterations=10000)
    gt = target.sample(batch_shape=(10000,))

    print(f'{x0[0] = }')

    plt.figure()
    plt.scatter(gt[..., 0], gt[..., 1], label='Ground truth', alpha=0.3)
    plt.scatter(ret[:, 0, 0], ret[:, 0, 1], label='MALA samples')
    plt.legend()
    plt.tight_layout()
    # lim = 50
    # plt.xlim(-lim, lim)
    # plt.ylim(-lim, lim)
    plt.show()
