import torch
import matplotlib.pyplot as plt
from potentials.synthetic.gaussian import StandardGaussian
from nfmc.mcmc.langevin_algorithm import mala

if __name__ == '__main__':
    torch.manual_seed(0)
    n_dim = 100
    n_chains = 100

    x0 = torch.randn(size=(n_chains, n_dim))
    target = StandardGaussian(event_shape=(n_dim,))

    ret = mala(x0, target, full_output=True, n_iterations=5000)
    print(f'{ret.shape = }')

    gt = target.sample(batch_shape=(10000,))

    plt.figure()
    plt.scatter(gt[..., 0], gt[..., 1], label='Ground truth', alpha=0.3)
    plt.scatter(ret[:, 0, 0], ret[:, 0, 1], label='MALA samples')
    plt.legend()
    plt.tight_layout()
    # lim = 50
    # plt.xlim(-lim, lim)
    # plt.ylim(-lim, lim)
    plt.show()
