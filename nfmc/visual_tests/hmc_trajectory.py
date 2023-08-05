import torch
import matplotlib.pyplot as plt
from nfmc.mcmc.hmc import hmc_trajectory
from potentials.synthetic.gaussian import StandardGaussian

if __name__ == '__main__':
    torch.manual_seed(0)
    n_dim = 100
    potential = StandardGaussian(event_shape=(n_dim,))
    x = torch.randn(1, n_dim)
    momentum = torch.randn_like(x)
    x, _, _ = hmc_trajectory(x, momentum, step_size=0.1, n_leapfrog_steps=100, potential=potential, full_output=True)
    x = x.detach()
    print(f'{x.shape = }')

    gt = potential.sample((1000,))

    plt.figure()
    chain_id = 0
    plt.scatter(gt[:, 0], gt[:, 1], label='Ground truth')
    plt.plot(x[:, chain_id, 0], x[:, chain_id, 1], '-o', label='Trajectory', c='tab:orange')
    plt.legend()
    plt.tight_layout()
    plt.show()
