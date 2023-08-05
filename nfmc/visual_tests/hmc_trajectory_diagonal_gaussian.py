import torch
import matplotlib.pyplot as plt
from nfmc.mcmc.hmc import hmc_trajectory
from potentials.synthetic.gaussian import StandardGaussian, DiagonalGaussian

if __name__ == '__main__':
    torch.manual_seed(0)
    n_dim = 2
    mu = torch.zeros(n_dim)
    sigma = torch.linspace(-3, 3, n_dim).exp()
    m = 1 / sigma[None] ** 2
    potential = DiagonalGaussian(mu, sigma)
    x = torch.tensor([[0.5, 40.0]])
    print(f'{x = }')
    noise = torch.randn_like(x)
    print(f'{noise = }')
    momentum = noise * torch.sqrt(m)
    print(f'{momentum = }')
    x, _, _ = hmc_trajectory(x, momentum, step_size=0.1, n_leapfrog_steps=100, potential=potential, full_output=True,
                             mass_diag=m)
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
