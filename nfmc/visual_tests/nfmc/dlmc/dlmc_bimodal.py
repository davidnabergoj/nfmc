import matplotlib.pyplot as plt
import torch

from nfmc.nfmc.deterministic_langevin import dlmc
from normalizing_flows import Flow, RealNVP
from potentials.synthetic.gaussian import StandardGaussian, DiagonalGaussian

if __name__ == '__main__':
    torch.manual_seed(0)

    n_dim = 2
    n_chains = 1000  # Need a good NF fit

    negative_log_prior = DiagonalGaussian(mu=torch.tensor([-5.0, 0.0]), sigma=torch.ones(2)).cuda()
    negative_log_likelihood = DiagonalGaussian(mu=torch.tensor([5.0, 0.0]), sigma=torch.ones(2)).cuda()
    x_prior = negative_log_prior.sample(batch_shape=(n_chains,))


    def potential(x):
        return negative_log_prior(x) + negative_log_likelihood(x)


    flow = Flow(RealNVP(n_dim)).cuda()

    ret = dlmc(x_prior, potential, negative_log_likelihood, flow, step_size=0.1, full_output=True, n_iterations=10)
    print(f'{ret.shape = }')

    chain_id = 5
    with torch.no_grad():
        x_pr = negative_log_prior.sample((1000,)).cpu().numpy()
        x_lh = negative_log_likelihood.sample((1000,)).cpu().numpy()
        x_dlmc = ret[:, chain_id].cpu().numpy()

        fig, ax = plt.subplots()
        plt.scatter(x_pr[:, 0], x_pr[:, -1], label='Prior')
        plt.scatter(x_lh[:, 0], x_lh[:, -1], label='Likelihood')
        ax.plot(x_dlmc[:, 0], x_dlmc[:, -1], c='tab:green')
        plt.scatter(x_dlmc[:, 0], x_dlmc[:, -1], label=f'DLMC (chain {chain_id})', c='tab:green', ec='w', zorder=2)
        plt.scatter(x_dlmc[0, 0], x_dlmc[0, -1], c='tab:red', ec='w', zorder=3, marker='*', s=2 ** 8,
                    label='Initial state')
        ax.scatter(x_dlmc[-1, 0], x_dlmc[-1, -1], c='yellow', ec='k', zorder=3, marker='*', s=2 ** 8,
                   label='Final state')
        ax.legend()
        fig.tight_layout()
        plt.show()

        x_dlmc = ret.cpu()[-1].numpy()
        fig, ax = plt.subplots()
        plt.scatter(x_pr[:, 0], x_pr[:, -1], label='Prior')
        plt.scatter(x_lh[:, 0], x_lh[:, -1], label='Likelihood')
        plt.scatter(x_dlmc[:, 0], x_dlmc[:, -1], label='DLMC', alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()
