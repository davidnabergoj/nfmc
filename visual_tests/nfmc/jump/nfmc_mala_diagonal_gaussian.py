import torch
import matplotlib.pyplot as plt
from normalizing_flows import Flow, RealNVP
from normalizing_flows.src.bijections.finite.linear import PositiveDiagonal
from potentials.synthetic.gaussian import DiagonalGaussian
from nfmc.nfmc.langevin_algorithm import mala

if __name__ == '__main__':
    """
    Note: more NF layers make its distribution broader.
    """

    torch.manual_seed(0)
    n_dim = 2
    n_chains = 100
    jump_period = 100

    mu = torch.ones(n_dim)
    sigma = torch.linspace(-3, 3, n_dim).exp()

    x0 = torch.randn(size=(n_chains, n_dim))
    target = DiagonalGaussian(mu, sigma)
    flow = Flow(RealNVP(n_dim, n_layers=3))
    # flow = Flow(PositiveDiagonal(event_shape=(n_dim,)))

    ret = mala(x0, flow, target, full_output=True, jump_period=jump_period, nf_adjustment=True)
    print(f'{ret.shape = }')
    nf_resample_mask = (torch.arange(1, len(ret) + 1) % jump_period) == 0

    gt = target.sample(batch_shape=(10000,))

    chain_id = 1

    plt.figure()
    plt.scatter(gt[..., 0], gt[..., 1], label='Ground truth', alpha=0.3)
    plt.scatter(ret[~nf_resample_mask, chain_id, 0], ret[~nf_resample_mask, chain_id, 1], label='MALA samples')
    plt.scatter(ret[nf_resample_mask, chain_id, 0], ret[nf_resample_mask, chain_id, 1], label='NF jumps', ec='w')
    plt.legend()
    plt.tight_layout()
    # lim = 50
    # plt.xlim(-lim, lim)
    # plt.ylim(-lim, lim)
    plt.show()

    # Plot the first chain from the beginning until the second jump
    plt.figure()
    plt.plot(ret[:2 * jump_period, chain_id, 0], ret[:2 * jump_period, chain_id, 1], '-o', label='Chain trajectory')
    plt.scatter(
        ret[:2 * jump_period][::jump_period, chain_id, 0],
        ret[:2 * jump_period][::jump_period, chain_id, 1],
        label='Jump',
        c='tab:orange',
        zorder=2
    )
    plt.legend()
    plt.show()
