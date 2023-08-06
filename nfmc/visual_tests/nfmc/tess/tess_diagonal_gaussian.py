import torch
import matplotlib.pyplot as plt

from nfmc.nfmc.elliptical import tess
from normalizing_flows import Flow, RealNVP
from potentials.synthetic.gaussian import DiagonalGaussian

if __name__ == '__main__':
    torch.manual_seed(0)
    n_dim = 2
    n_chains = 100

    x0 = torch.randn(size=(n_chains, n_dim))
    mu = torch.zeros(n_dim)
    sigma = torch.tensor([1e-01, 1e+01])
    target = DiagonalGaussian(mu, sigma)
    flow = Flow(RealNVP(n_dim, n_layers=3))

    ret = tess(x0, flow, target, full_output=True)
    print(f'{ret.shape = }')

    gt = target.sample(batch_shape=(10000,))
    xf = flow.sample(1000, no_grad=True).detach().cpu()

    plt.figure()
    plt.scatter(gt[..., 0], gt[..., 1], label='Ground truth', alpha=0.3)
    plt.scatter(ret[:, :, 0].view(-1, n_dim), ret[:, :, 1].view(-1, n_dim), label='TESS', s=8)
    # plt.plot(ret[:, 0, 0], ret[:, 0, 1], '-o', label='TESS chain 0', c='tab:red')
    plt.scatter(xf[:, 0], xf[:, 1], alpha=0.2, label='Flow')
    plt.legend()
    plt.tight_layout()
    plt.show()
