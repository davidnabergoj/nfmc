import torch
import matplotlib.pyplot as plt
from normalizing_flows import Flow, RealNVP
from potentials import IllConditionedGaussian
from nfmc.nfmc.langevin_algorithm import mala

if __name__ == '__main__':
    # This doesn't work well because MALA dynamics are not good enough for this target.
    # HMC/MALT/MCHMC would probably work better.

    torch.manual_seed(0)
    n_dim = 100
    n_chains = 100
    jump_period = 100

    x0 = torch.randn(size=(n_chains, n_dim))
    target = IllConditionedGaussian(n_dim)
    flow = Flow(RealNVP(n_dim, n_layers=3))

    ret = mala(x0, flow, target, full_output=True, jump_period=jump_period)
    print(f'{ret.shape = }')
    nf_resample_mask = (torch.arange(1, len(ret) + 1) % jump_period) == 0
    ret = ret[:, :4, :]  # first four chains only

    gt = target.sample(batch_shape=(10000,))

    plt.figure()
    plt.scatter(gt[..., 0], gt[..., -1], label='Ground truth', alpha=0.3)
    plt.scatter(ret[~nf_resample_mask, :, 0], ret[~nf_resample_mask, :, -1], label='MALA samples')
    plt.scatter(ret[nf_resample_mask, :, 0], ret[nf_resample_mask, :, -1], label='NF jumps', ec='w')
    plt.legend()
    plt.tight_layout()
    plt.show()
