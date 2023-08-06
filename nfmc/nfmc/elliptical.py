import math
from copy import deepcopy

import torch
from tqdm import tqdm

from normalizing_flows import Flow


def tess_base(u, flow: Flow, potential: callable, max_iterations: int = 5):
    n_chains, n_dim = u.shape

    log_phi = flow.base.log_prob

    def log_pi_hat(latent):
        target, log_det = flow.bijection.inverse(latent)
        return -potential(target) + log_det

    v = torch.randn_like(u)
    w = torch.rand(size=(n_chains,))
    log_s = log_pi_hat(u) + log_phi(v) + w.log()
    theta = torch.randn_like(w) * (2 * math.pi)
    theta_min, theta_max = theta - 2 * math.pi, theta

    finished = torch.zeros(size=(n_chains,), dtype=torch.bool)

    u_prime_final = torch.zeros_like(u)
    x_prime_final = torch.zeros_like(u)

    # In case somebody uses max_iterations = 0
    u_prime = torch.zeros_like(u)
    x_prime = torch.zeros_like(u)
    for i in range(max_iterations):
        u_prime = u * torch.cos(theta.view(n_chains, 1)) + v * torch.sin(theta.view(n_chains, 1))
        v_prime = v * torch.cos(theta.view(n_chains, 1)) - u * torch.sin(theta.view(n_chains, 1))
        x_prime, _ = flow.bijection.inverse(u_prime)

        finished_update_mask = ((log_pi_hat(u_prime) + log_phi(v_prime)) > log_s)
        # We update the outputs the first time this criterion is satisfied
        x_prime_final[finished_update_mask & (~finished)] = x_prime[finished_update_mask & (~finished)]
        u_prime_final[finished_update_mask & (~finished)] = u_prime[finished_update_mask & (~finished)]

        finished |= finished_update_mask

        mask = (theta < 0)
        theta_min[mask] = theta[mask]
        theta_max[~mask] = theta[~mask]
        theta = torch.rand(size=(n_chains,)) * (theta_max - theta_min) + theta_min

    # print(f'{float(torch.mean(finished.float())):.2f}')

    x_prime_final[~finished] = x_prime[~finished]
    u_prime_final[~finished] = u_prime[~finished]

    return x_prime_final.detach(), u_prime_final.detach()


def tess(u: torch.Tensor,
         flow: Flow,
         potential: callable,
         n_warmup_iterations: int = 100,
         n_sampling_iterations: int = 250,
         full_output: bool = False):
    # Warmup
    for _ in tqdm(range(n_warmup_iterations), desc='Warmup'):
        x, u = tess_base(u, flow, potential)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(x[:, 0], x[:, 1])
        # plt.xlim(-10, 10)
        # plt.ylim(-10, 10)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        flow.fit(x.detach())

    # Sampling
    xs = []
    x = None  # In case somebody uses n_sampling_iterations = 0
    for _ in tqdm(range(n_sampling_iterations), desc='Sampling'):
        x, u = tess_base(u, flow, potential)
        if full_output:
            xs.append(deepcopy(x.detach()))

    if full_output:
        return torch.stack(xs)
    else:
        return x
