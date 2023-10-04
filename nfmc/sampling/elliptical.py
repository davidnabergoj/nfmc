import math
from copy import deepcopy

import torch
from tqdm import tqdm

from normalizing_flows import Flow


def transport_elliptical_slice_sampling_helper(u, flow: Flow, potential: callable, max_iterations: int = 5):
    n_chains, *event_shape = u.shape

    log_phi = flow.base_log_prob

    @torch.no_grad()
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
        theta_padded = theta.view(n_chains, *[1 for _ in event_shape])
        assert len(theta_padded.shape) == len(u.shape) == len(v.shape)

        u_prime = u * torch.cos(theta_padded) + v * torch.sin(theta_padded)
        v_prime = v * torch.cos(theta_padded) - u * torch.sin(theta_padded)
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


def transport_elliptical_slice_sampling_base(u: torch.Tensor,
                                             flow: Flow,
                                             potential: callable,
                                             n_warmup_iterations: int = 100,
                                             n_sampling_iterations: int = 250,
                                             full_output: bool = True):
    # Warmup
    for _ in tqdm(range(n_warmup_iterations), desc='Warmup'):
        x, u = transport_elliptical_slice_sampling_helper(u, flow, potential)
        flow.fit(x.detach())

    # Sampling
    xs = []
    x = None  # In case somebody uses n_sampling_iterations = 0
    for _ in tqdm(range(n_sampling_iterations), desc='Sampling'):
        x, u = transport_elliptical_slice_sampling_helper(u, flow, potential)
        if full_output:
            xs.append(deepcopy(x.detach()))

    if full_output:
        return torch.stack(xs)
    else:
        return x
