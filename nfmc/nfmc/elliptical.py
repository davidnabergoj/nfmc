import math
from copy import deepcopy

import torch

from normalizing_flows import Flow


def tess_base(u0, flow: Flow, potential: callable, max_iterations: int = 50):
    u = deepcopy(u0)
    v = torch.randn_like(u)
    w = torch.rand()
    log_s = -potential(u) + flow.log_prob(v) + w.log()
    theta = torch.rand() * (2 * math.pi)
    theta_min, theta_max = theta - 2 * math.pi, theta

    for i in range(max_iterations):
        u_prime = u * torch.cos(theta) + v * torch.sin(theta)
        v_prime = v * torch.cos(theta) + u * torch.sin(theta)

        if -potential(u_prime) + flow.log_prob(v_prime) > log_s:
            x_prime, _ = flow.inverse(u_prime)
            return x_prime, u_prime
        else:
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = torch.rand() * (theta_max - theta_min) + theta_min

    u_prime = u * torch.cos(theta) + v * torch.sin(theta)
    x_prime, _ = flow.inverse(u_prime)
    return x_prime, u_prime


def tess(u0: torch.Tensor,
         flow: Flow,
         potential: callable,
         n_warmup_iterations: int = 1000,
         n_sampling_iterations: int = 1000,
         full_output: bool = False):
    # Warmup
    u = deepcopy(u0)
    for i in range(n_warmup_iterations):
        x, u = tess_base(u, flow, potential)
        flow.fit(x)

    # Sampling
    xs = []
    x = None  # In case somebody uses n_sampling_iterations = 0
    for i in range(n_sampling_iterations):
        x, u = tess_base(u, flow, potential)
        if full_output:
            xs.append(x)
    
    if full_output:
        return torch.stack(xs)
    else:
        return x
