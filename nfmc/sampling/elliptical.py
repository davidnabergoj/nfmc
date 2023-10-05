import math
from copy import deepcopy

import torch
from tqdm import tqdm

from potentials.base import Potential
from normalizing_flows import Flow
from normalizing_flows.utils import get_batch_shape


def elliptical_slice_sampling_step(
        f: torch.Tensor,
        target: Potential,
        cov: torch.Tensor = None,
        max_iterations: int = 5,
):
    """
    :param f: current state with shape (*batch_shape, *event_shape).
    :param target: negative log likelihood callable which receives as input a tensor with shape
        (*batch_shape, *event_shape) and outputs negative log likelihood values with shape batch_shape.
    :param cov: covariance matrix for the prior with shape (event_size, event_size) where event_size is
        equal to the product of elements of event_shape. If None, the covariance is assumed to be identity.
    :param max_iterations: maximum number of iterations where the proposal bracket is shrunk.
    """
    batch_shape = get_batch_shape(f, target.event_shape)

    # 1. Choose ellipse
    if cov is None:
        nu = torch.randn_like(f)
    else:
        event_size = int(torch.prod(torch.as_tensor(target.event_shape)))
        assert cov.shape == (event_size, event_size)
        nu_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(event_size), covariance_matrix=cov)
        nu_flat = nu_dist.sample(batch_shape)
        nu = nu_flat.view(*batch_shape, *target.event_shape)

    # 2. Log-likelihood threshold
    u = torch.rand(size=batch_shape)
    log_y = -target(f) + torch.log(u)

    # 3. Draw an initial proposal, also defining a bracket
    theta = torch.rand(size=[*batch_shape, *([1] * len(target.event_shape))]) * 2 * torch.pi
    theta_min = theta - 2 * torch.pi
    theta_max = theta

    accepted_mask = torch.zeros(size=batch_shape, dtype=torch.bool)
    f_proposed = f
    for _ in range(max_iterations):
        f_prime = f * torch.cos(theta) + nu * torch.sin(theta)
        accepted_mask_update = -target(f_prime) > log_y

        # Must have been accepted now and not previously
        f_proposed[accepted_mask_update & (~accepted_mask)] = f_prime[accepted_mask_update & (~accepted_mask)]

        # Update theta (we overwrite old thetas as they are unnecessary)
        theta_mask = theta < 0  # To avoid overwriting, we would consider the global accepted mask here
        theta_min[theta_mask] = theta[theta_mask]
        theta_max[~theta_mask] = theta[~theta_mask]

        # Draw new theta uniformly from [theta_min, theta_max]
        uniform_noise = torch.rand(size=[*batch_shape, *([1] * len(target.event_shape))])
        theta = uniform_noise * (theta_max - theta_min) + theta_min

        # Update the global accepted mask
        accepted_mask |= accepted_mask_update

    return f_proposed, accepted_mask


def elliptical_slice_sampler(
        target: Potential,
        n_chains: int = 100,
        n_iterations: int = 1000,
        cov: torch.Tensor = None,
        show_progress: bool = False,
        **kwargs
):
    """
    A sampling method that samples from a posterior, defined by a likelihood and a multivariate normal prior.

    :param target: negative log likelihood callable which receives as input a tensor with shape
        (*batch_shape, *event_shape) and outputs negative log likelihood values with shape batch_shape.
    :param n_chains: number of independent parallel chains.
    :param n_iterations: number of iterations.
    :param cov: covariance matrix for the prior with shape (event_size, event_size) where event_size is
        equal to the product of elements of event_shape. If None, the covariance is assumed to be identity.
    :param show_progress: optionally show a progress bar.
    :param kwargs: keyword arguments for the slice sampling step.
    """
    if cov is None:
        x = torch.randn(size=(n_chains, *target.event_shape))
    else:
        event_size = int(torch.prod(torch.as_tensor(target.event_shape)))
        assert cov.shape == (event_size, event_size)
        dist = torch.distributions.MultivariateNormal(loc=torch.zeros(event_size), covariance_matrix=cov)
        x_flat = dist.sample(sample_shape=torch.Size((n_chains,)))
        x = x_flat.view(n_chains, *target.event_shape)
    draws = []

    if show_progress:
        iterator = tqdm(range(n_iterations), desc="Elliptical slice sampling")
    else:
        iterator = range(n_iterations)

    for _ in iterator:
        x, accepted_mask = elliptical_slice_sampling_step(x, target, cov=cov, **kwargs)
        draws.append(torch.clone(x))
        if show_progress:
            acceptance_rate = float(torch.mean(torch.as_tensor(accepted_mask, dtype=torch.float)))
            iterator.set_postfix_str(f'accept-rate: {acceptance_rate:.4f}')
    return torch.stack(draws)


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
                                             n_epochs: int = 10,
                                             full_output: bool = True):
    # Warmup
    for _ in tqdm(range(n_warmup_iterations), desc='TESS warmup and training'):
        x, u = transport_elliptical_slice_sampling_helper(u, flow, potential)
        flow.fit(x.detach(), n_epochs=n_epochs)

    # Sampling
    xs = []
    x = None  # In case somebody uses n_sampling_iterations = 0
    for _ in tqdm(range(n_sampling_iterations), desc='Transport elliptical slice sampling'):
        x, u = transport_elliptical_slice_sampling_helper(u, flow, potential)
        if full_output:
            xs.append(deepcopy(x.detach()))

    if full_output:
        return torch.stack(xs)
    else:
        return x
