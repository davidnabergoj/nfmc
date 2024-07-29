import math
from copy import deepcopy

import torch
from tqdm import tqdm

from nfmc.util import MCMCOutput
from potentials.base import Potential
from normalizing_flows import Flow
from normalizing_flows.utils import get_batch_shape


def multivariate_normal_sample(batch_shape, event_shape, cov):
    """
    Draw samples from N(0, cov).
    If cov is None, we assume cov = identity.
    """
    if cov is None:
        samples = torch.randn(size=(*batch_shape, *event_shape))
    else:
        event_size = int(torch.prod(torch.as_tensor(event_shape)))
        assert cov.shape == (event_size, event_size)
        samples_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(event_size), covariance_matrix=cov)
        samples_flat = samples_dist.sample(batch_shape)
        samples = samples_flat.view(*batch_shape, *event_shape)
    return samples

@torch.no_grad()
def elliptical_slice_sampling_step(
        f: torch.Tensor,
        negative_log_likelihood: callable,
        event_shape,
        cov: torch.Tensor = None,
        max_iterations: int = 5,
):
    """
    :param f: current state with shape (*batch_shape, *event_shape).
    :param negative_log_likelihood: negative log likelihood callable which receives as input a tensor with shape
        (*batch_shape, *event_shape) and outputs negative log likelihood values with shape batch_shape.
    :param event_shape: shape of each instance/draw.
    :param cov: covariance matrix for the prior with shape (event_size, event_size) where event_size is
        equal to the product of elements of event_shape. If None, the covariance is assumed to be identity.
    :param max_iterations: maximum number of iterations where the proposal bracket is shrunk.
    """
    batch_shape = get_batch_shape(f, event_shape)

    # 1. Choose ellipse
    nu = multivariate_normal_sample(batch_shape, event_shape, cov)

    # 2. Log-likelihood threshold
    u = torch.rand(size=batch_shape)
    log_y = -negative_log_likelihood(f) + torch.log(u)

    # 3. Draw an initial proposal, also defining a bracket
    theta = torch.rand(size=[*batch_shape, *([1] * len(event_shape))]) * 2 * torch.pi
    theta_min = theta - 2 * torch.pi
    theta_max = theta

    accepted_mask = torch.zeros(size=batch_shape, dtype=torch.bool)
    f_proposed = f
    for _ in range(max_iterations):
        f_prime = f * torch.cos(theta) + nu * torch.sin(theta)
        accepted_mask_update = -negative_log_likelihood(f_prime) > log_y

        # Must have been accepted now and not previously
        f_proposed[accepted_mask_update & (~accepted_mask)] = f_prime[accepted_mask_update & (~accepted_mask)]

        # Update theta (we overwrite old thetas as they are unnecessary)
        theta_mask = theta < 0  # To avoid overwriting, we would consider the global accepted mask here
        theta_min[theta_mask] = theta[theta_mask]
        theta_max[~theta_mask] = theta[~theta_mask]

        # Draw new theta uniformly from [theta_min, theta_max]
        uniform_noise = torch.rand(size=[*batch_shape, *([1] * len(event_shape))])
        theta = uniform_noise * (theta_max - theta_min) + theta_min

        # Update the global accepted mask
        accepted_mask |= accepted_mask_update

    return f_proposed.detach(), accepted_mask


def elliptical_slice_sampler(
        negative_log_likelihood: callable,
        event_shape,
        n_chains: int = 100,
        n_iterations: int = 1000,
        cov: torch.Tensor = None,
        show_progress: bool = False,
        **kwargs
):
    """
    A sampling method that samples from a posterior, defined by a likelihood and a multivariate normal prior.

    :param negative_log_likelihood: negative log likelihood callable which receives as input a tensor with shape
        (*batch_shape, *event_shape) and outputs negative log likelihood values with shape batch_shape.
    :param event_shape: shape of each instance/draw.
    :param n_chains: number of independent parallel chains.
    :param n_iterations: number of iterations.
    :param cov: covariance matrix for the prior with shape (event_size, event_size) where event_size is
        equal to the product of elements of event_shape. If None, the covariance is assumed to be identity.
    :param show_progress: optionally show a progress bar.
    :param kwargs: keyword arguments for the slice sampling step.
    """
    x = multivariate_normal_sample((n_chains,), event_shape, cov)
    draws = []

    if show_progress:
        iterator = tqdm(range(n_iterations), desc="Elliptical slice sampling")
    else:
        iterator = range(n_iterations)

    for _ in iterator:
        x, accepted_mask = elliptical_slice_sampling_step(
            x,
            negative_log_likelihood,
            event_shape=event_shape,
            cov=cov,
            **kwargs
        )
        draws.append(torch.clone(x))
        if show_progress:
            acceptance_rate = float(torch.mean(torch.as_tensor(accepted_mask, dtype=torch.float)))
            iterator.set_postfix_str(f'accept-rate: {acceptance_rate:.4f}')
    return torch.stack(draws)


@torch.no_grad()
def transport_elliptical_slice_sampling_step(
        u: torch.Tensor,
        flow: Flow,
        potential: Potential,
        cov: torch.Tensor = None,
        max_iterations: int = 5
):
    n_chains, *event_shape = u.shape
    batch_shape = get_batch_shape(u, event_shape)
    log_phi = flow.base_log_prob

    def log_pi_hat(u_):
        x, log_det = flow.bijection.inverse(u_)
        return -potential(x) - log_det

    # 1. Choose ellipse
    v = multivariate_normal_sample(batch_shape, event_shape, cov)

    # 2. Log-likelihood threshold
    w = torch.rand(size=batch_shape)
    log_s = log_pi_hat(u) + log_phi(v) + w.log()

    # 3. Draw an initial proposal, also defining a bracket
    theta = torch.randn_like(w) * (2 * torch.pi)
    expanded_shape = (*batch_shape, *([1] * len(event_shape)))
    theta = theta.view(*expanded_shape)
    theta_min, theta_max = theta - 2 * torch.pi, theta

    accepted_mask = torch.zeros(size=batch_shape, dtype=torch.bool)
    u_proposed = torch.clone(u)
    x_proposed, _ = flow.bijection.inverse(u_proposed)
    for i in range(max_iterations):
        u_prime = u * torch.cos(theta) + v * torch.sin(theta)
        v_prime = v * torch.cos(theta) - u * torch.sin(theta)
        x_prime, _ = flow.bijection.inverse(u_prime)
        accepted_mask_update = ((log_pi_hat(u_prime) + log_phi(v_prime)) > log_s)

        # We update the outputs the first time this criterion is satisfied
        x_proposed[accepted_mask_update & (~accepted_mask)] = x_prime[accepted_mask_update & (~accepted_mask)]
        u_proposed[accepted_mask_update & (~accepted_mask)] = u_prime[accepted_mask_update & (~accepted_mask)]

        # Update theta (we overwrite old thetas as they are unnecessary)
        theta_mask = (theta < 0)
        theta_min[theta_mask] = theta[theta_mask]
        theta_max[~theta_mask] = theta[~theta_mask]

        # Draw new theta uniformly from [theta_min, theta_max]
        uniform_noise = torch.rand(size=[*batch_shape, *([1] * len(event_shape))])
        theta = uniform_noise * (theta_max - theta_min) + theta_min

        # Update the global accepted mask
        accepted_mask |= accepted_mask_update

    return x_proposed.detach(), u_proposed.detach(), accepted_mask


def transport_elliptical_slice_sampler(flow: Flow,
                                       negative_log_likelihood: callable,
                                       n_chains: int = 100,
                                       n_warmup_iterations: int = 100,
                                       n_sampling_iterations: int = 250,
                                       n_epochs: int = 10,
                                       cov: torch.Tensor = None,
                                       show_progress: bool = False,
                                       full_output: bool = True):
    u = multivariate_normal_sample((n_chains,), flow.bijection.event_shape, cov)

    if show_progress:
        iterator = tqdm(range(n_warmup_iterations), desc='TESS warmup and training')
    else:
        iterator = range(n_warmup_iterations)

    # Warmup
    for _ in iterator:
        x, u, accepted_mask = transport_elliptical_slice_sampling_step(u, flow, negative_log_likelihood, cov=cov)
        if show_progress:
            acceptance_rate = float(torch.mean(torch.as_tensor(accepted_mask, dtype=torch.float)))
            iterator.set_postfix_str(f'accept-rate: {acceptance_rate:.4f}')
        flow.fit(x.detach(), n_epochs=n_epochs)

    # Sampling
    draws = []

    if show_progress:
        iterator = tqdm(range(n_sampling_iterations), desc='TESS sampling')
    else:
        iterator = range(n_sampling_iterations)

    for _ in iterator:
        x, u, accepted_mask = transport_elliptical_slice_sampling_step(u, flow, negative_log_likelihood, cov=cov)
        if show_progress:
            acceptance_rate = float(torch.mean(torch.as_tensor(accepted_mask, dtype=torch.float)))
            iterator.set_postfix_str(f'accept-rate: {acceptance_rate:.4f}')
        if full_output:
            draws.append(deepcopy(x.detach()))

    draws = torch.stack(draws)

    if full_output:
        return MCMCOutput(samples=draws)
    return draws
