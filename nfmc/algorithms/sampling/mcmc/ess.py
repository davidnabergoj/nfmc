import time
from dataclasses import dataclass
from typing import Sized

import torch
from tqdm import tqdm

from nfmc.algorithms.sampling.base import Sampler, MCMCKernel, MCMCParameters, MCMCStatistics, MCMCOutput
from nfmc.util import multivariate_normal_sample
from torchflows.utils import get_batch_shape


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


@dataclass
class ESSKernel(MCMCKernel):
    event_shape: Sized
    cov: torch.Tensor = None


@dataclass
class ESSParameters(MCMCParameters):
    max_ess_step_iterations: int = 5


class ESS(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 negative_log_likelihood: callable,
                 kernel: ESSKernel = None,
                 params: ESSParameters = None):
        if kernel is None:
            kernel = ESSKernel(event_shape)
        if params is None:
            params = ESSParameters()
        super().__init__(event_shape, target, kernel, params)
        self.negative_log_likelihood = negative_log_likelihood

    def sample(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: ESSKernel
        self.params: ESSParameters
        statistics = MCMCStatistics(n_accepted_trajectories=0, n_divergences=0)

        t0 = time.time()
        n_chains, *event_shape = x0.shape
        x = multivariate_normal_sample((n_chains,), event_shape, self.kernel.cov)
        xs = torch.zeros(size=(self.params.n_iterations, n_chains, *event_shape), dtype=x.dtype, device=x.device)
        statistics.elapsed_time_seconds += time.time() - t0

        for i in (pbar := tqdm(range(self.params.n_iterations), desc="ESS", disable=not show_progress)):
            t0 = time.time()
            x, accepted_mask = elliptical_slice_sampling_step(
                x,
                self.negative_log_likelihood,
                event_shape=event_shape,
                cov=self.kernel.cov,
                max_iterations=self.params.max_ess_step_iterations
            )
            statistics.n_target_calls += (self.params.max_ess_step_iterations + 1) * n_chains
            xs[i] = x.detach()
            t1 = time.time()

            statistics.elapsed_time_seconds = t1 - t0
            statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            statistics.n_attempted_trajectories += n_chains
            pbar.set_postfix_str(f'{statistics}')
        return MCMCOutput(samples=xs, statistics=statistics)
