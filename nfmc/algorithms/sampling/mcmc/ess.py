from dataclasses import dataclass
from typing import Sized, Union, Tuple, Dict, Any

import torch

from nfmc.algorithms.sampling.base import MCMCKernel, MCMCParameters, MCMCOutput
from nfmc.algorithms.sampling.mcmc.base import MCMCSampler
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


class ESS(MCMCSampler):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
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

    @property
    def name(self):
        return "ESS"

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        n_chains = x.shape[0]

        try:
            x_prime, mask = elliptical_slice_sampling_step(
                x,
                self.negative_log_likelihood,
                event_shape=self.event_shape,
                cov=self.kernel.cov,
                max_iterations=self.params.max_ess_step_iterations
            )
            mask = torch.ones_like(mask)  # All are accepted (technical hack)
            n_divergences = 0
        except ValueError:
            x_prime = x
            mask = torch.zeros(size=(n_chains,), dtype=torch.bool)
            n_divergences = 1

        n_calls = (self.params.max_ess_step_iterations + 1) * n_chains
        n_grads = 0
        return x_prime, mask, n_calls, n_grads, n_divergences

    def update_kernel(self, data: Dict[str, Any]):
        pass

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        n_chains = x0.shape[0]
        x0 = multivariate_normal_sample((n_chains,), self.event_shape, self.kernel.cov)
        return super().sample(x0=x0, show_progress=show_progress, time_limit_seconds=time_limit_seconds)
