from dataclasses import dataclass
from typing import Sized, Optional

import torch
from tqdm import tqdm

from nfmc.algorithms.base import Sampler, NFMCParameters, MCMCStatistics, NFMCKernel, MCMCOutput
from nfmc.algorithms.sampling.mcmc.ess import ESSKernel, ESSParameters
from nfmc.util import multivariate_normal_sample
from potentials.base import Potential
from normalizing_flows import Flow
from normalizing_flows.utils import get_batch_shape


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


@dataclass
class TESSKernel(ESSKernel, NFMCKernel):
    cov: torch.Tensor = None


@dataclass
class TESSParameters(ESSParameters, NFMCParameters):
    n_warmup_iterations: int = 20


class TESS(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 negative_log_likelihood: callable,
                 kernel: Optional[TESSKernel] = None,
                 params: Optional[TESSParameters] = None):
        if kernel is None:
            kernel = TESSKernel(event_shape)
        if params is None:
            params = TESSParameters()
        super().__init__(event_shape, target, kernel, params)
        self.negative_log_likelihood = negative_log_likelihood

    def sample(self, x0: torch.Tensor, show_progress: bool = False) -> MCMCOutput:
        self.kernel: TESSKernel
        self.params: TESSParameters
        n_chains, *event_shape = x0.shape
        u = multivariate_normal_sample((n_chains,), self.kernel.flow.bijection.event_shape, self.kernel.cov)
        warmup_statistics = MCMCStatistics()
        sampling_statistics = MCMCStatistics()

        # Warmup
        for i in (pbar := tqdm(range(self.params.n_warmup_iterations), desc='TESS warmup', disable=not show_progress)):
            pbar.set_description_str('TESS warmup sampling')
            x, u, accepted_mask = transport_elliptical_slice_sampling_step(
                u,
                self.kernel.flow,
                self.negative_log_likelihood,
                cov=self.kernel.cov
            )
            warmup_statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            warmup_statistics.n_attempted_trajectories += n_chains
            pbar.set_postfix_str(f'{warmup_statistics}')

            x_train = x.detach().clone()
            x_train = x_train[torch.randperm(len(x_train))]
            n_train = int(len(x_train) * self.params.train_pct)
            x_train, x_val = x_train[:n_train], x_train[n_train:]
            pbar.set_description_str('TESS warmup NF fit')
            self.kernel.flow.fit(x_train, x_val=x_val, **self.params.flow_fit_kwargs)

        # Sampling
        xs = torch.zeros(size=(self.params.n_iterations, n_chains, *event_shape), dtype=x.dtype, device=x.device)
        for i in (pbar := tqdm(range(self.params.n_iterations), desc='TESS sampling')):
            x, u, accepted_mask = transport_elliptical_slice_sampling_step(
                u,
                self.kernel.flow,
                self.negative_log_likelihood,
                cov=self.kernel.cov
            )
            sampling_statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            sampling_statistics.n_attempted_trajectories += n_chains
            pbar.set_postfix_str(f'{sampling_statistics}')
            xs[i] = x.detach()
        return MCMCOutput(samples=xs, statistics=sampling_statistics)
