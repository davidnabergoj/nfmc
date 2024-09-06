import time
from dataclasses import dataclass
from typing import Sized, Optional

import torch
from tqdm import tqdm

from nfmc.algorithms.sampling.base import Sampler, NFMCParameters, MCMCStatistics, NFMCKernel, MCMCOutput
from nfmc.algorithms.sampling.mcmc.ess import ESSKernel, ESSParameters
from nfmc.util import multivariate_normal_sample
from potentials.base import Potential
from torchflows.flows import Flow
from torchflows.utils import get_batch_shape


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

    def log_phi(inputs):
        return flow.base_log_prob(inputs.to(flow.get_device())).cpu()

    def log_pi_hat(inputs):
        x, log_det = flow.bijection.inverse(inputs.to(flow.get_device()))
        out = -potential(x) - log_det
        return out.cpu()

    def bijection_inverse(inputs):
        return flow.bijection.inverse(inputs.to(flow.get_device()))[0].cpu()

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
    x_proposed = bijection_inverse(u_proposed)
    for i in range(max_iterations):
        u_prime = u * torch.cos(theta) + v * torch.sin(theta)
        v_prime = v * torch.cos(theta) - u * torch.sin(theta)
        x_prime = bijection_inverse(u_prime)
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

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: int = None) -> MCMCOutput:
        self.kernel: TESSKernel
        self.params: TESSParameters

        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=self.params.store_samples)

        t0 = time.time()
        n_chains, *event_shape = x0.shape
        u = multivariate_normal_sample((n_chains,), self.kernel.flow.bijection.event_shape, self.kernel.cov)
        out.statistics.elapsed_time_seconds += time.time() - t0

        pbar = tqdm(range(self.params.n_warmup_iterations), desc='[Warmup] TESS', disable=not show_progress)
        for i in pbar:
            if time_limit_seconds is not None and out.statistics.elapsed_time_seconds >= time_limit_seconds:
                break

            t0 = time.time()
            pbar.set_description_str('[Warmup] TESS sampling')
            x, u, accepted_mask = transport_elliptical_slice_sampling_step(
                u,
                self.kernel.flow,
                self.negative_log_likelihood,
                cov=self.kernel.cov
            )
            out.running_samples.add(u)
            out.statistics.n_target_calls += (self.params.max_ess_step_iterations + 1) * n_chains

            out.statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            out.statistics.n_attempted_trajectories += n_chains
            pbar.set_postfix_str(f'{out.statistics}')

            x_train = x.detach().clone()
            x_train = x_train[torch.randperm(len(x_train))]
            n_train = int(len(x_train) * self.params.train_pct)
            x_train, x_val = x_train[:n_train], x_train[n_train:]
            pbar.set_description_str('[Warmup] TESS training')
            self.kernel.flow.fit(x_train, x_val=x_val, **self.params.flow_fit_kwargs)
            out.statistics.elapsed_time_seconds += time.time() - t0

        out.kernel = self.kernel
        return out

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: int = None) -> MCMCOutput:
        self.kernel: TESSKernel
        self.params: TESSParameters

        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=self.params.store_samples)

        t0 = time.time()
        n_chains, *event_shape = x0.shape
        u = x0
        out.statistics.elapsed_time_seconds += time.time() - t0

        for i in (pbar := tqdm(range(self.params.n_iterations), desc='TESS sampling')):
            if time_limit_seconds is not None and out.statistics.elapsed_time_seconds >= time_limit_seconds:
                break

            t0 = time.time()
            x, u, accepted_mask = transport_elliptical_slice_sampling_step(
                u,
                self.kernel.flow,
                self.negative_log_likelihood,
                cov=self.kernel.cov
            )
            out.statistics.n_target_calls += (self.params.max_ess_step_iterations + 1) * n_chains

            out.statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            out.statistics.n_attempted_trajectories += n_chains
            pbar.set_postfix_str(f'{out.statistics}')
            out.running_samples.add(x)
            out.statistics.elapsed_time_seconds += time.time() - t0

        out.kernel = self.kernel
        return out
