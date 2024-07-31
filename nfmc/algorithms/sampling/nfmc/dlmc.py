from typing import Sized, Optional, Any

from tqdm import tqdm
import torch

from nfmc.algorithms.base import Sampler, MCMCOutput, MCMCParameters, MCMCStatistics, NFMCKernel, NFMCParameters
from nfmc.util import metropolis_acceptance_log_ratio, compute_grad
from dataclasses import dataclass


@dataclass
class DLMCKernel(NFMCKernel):
    step_size: float = 0.05


@dataclass
class DLMCParameters(NFMCParameters):
    latent_updates: bool = False


class DLMC(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 negative_log_likelihood: callable,
                 kernel: Optional[DLMCKernel] = None,
                 params: Optional[DLMCParameters] = None):
        if kernel is None:
            kernel = DLMCKernel(event_shape)
        if params is None:
            params = DLMCParameters()
        super().__init__(event_shape, target, kernel, params)
        self.negative_log_likelihood = negative_log_likelihood

    def sample(self, x0: torch.Tensor, show_progress: bool = False) -> MCMCOutput:
        self.kernel: DLMCKernel
        self.params: DLMCParameters
        n_chains, *event_shape = x0.shape
        statistics = MCMCStatistics(n_accepted_trajectories=0)

        # Initial update
        grad = compute_grad(self.negative_log_likelihood, x0)
        x = x0 - self.kernel.step_size * grad
        x.requires_grad_(False)

        xs = torch.zeros(size=(self.params.n_iterations, n_chains, *event_shape), dtype=x.dtype, device=x.device)
        for i in (pbar := tqdm(range(self.params.n_iterations), desc='DLMC sampling', disable=not show_progress)):
            x_train = x.detach().clone()
            x_train = x_train[torch.randperm(len(x_train))]
            n_train = int(len(x_train) * self.params.train_pct)
            x_train, x_val = x_train[:n_train], x_train[n_train:]
            x_train = x_train[:self.params.max_train_size]
            x_val = x_val[:self.params.max_val_size]
            self.kernel.flow.fit(x_train, x_val=x_val, **self.params.flow_fit_kwargs)

            if self.params.latent_updates:
                z, _ = self.kernel.flow.bijection.forward(x)
                grad = compute_grad(self.target, x)
                z = z - self.kernel.step_size * (grad - z)
                x, _ = self.kernel.flow.bijection.inverse(z)
            else:
                grad = compute_grad(lambda v: self.target(v) + self.kernel.flow.log_prob(v), x)
                x = x - self.kernel.step_size * grad

            x_tilde = self.kernel.flow.sample(n_chains, no_grad=True)
            log_alpha = metropolis_acceptance_log_ratio(
                log_prob_curr=-self.target(x),
                log_prob_prime=-self.target(x_tilde),
                log_proposal_curr=self.kernel.flow.log_prob(x),
                log_proposal_prime=self.kernel.flow.log_prob(x_tilde)
            )
            log_u = torch.rand(n_chains).log().to(log_alpha)
            accepted_mask = torch.less(log_u, log_alpha)
            x[accepted_mask] = x_tilde[accepted_mask]
            x = x.detach()
            xs[i] = x

            statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            pbar.set_postfix_str(f'{statistics}')

        statistics.acceptance_rate = statistics.n_accepted_trajectories / (self.params.n_iterations * n_chains)
        return MCMCOutput(samples=xs, statistics=statistics)
