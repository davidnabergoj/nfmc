import time
from typing import Optional, Union, Tuple

from tqdm import tqdm
import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCOutput, NFMCKernel, NFMCParameters
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
                 event_shape: Union[Tuple[int, ...], torch.Size],
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

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=self.params.store_samples)
        out.running_samples.add(x0)
        return out

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        self.kernel: DLMCKernel
        self.params: DLMCParameters
        n_chains = x0.shape[0]

        def flow_log_prob(inputs):
            return self.kernel.flow.log_prob(inputs.to(self.kernel.flow.get_device())).cpu()

        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=self.params.store_samples)

        # Initial update
        t0 = time.time()
        grad = compute_grad(self.negative_log_likelihood, x0)
        x = x0 - self.kernel.step_size * grad
        x.requires_grad_(False)

        out.statistics.update_counters(
            n_target_calls=n_chains,
            n_target_gradient_calls=n_chains,
        )
        out.statistics.update_elapsed_time(time.time() - t0)

        for i in (pbar := tqdm(range(self.params.n_iterations), desc='DLMC sampling', disable=not show_progress)):
            if time_limit_seconds is not None and out.statistics.elapsed_time_seconds >= time_limit_seconds:
                break
            t0 = time.time()
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
                grad = compute_grad(lambda v: self.target(v).cpu() + flow_log_prob(v), x)
                x = x - self.kernel.step_size * grad

            out.statistics.update_counters(
                n_target_calls=n_chains,
                n_target_gradient_calls=n_chains
            )
            x_tilde = self.kernel.flow.sample(n_chains, no_grad=True)
            log_alpha = metropolis_acceptance_log_ratio(
                log_prob_target_curr=-self.target(x).cpu(),
                log_prob_target_prime=-self.target(x_tilde).cpu(),
                log_prob_proposal_curr=flow_log_prob(x).cpu(),
                log_prob_proposal_prime=flow_log_prob(x_tilde).cpu()
            )
            log_u = torch.rand(n_chains).log().to(log_alpha)
            accepted_mask = torch.less(log_u, log_alpha)
            x[accepted_mask] = x_tilde[accepted_mask].to(x)
            x = x.detach()

            # Update output
            out.running_samples.add(x)
            out.statistics.expectations.update(x)
            out.statistics.update_counters(
                n_target_calls=2 * n_chains,
                n_accepted_trajectories=int(torch.sum(accepted_mask)),
                n_attempted_trajectories=n_chains,
            )
            out.statistics.update_elapsed_time(time.time() - t0)

            pbar.set_postfix_str(f'{out.statistics}')

        out.kernel = self.kernel
        return out
