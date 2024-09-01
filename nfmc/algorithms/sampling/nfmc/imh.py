import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from tqdm import tqdm
import torch

from nfmc.algorithms.sampling.base import Sampler, NFMCKernel, NFMCParameters, MCMCOutput
from nfmc.util import metropolis_acceptance_log_ratio


@dataclass
class IMHKernel(NFMCKernel):
    pass


@dataclass
class IMHParameters(NFMCParameters):
    adaptation: bool = True
    train_distribution: str = 'uniform'
    adaptation_dropoff: float = 0.9999
    warmup_fit_kwargs: dict = None

    def __post_init__(self):
        if self.train_distribution not in ['bounded_geom_approx', 'bounded_geom', 'uniform']:
            raise ValueError
        if self.warmup_fit_kwargs is None:
            self.warmup_fit_kwargs = {
                'early_stopping': True,
                'early_stopping_threshold': 50,
                'keep_best_weights': True,
                'n_samples': 1,
                'n_epochs': 500,
                'lr': 0.05,
                'check_for_divergences': True
            }


def sample_bounded_geom(p, max_val):
    v = torch.arange(0, max_val + 1)
    pdf = p * (1 - p) ** (max_val - v) / (1 - (1 - p) ** (max_val + 1))
    cdf = torch.cumsum(pdf, dim=0)
    assert torch.isclose(cdf[-1], torch.tensor(1.0))
    u = float(torch.rand(size=()))
    return int(torch.searchsorted(cdf, u, right=True))


class AdaptiveIMH(Sampler):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: Optional[IMHKernel] = None,
                 params: Optional[IMHParameters] = None):
        if kernel is None:
            kernel = IMHKernel(event_shape)
        if params is None:
            params = IMHParameters()
        super().__init__(event_shape, target, kernel, params)

    @property
    def name(self):
        return "Adaptive IMH"

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               thinning: int = 1,
               time_limit_seconds: int = 3600 * 24) -> MCMCOutput:
        self.kernel: IMHKernel
        self.params: IMHParameters

        self.kernel.flow.variational_fit(
            lambda v: -self.target(v),
            **self.params.warmup_fit_kwargs,
            show_progress=show_progress
        )

        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=True)
        out.running_samples.add(self.kernel.flow.sample(x0.shape[0]).detach())
        return out

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               thinning: int = 1,
               time_limit_seconds: int = 3600 * 24) -> MCMCOutput:
        self.kernel: IMHKernel
        self.params: IMHParameters

        out = MCMCOutput(event_shape=x0.shape[1:], store_samples=self.params.adaptation or self.params.store_samples)
        out.running_samples.thinning = thinning

        t0 = time.time()
        n_chains = x0.shape[0]
        x = deepcopy(x0)
        out.statistics.elapsed_time_seconds += time.time() - t0

        for i in (pbar := tqdm(range(self.params.n_iterations), desc=self.name, disable=not show_progress)):
            if out.statistics.elapsed_time_seconds >= time_limit_seconds:
                break

            t0 = time.time()
            with torch.no_grad():
                x_prime = self.kernel.flow.sample(n_chains, no_grad=True)
                try:
                    log_alpha = metropolis_acceptance_log_ratio(
                        log_prob_curr=-self.target(x).cpu(),
                        log_prob_prime=-self.target(x_prime).cpu(),
                        log_proposal_curr=self.kernel.flow.log_prob(x).cpu(),
                        log_proposal_prime=self.kernel.flow.log_prob(x_prime).cpu()
                    )
                    log_u = torch.rand(n_chains).log().to(log_alpha)
                    accepted_mask = torch.less(log_u, log_alpha)
                    x[accepted_mask] = x_prime[accepted_mask].to(x)
                    x = x.detach()
                except ValueError:
                    accepted_mask = torch.zeros(size=(n_chains,), dtype=torch.bool, device=x0.device)
                    out.statistics.n_divergences += 1

            out.statistics.expectations.update(x)
            out.statistics.n_target_calls += 2 * n_chains

            out.running_samples.add(x)
            out.statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            out.statistics.n_attempted_trajectories += n_chains

            if self.params.adaptation:
                u_prime = torch.rand(size=())
                alpha_prime = self.params.adaptation_dropoff ** i
                if u_prime < alpha_prime:
                    # only use recent states to adapt
                    # this is an approximation of a bounded "geometric distribution" that picks the training data
                    # we can program the exact bounded geometric as well. Then its parameter p can be adapted with dual
                    # averaging.
                    n_samples = out.running_samples.n_samples
                    if self.params.train_distribution == 'uniform':
                        k = int(torch.randint(low=0, high=n_samples, size=()))
                    elif self.params.train_distribution == 'bounded_geom_approx':
                        k = int(torch.randint(low=max(0, n_samples - 100), high=n_samples, size=()))
                    elif self.params.train_distribution == 'bounded_geom':
                        k = sample_bounded_geom(p=0.025, max_val=n_samples - 1)
                    else:
                        raise ValueError

                    x_train = out.running_samples[k]

                    flow_weights = deepcopy(self.kernel.flow.state_dict())
                    try:
                        self.kernel.flow.fit(x_train, n_epochs=1)
                    except ValueError:
                        self.kernel.flow.load_state_dict(flow_weights)

            out.statistics.elapsed_time_seconds += time.time() - t0
            pbar.set_postfix_str(f'{out.statistics}')

        out.kernel = self.kernel
        return out


class FixedIMH(AdaptiveIMH):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: Optional[IMHKernel] = None,
                 params: Optional[IMHParameters] = None):
        if kernel is None:
            kernel = IMHKernel(event_shape)
        if params is None:
            params = IMHParameters(adaptation=False)
        super().__init__(event_shape, target, kernel, params)

    @property
    def name(self):
        return "Fixed IMH"
