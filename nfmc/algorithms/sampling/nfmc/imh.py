from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Sized, Optional

from tqdm import tqdm
import torch

from nfmc.algorithms.sampling.base import Sampler, NFMCKernel, NFMCParameters, MCMCStatistics, MCMCOutput
from nfmc.util import metropolis_acceptance_log_ratio
from normalizing_flows import Flow


@dataclass
class IMHKernel(NFMCKernel):
    pass


@dataclass
class IMHParameters(NFMCParameters):
    train_distribution: str = 'uniform'
    adaptation_dropoff: float = 0.9999

    def __post_init__(self):
        if self.train_distribution not in ['bounded_geom_approx', 'bounded_geom', 'uniform']:
            raise ValueError


def sample_bounded_geom(p, max_val):
    v = torch.arange(0, max_val + 1)
    pdf = p * (1 - p) ** (max_val - v) / (1 - (1 - p) ** (max_val + 1))
    cdf = torch.cumsum(pdf, dim=0)
    assert torch.isclose(cdf[-1], torch.tensor(1.0))
    u = float(torch.rand(size=()))
    return int(torch.searchsorted(cdf, u, right=True))


class AdaptiveIMH(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: Optional[IMHKernel] = None,
                 params: Optional[IMHParameters] = None):
        if kernel is None:
            kernel = IMHKernel(event_shape)
        if params is None:
            params = IMHParameters()
        super().__init__(event_shape, target, kernel, params)

    def sample(self, x0: torch.Tensor, show_progress: bool = False) -> MCMCOutput:
        # FIXME sometimes IMH disables autograd for flows in place
        # TODO make initial states be sampled from the flow

        self.kernel: IMHKernel
        self.params: IMHParameters

        xs = torch.empty(size=(self.params.n_iterations, *x0.shape), dtype=x0.dtype, device=x0.device)
        n_chains, *event_shape = x0.shape
        x = deepcopy(x0)
        statistics = MCMCStatistics(n_accepted_trajectories=0, n_divergences=0)

        for i in (pbar := tqdm(range(self.params.n_iterations), desc="Adaptive IMH", disable=not show_progress)):
            with torch.no_grad():
                x_prime = self.kernel.flow.sample(n_chains, no_grad=True).to(xs)
                try:
                    log_alpha = metropolis_acceptance_log_ratio(
                        log_prob_curr=-self.target(x).to(xs),
                        log_prob_prime=-self.target(x_prime).to(xs),
                        log_proposal_curr=self.kernel.flow.log_prob(x).to(xs),
                        log_proposal_prime=self.kernel.flow.log_prob(x_prime).to(xs)
                    )
                    log_u = torch.rand(n_chains).log().to(log_alpha)
                    accepted_mask = torch.less(log_u, log_alpha)
                    x[accepted_mask] = x_prime[accepted_mask]
                    x = x.detach()
                except ValueError as e:
                    accepted_mask = torch.zeros(size=(n_chains,), dtype=torch.bool, device=x0.device)
                    statistics.n_divergences += 1
            xs[i] = x
            statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            statistics.acceptance_rate + statistics.n_accepted_trajectories / ((i + 1) * n_chains)

            u_prime = torch.rand(size=())
            alpha_prime = self.params.adaptation_dropoff ** i
            if u_prime < alpha_prime:
                # only use recent states to adapt
                # this is an approximation of a bounded "geometric distribution" that picks the training data
                # we can program the exact bounded geometric as well. Then it's parameter p can be adapted with dual
                # averaging.
                if self.params.train_distribution == 'uniform':
                    k = int(torch.randint(low=0, high=(i + 1), size=()))
                elif self.params.train_distribution == 'bounded_geom_approx':
                    k = int(torch.randint(low=max(0, (i + 1) - 100), high=(i + 1), size=()))
                elif self.params.train_distribution == 'bounded_geom':
                    k = sample_bounded_geom(p=0.025, max_val=(i + 1) - 1)
                else:
                    raise ValueError

                x_train = xs[k]
                self.kernel.flow.fit(x_train, n_epochs=1)

            pbar.set_postfix_str(f'{statistics} | adaptation probability: {alpha_prime:.6f}')

        return MCMCOutput(samples=xs, statistics=statistics)


class FixedIMH(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: Optional[IMHKernel] = None,
                 params: Optional[IMHParameters] = None):
        if kernel is None:
            kernel = IMHKernel(event_shape)
        if params is None:
            params = IMHParameters()
        super().__init__(event_shape, target, kernel, params)

    def sample(self, x0: torch.Tensor, show_progress: bool = False) -> MCMCOutput:
        xs = torch.empty(size=(self.params.n_iterations, *x0.shape), dtype=x0.dtype, device=x0.device)
        n_chains, *event_shape = x0.shape
        statistics = MCMCStatistics(n_accepted_trajectories=0, n_divergences=0)
        x = deepcopy(x0)
        for i in (pbar := tqdm(range(self.params.n_iterations), desc="Fixed IMH", disable=not show_progress)):
            with torch.no_grad():
                x_prime = self.kernel.flow.sample(n_chains, no_grad=True).to(xs)
                try:
                    log_alpha = metropolis_acceptance_log_ratio(
                        log_prob_curr=-self.target(x).to(xs),
                        log_prob_prime=-self.target(x_prime).to(xs),
                        log_proposal_curr=self.kernel.flow.log_prob(x).to(xs),
                        log_proposal_prime=self.kernel.flow.log_prob(x_prime).to(xs)
                    )
                    log_u = torch.rand(n_chains).log().to(log_alpha)
                    accepted_mask = torch.less(log_u, log_alpha)
                    x[accepted_mask] = x_prime[accepted_mask]
                    x = x.detach()
                except ValueError as e:
                    accepted_mask = torch.zeros(size=(n_chains,), dtype=torch.bool, device=x0.device)
                    statistics.n_divergences += 1
            xs[i] = x
            statistics.n_accepted_trajectories += int(torch.sum(accepted_mask))
            statistics.acceptance_rate + statistics.n_accepted_trajectories / ((i + 1) * n_chains)

            pbar.set_postfix_str(f'{statistics}')
        return MCMCOutput(samples=xs, statistics=statistics)


def imh_fixed_flow(x0: torch.Tensor,
                   flow: Flow,
                   target: callable,
                   n_iterations: int = 1000,
                   device=torch.device('cpu'),
                   show_progress: bool = True):
    """

    :param x0:
    :param flow:
    :param target: target potential.
    :param n_iterations:
    :param device:
    :param show_progress:
    :return:
    """
    xs = torch.empty(size=(n_iterations, *x0.shape), dtype=x0.dtype)
    n_accepted = 0
    n_total = 0
    n_chains, *event_shape = x0.shape
    x = deepcopy(x0)
    for i in (pbar := tqdm(range(n_iterations), desc="Independent Metropolis-Hastings", disable=not show_progress)):
        with torch.no_grad():
            x_proposed = flow.sample(n_chains, no_grad=True).to(device)
            try:
                log_alpha = metropolis_acceptance_log_ratio(
                    log_prob_curr=-target(x).to(device),
                    log_prob_prime=-target(x_proposed).to(device),
                    log_proposal_curr=flow.log_prob(x).to(device),
                    log_proposal_prime=flow.log_prob(x_proposed).to(device)
                )
            except ValueError as e:
                print("Encountered error in Metropolis acceptance ratio computation")
                print(e)
                torch.fill(xs[i:], torch.nan)
                break

            log_u = torch.rand(n_chains).log().to(log_alpha)
            accepted_mask = torch.less(log_u, log_alpha)
            x[accepted_mask] = x_proposed[accepted_mask]
            x = x.detach()
            xs[i] = deepcopy(x)
        n_accepted += int(torch.sum(accepted_mask.long()))
        n_total += n_chains
        pbar.set_postfix_str(f'Acceptance rate: {n_accepted / n_total:.6f}')
    return MCMCOutput(samples=xs)


def imh(x0: torch.Tensor,
        flow: Flow,
        target: callable,
        n_iterations: int = 1000,
        adaptation_dropoff: float = 0.9999,
        train_dist: str = 'uniform',
        device=torch.device('cpu'),
        show_progress: bool = True,
        **kwargs):
    """

    :param x0:
    :param flow:
    :param target: target potential.
    :param n_iterations:
    :param adaptation_dropoff:
    :param train_dist:
    :param device:
    :param kwargs:
    :return:
    """
    # FIXME sometimes IMH disables autograd for flows in place
    assert train_dist in ['bounded_geom_approx', 'bounded_geom', 'uniform']
    # Exponentially diminishing adaptation probability sequence
    # TODO make initial states be sampled from the flow

    xs = torch.empty(size=(n_iterations, *x0.shape), dtype=x0.dtype)

    n_accepted = 0
    n_total = 0

    n_chains, *event_shape = x0.shape
    x = deepcopy(x0)
    for i in (pbar := tqdm(range(n_iterations), desc="Independent Metropolis-Hastings", disable=not show_progress)):
        with torch.no_grad():
            x_proposed = flow.sample(n_chains, no_grad=True).to(device)

            try:
                log_alpha = metropolis_acceptance_log_ratio(
                    log_prob_curr=-target(x).to(device),
                    log_prob_prime=-target(x_proposed).to(device),
                    log_proposal_curr=flow.log_prob(x).to(device),
                    log_proposal_prime=flow.log_prob(x_proposed).to(device)
                )
            except ValueError as e:
                print("Encountered error in Metropolis acceptance ratio computation")
                print(e)
                torch.fill(xs[i:], torch.nan)
                break

            log_u = torch.rand(n_chains).log().to(log_alpha)
            accepted_mask = torch.less(log_u, log_alpha)
            x[accepted_mask] = x_proposed[accepted_mask]
            x = x.detach()
            xs[i] = deepcopy(x)

        n_accepted += int(torch.sum(accepted_mask.long()))
        n_total += n_chains
        # instantaneous_acceptance_rate = float(torch.mean(accepted_mask.float()))

        u_prime = torch.rand(size=())
        alpha_prime = adaptation_dropoff ** i
        if u_prime < alpha_prime:
            # only use recent states to adapt
            # this is an approximation of a bounded "geometric distribution" that picks the training data
            # we can program the exact bounded geometric as well. Then it's parameter p can be adapted with dual
            # averaging.
            if train_dist == 'uniform':
                k = int(torch.randint(low=0, high=(i + 1), size=()))
            elif train_dist == 'bounded_geom_approx':
                k = int(torch.randint(low=max(0, (i + 1) - 100), high=(i + 1), size=()))
            elif train_dist == 'bounded_geom':
                k = sample_bounded_geom(p=0.025, max_val=(i + 1) - 1)
            else:
                raise ValueError
            x_train = xs[k]
            flow.fit(x_train, n_epochs=1, **kwargs)

        pbar.set_postfix_str(f'Acceptance rate: {n_accepted / n_total:.6f} | adapt-prob: {alpha_prime:.6f}')

    return MCMCOutput(
        samples=xs
    )


def aggressive_imh(x0: torch.Tensor,
                   flow: Flow,
                   potential: callable,
                   n_iterations: int = 1000,
                   adaptation_dropoff: float = 0.9999,
                   train_dist: str = 'uniform',
                   device=torch.device('cpu'),
                   **kwargs):
    n_chains, *event_shape = x0.shape
    xs = torch.zeros(size=(n_iterations, n_chains, *event_shape), dtype=x0.dtype)

    n_accepted = 0
    n_total = 0

    train_indices = defaultdict(int)
    x = deepcopy(x0)
    for i in (pbar := tqdm(range(n_iterations))):
        with torch.no_grad():
            x_proposed = flow.sample(n_chains, no_grad=True).to(device)
            log_alpha = metropolis_acceptance_log_ratio(
                log_prob_curr=-potential(x).to(device),
                log_prob_prime=-potential(x_proposed).to(device),
                log_proposal_curr=flow.log_prob(x).to(device),
                log_proposal_prime=flow.log_prob(x_proposed).to(device)
            )
            log_u = torch.rand(n_chains).log().to(log_alpha)
            accepted_mask = torch.less(log_u, log_alpha)
            x[accepted_mask] = x_proposed[accepted_mask]
            x = x.detach()
            xs[i] = deepcopy(x)

        n_accepted += int(torch.sum(accepted_mask.long()))
        n_total += n_chains
        # instantaneous_acceptance_rate = float(torch.mean(accepted_mask.float()))

        u_prime = torch.rand(size=())
        alpha_prime = adaptation_dropoff ** i
        if u_prime < alpha_prime:
            # only use recent states to adapt
            # this is an approximation of a bounded "geometric distribution" that picks the training data
            # we can program the exact bounded geometric as well. Then it's parameter p can be adapted with dual
            # averaging.
            if train_dist == 'uniform':
                k = int(torch.randint(low=0, high=(i + 1), size=()))
            elif train_dist == 'bounded_geom_approx':
                k = int(torch.randint(low=max(0, (i + 1) - 100), high=(i + 1), size=()))
            elif train_dist == 'bounded_geom':
                k = int(sample_bounded_geom(p=0.025, max_val=(i + 1) - 1))
            else:
                raise ValueError
            train_indices[k] += 1

            train_data_indices, train_data_weights = zip(*train_indices.items())
            idx_train = torch.tensor(list(train_data_indices), dtype=torch.long)
            w_train = torch.tensor(list(train_data_weights), dtype=torch.float)
            w_train = w_train.unsqueeze(1).repeat(1, n_chains).view(-1)
            x_train = xs[idx_train].view(-1, *event_shape)

            idx_final = torch.randperm(len(x_train))[:1000]

            flow.fit(x_train[idx_final], n_epochs=1, w_train=w_train[idx_final], **kwargs)

        pbar.set_postfix_str(f'accept-frac: {n_accepted / n_total:.6f} | adapt-prob: {alpha_prime:.6f}')

    return xs
