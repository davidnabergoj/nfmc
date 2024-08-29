import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Sized

import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCOutput, NFMCParameters, NFMCKernel, MCMCStatistics, MCMCKernel, \
    MCMCParameters
from tqdm import tqdm

from nfmc.algorithms.sampling.mcmc import HMC, UHMC, MALA, ULA, MH, NUTS
from nfmc.algorithms.sampling.mcmc.ess import ESS
from nfmc.algorithms.sampling.tuning import train_val_split
from nfmc.util import metropolis_acceptance_log_ratio


@dataclass
class JumpNFMCParameters(NFMCParameters):
    adjusted_jumps: bool = True
    fit_nf: bool = False
    warmup_fit_kwargs: dict = None
    n_jumps_before_training: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.warmup_fit_kwargs is None:
            self.warmup_fit_kwargs = {
                'early_stopping': True,
                'early_stopping_threshold': 50,
                'keep_best_weights': True,
                'n_samples': 1,
                'n_epochs': 500,
                'lr': 0.05
            }


@dataclass
class JumpNFMCStatistics(MCMCStatistics):
    n_accepted_jumps: int = 0
    n_attempted_jumps: int = 0
    n_divergences: int = 0

    @property
    def jump_acceptance_rate(self):
        if self.n_attempted_jumps == 0:
            return torch.nan
        return self.n_accepted_jumps / self.n_attempted_jumps

    def __repr__(self):
        return (
            f"MCMC acc-rate: {self.acceptance_rate:.2f}, "
            f"Jump acc-rate: {self.jump_acceptance_rate:.2f}, "
            f"kcalls/s: {self.calls_per_second / 1000:.2f}, "
            f"kgrads/s: {self.grads_per_second / 1000:.2f}, "
            f"divergences: {self.n_divergences}"
        )


class JumpNFMC(Sampler):
    """

    Requires flow with an efficient inverse method.
    Requires flow with an efficient forward method if using adjusted jumps (default: True). This makes masked autoregressive flows unsuitable.
    """
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 inner_sampler: Sampler,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None):
        if kernel is None:
            kernel = NFMCKernel(event_shape)
        if params is None:
            params = JumpNFMCParameters()
        super().__init__(event_shape, target, kernel, params)
        self.inner_sampler = inner_sampler

    def warmup(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: NFMCKernel
        self.params: JumpNFMCParameters

        # Fit flow to target via variational inference
        flow_params = deepcopy(self.kernel.flow.state_dict())
        try:
            self.kernel.flow.variational_fit(
                lambda v: -self.target(v),
                **self.params.warmup_fit_kwargs,
                show_progress=show_progress
            )
        except ValueError:
            self.kernel.flow.load_state_dict(flow_params)
        x0 = self.kernel.flow.sample(len(x0)).detach()

        warmup_output = self.inner_sampler.warmup(x0, show_progress=show_progress)
        x_train, x_val = train_val_split(
            warmup_output.samples,
            train_pct=self.params.train_pct,
            max_train_size=self.params.max_train_size,
            max_val_size=self.params.max_val_size
        )

        flow_params = deepcopy(self.kernel.flow.state_dict())
        try:
            self.kernel.flow.fit(
                x_train=x_train,
                x_val=x_val,
                **{
                    **self.params.flow_fit_kwargs,
                    **dict(show_progress=show_progress)
                }
            )
        except ValueError:
            self.kernel.flow.load_state_dict(flow_params)

        return MCMCOutput(samples=self.kernel.flow.sample(x0.shape[0]).detach()[None])

    def sample(self, x0: torch.Tensor, show_progress: bool = True, thinning: int = 1) -> MCMCOutput:
        self.kernel: NFMCKernel
        self.params: JumpNFMCParameters

        n_chains, *event_shape = x0.shape
        xs = torch.zeros(
            size=(self.params.n_iterations // thinning, self.inner_sampler.params.n_iterations + 1, *x0.shape),
            device=x0.device,
            dtype=x0.dtype
        )  # (jumps, trajectories per jump, chains, *event)
        statistics = JumpNFMCStatistics()

        x = torch.clone(x0)
        data_index = 0
        for i in (pbar := tqdm(range(self.params.n_iterations), desc='Jump MCMC', disable=not show_progress)):
            # Trajectories
            pbar.set_description_str(f'Jump MCMC (sampling)')
            mcmc_output = self.inner_sampler.sample(x0=x, show_progress=False)
            statistics.n_accepted_trajectories += mcmc_output.statistics.n_accepted_trajectories
            statistics.n_attempted_trajectories += mcmc_output.statistics.n_attempted_trajectories
            statistics.n_divergences += mcmc_output.statistics.n_divergences

            statistics.n_target_calls += mcmc_output.statistics.n_target_calls
            statistics.n_target_gradient_calls += mcmc_output.statistics.n_target_gradient_calls
            statistics.elapsed_time_seconds += mcmc_output.statistics.elapsed_time_seconds

            t0 = time.time()
            if i % thinning == 0:
                xs[data_index, :-1] = mcmc_output.samples

            # Fit flow
            if self.params.fit_nf and i >= self.params.n_jumps_before_training:
                pbar.set_description_str(f'Jump MCMC (training)')
                x_train, x_val = train_val_split(
                    xs,
                    train_pct=self.params.train_pct,
                    max_train_size=self.params.max_train_size,
                    max_val_size=self.params.max_val_size
                )
                self.kernel.flow.fit(x_train=x_train, x_val=x_val, **self.params.flow_fit_kwargs)

            # Jump
            pbar.set_description_str(f'Jump MCMC (jumping)')
            x_prime, f_x_prime = self.kernel.flow.sample(n_chains, return_log_prob=True)
            x_prime = x_prime.detach()
            f_x_prime = f_x_prime.detach()

            x = mcmc_output.samples[-1]
            if self.params.adjusted_jumps:
                try:
                    u_x = self.target(x)
                    statistics.n_target_calls += n_chains

                    u_x_prime = self.target(x_prime)
                    statistics.n_target_calls += n_chains

                    f_x = self.kernel.flow.log_prob(x)
                    log_alpha = metropolis_acceptance_log_ratio(
                        log_prob_curr=-u_x,
                        log_prob_prime=-u_x_prime,
                        log_proposal_curr=f_x,
                        log_proposal_prime=f_x_prime
                    )
                    acceptance_mask = torch.rand_like(log_alpha).log() < log_alpha
                except ValueError:
                    acceptance_mask = torch.zeros(size=x.shape[:-len(event_shape)], dtype=torch.bool)
            else:
                acceptance_mask = torch.ones(size=x.shape[:-len(event_shape)], dtype=torch.bool)
            x[acceptance_mask] = x_prime[acceptance_mask]
            t1 = time.time()

            statistics.elapsed_time_seconds += t1 - t0
            statistics.n_attempted_jumps += n_chains
            statistics.n_accepted_jumps += int(torch.sum(acceptance_mask))
            pbar.set_postfix_str(f'{statistics}')

            if i % thinning == 0:
                xs[data_index, -1] = x
                data_index += 1
        xs = xs.flatten(0, 1)
        return MCMCOutput(
            samples=xs,
            statistics=statistics,
            kernel=self.kernel
        )


class JumpHMC(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = HMC(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpUHMC(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = UHMC(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpMALA(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = MALA(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpULA(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = ULA(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpMH(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = MH(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpESS(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 negative_log_likelihood: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = ESS(event_shape, target, negative_log_likelihood, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpNUTS(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = NUTS(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)
