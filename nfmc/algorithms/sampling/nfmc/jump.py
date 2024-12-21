import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, Tuple, Optional

import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCOutput, NFMCParameters, NFMCKernel, MCMCStatistics, MCMCKernel, \
    MCMCParameters
from tqdm import tqdm

from nfmc.algorithms.sampling.mcmc.hmc import HMC, UHMC
from nfmc.algorithms.sampling.mcmc.langevin import MALA, ULA
from nfmc.algorithms.sampling.mcmc.mh import MH
from nfmc.algorithms.sampling.mcmc.nuts import NUTS
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

    @property
    def jump_acceptance_rate(self):
        if self.n_attempted_jumps == 0:
            return torch.nan
        return self.n_accepted_jumps / self.n_attempted_jumps

    def update_counters(self,
                        n_accepted_jumps: int = 0,
                        n_attempted_jumps: int = 0,
                        **kwargs):
        super().update_counters(**kwargs)
        self.n_accepted_jumps = int(self.n_accepted_jumps + n_accepted_jumps)
        self.n_attempted_jumps = int(self.n_attempted_jumps + n_attempted_jumps)

    def __repr__(self):
        return (
            f"MCMC acc-rate: {self.acceptance_rate:.2f}, "
            f"Jump acc-rate: {self.jump_acceptance_rate:.2f}, "
            f"kcalls/s: {self.calls_per_second / 1000:.2f}, "
            f"kgrads/s: {self.grads_per_second / 1000:.2f}, "
            f"divergences: {self.n_divergences}"
        )

    def __dict__(self):
        return {
            **super().__dict__(),
            **{'jump_acceptance_rate': self.jump_acceptance_rate}
        }


@dataclass
class JumpNFMCOutput(MCMCOutput):
    statistics: Optional[JumpNFMCStatistics] = None

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size], *args, **kwargs):
        super().__init__(event_shape, *args, **{**kwargs, **dict(statistics=JumpNFMCStatistics(event_shape))})


class JumpNFMC(Sampler):
    """

    Requires flow with an efficient inverse method.
    Requires flow with an efficient forward method if using adjusted jumps (default: True). This makes masked autoregressive flows unsuitable.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
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

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        self.kernel: NFMCKernel
        self.params: JumpNFMCParameters

        if time_limit_seconds is not None:
            inner_sampler_warmup_time_limit = 0.7 * time_limit_seconds
        else:
            inner_sampler_warmup_time_limit = None

        t0 = time.time()
        self.inner_sampler.params.store_samples = True
        warmup_output = self.inner_sampler.warmup(
            x0,
            show_progress=show_progress,
            time_limit_seconds=inner_sampler_warmup_time_limit
        )

        x_train, x_val = train_val_split(
            warmup_output.samples,
            train_pct=self.params.train_pct,
            max_train_size=self.params.max_train_size,
            max_val_size=self.params.max_val_size
        )
        flow_params = deepcopy(self.kernel.flow.state_dict())
        elapsed_time = time.time() - t0

        if time_limit_seconds is not None:
            flow_fit_time_limit = time_limit_seconds - elapsed_time
        else:
            flow_fit_time_limit = None

        try:
            self.kernel.flow.fit(
                x_train=x_train,
                x_val=x_val,
                **{
                    **self.params.flow_fit_kwargs,
                    **dict(
                        show_progress=show_progress,
                        time_limit_seconds=flow_fit_time_limit
                    )
                }
            )
        except ValueError:
            self.kernel.flow.load_state_dict(flow_params)

        # Prefer initialization to MCMC samples because unadjusted flow sampling can generate outliers
        return warmup_output

    def sample(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MCMCOutput:
        self.kernel: NFMCKernel
        self.params: JumpNFMCParameters

        if not self.inner_sampler.params.store_samples:
            raise ValueError("Inner sampler in jump HMC must store samples")

        n_chains, *event_shape = x0.shape
        event_shape = tuple(event_shape)

        out = JumpNFMCOutput(event_shape=x0.shape[1:], store_samples=self.params.store_samples)

        x = torch.clone(x0)

        for i in (pbar := tqdm(range(self.params.n_iterations), desc='Jump MCMC', disable=not show_progress)):
            if time_limit_seconds is not None and out.statistics.elapsed_time_seconds >= time_limit_seconds:
                break
            # Trajectories
            pbar.set_description_str(f'Jump MCMC')
            mcmc_output = self.inner_sampler.sample(x0=x, show_progress=False)

            out.statistics.update_counters(
                n_accepted_trajectories=mcmc_output.statistics.n_accepted_trajectories,
                n_attempted_trajectories=mcmc_output.statistics.n_attempted_trajectories,
                n_divergences=mcmc_output.statistics.n_divergences,
                n_target_calls=mcmc_output.statistics.n_target_calls,
                n_target_gradient_calls=mcmc_output.statistics.n_target_gradient_calls,
            )
            out.statistics.update_elapsed_time(mcmc_output.statistics.elapsed_time_seconds)
            out.statistics.expectations.update(mcmc_output.samples)
            out.running_samples.add(mcmc_output.samples)

            t0 = time.time()
            # Fit flow
            if self.params.fit_nf and i >= self.params.n_jumps_before_training:
                pbar.set_description_str(f'Jump MCMC (training)')
                x_train, x_val = train_val_split(
                    mcmc_output.samples,
                    train_pct=self.params.train_pct,
                    max_train_size=self.params.max_train_size,
                    max_val_size=self.params.max_val_size
                )
                self.kernel.flow.fit(x_train=x_train, x_val=x_val, **self.params.flow_fit_kwargs)

            # Jump
            pbar.set_description_str(f'Jump MCMC')
            x_prime, f_x_prime = self.kernel.flow.sample(n_chains, return_log_prob=True)
            x_prime = x_prime.detach()
            f_x_prime = f_x_prime.detach()

            x = mcmc_output.running_samples[-1]
            if self.params.adjusted_jumps:
                try:
                    u_x = self.target(x)
                    u_x_prime = self.target(x_prime)
                    out.statistics.update_counters(
                        n_target_calls=2 * n_chains,
                    )

                    f_x = self.kernel.flow.log_prob(x)
                    log_alpha = metropolis_acceptance_log_ratio(
                        log_prob_target_curr=-u_x.cpu(),
                        log_prob_target_prime=-u_x_prime.cpu(),
                        log_prob_proposal_curr=f_x.cpu(),
                        log_prob_proposal_prime=f_x_prime.cpu()
                    )
                    mask = torch.rand_like(log_alpha).log() < log_alpha
                except ValueError:
                    mask = torch.zeros(size=x.shape[:-len(event_shape)], dtype=torch.bool)
            else:
                mask = torch.ones(size=x.shape[:-len(event_shape)], dtype=torch.bool)

            x[mask] = x_prime[mask].to(x)
            t1 = time.time()

            # Update output
            out.statistics.update_elapsed_time(t1 - t0)
            out.statistics.update_counters(
                n_attempted_jumps=n_chains,
                n_accepted_jumps=int(torch.sum(mask)),
            )
            out.statistics.expectations.update(x)
            pbar.set_postfix_str(f'{out.statistics}')

            out.running_samples.add(x)

        out.kernel = self.kernel
        return out


class JumpHMC(JumpNFMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = HMC(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpUHMC(JumpNFMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = UHMC(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpMALA(JumpNFMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = MALA(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpULA(JumpNFMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = ULA(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpMH(JumpNFMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = MH(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpESS(JumpNFMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
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
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = NUTS(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)
